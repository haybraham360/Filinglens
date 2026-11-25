"""Hand-written tool-use loop on the Messages API (specs/04).

model -> tool calls -> results -> model, at most ``max_steps`` model requests. The
history is append-only: every response's full ``content`` (thinking blocks included) is
sent back unchanged, all tool results for a turn go back in one user message, and the
tool list and system prompt never change mid-run. On the last step the model is told to
answer and ``tool_choice`` is set to ``none``.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
from pydantic import ValidationError

from filinglens.agent.guards import unsupported_numbers
from filinglens.agent.sources import SourceRegistry
from filinglens.agent.tools import TOOLS, AgentContext, run_tool
from filinglens.llm.calllog import CallLog, CallRecord, SpendCapExceededError
from filinglens.llm.pricing import cost_usd
from filinglens.llm.prompts import Prompt, load_prompt
from filinglens.llm.provider import FALLBACK_BETA, FALLBACK_MODELS
from filinglens.rag.answer import filing_index_url
from filinglens.rag.citations import QUOTE_THRESHOLD, quote_match_score
from filinglens.rag.schemas import Answer, AnswerDraft, Citation

TOOL_TIMEOUT_S = 20.0
_MARKER = re.compile(r"\[([CPT]\d+)\]")
FINAL_NUDGE = (
    "You have reached the tool-call limit for this question. Answer now, using only the "
    "results above, in the required JSON format."
)


@dataclass
class Step:
    """One traced step: an LLM request or a tool call."""

    index: int
    kind: str  # "llm" | "tool"
    name: str
    latency_ms: int
    args: dict[str, Any] | None = None
    is_error: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    result_preview: str | None = None


@dataclass
class AgentResult:
    """The final answer and the full trace."""

    answer: Answer
    trace_id: str
    steps: list[Step] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: int = 0
    model: str | None = None


def _insufficient(message: str) -> Answer:
    return Answer(answer=message, citations=[], insufficient_context=True, confidence="low")


def resolve_agent_citations(
    draft: AnswerDraft, registry: SourceRegistry, threshold: float = QUOTE_THRESHOLD
) -> Answer:
    """Map C#/P#/T# handles to sources, verify each quote, renumber markers [1], [2], ..."""
    order: list[str] = []
    for ref in _MARKER.findall(draft.answer) + [
        c.ref.strip("[] ").upper() for c in draft.citations
    ]:
        if registry.get(ref) is not None and ref not in order:
            order.append(ref)
    number = {ref: i + 1 for i, ref in enumerate(order)}

    citations: list[Citation] = []
    for c in draft.citations:
        ref = c.ref.strip("[] ").upper()
        src = registry.get(ref)
        if src is None:
            continue
        score = quote_match_score(c.quote, src.text)
        citations.append(
            Citation(
                n=number[ref],
                source=src.kind,
                chunk_id=src.chunk_id,
                accession_no=src.accession_no,
                item=src.item,
                tool_name=src.tool_name,
                char_start=src.char_start,
                char_end=src.char_end,
                ticker=src.ticker,
                form=src.form,
                fiscal_year=src.fiscal_year,
                quote=c.quote,
                verified=score >= threshold,
                match_score=round(score, 3),
                filing_url=filing_index_url(src.cik, src.accession_no)
                if src.cik and src.accession_no
                else None,
            )
        )
    citations.sort(key=lambda c: (c.n, c.chunk_id or ""))
    text = _MARKER.sub(
        lambda m: f"[{number[m.group(1)]}]" if m.group(1) in number else "", draft.answer
    )
    text = re.sub(r"\s+([.,;:])", r"\1", text).strip()
    validity = (sum(c.verified for c in citations) / len(citations)) if citations else None
    return Answer(
        answer=text,
        citations=citations,
        insufficient_context=draft.insufficient_context,
        confidence=draft.confidence,
        citation_validity=validity,
    )


class Agent:
    """Runs one question through the tool loop."""

    def __init__(
        self,
        client: Any,
        model: str,
        call_log: CallLog,
        context_factory: Any,
        *,
        effort: str = "medium",
        max_steps: int = 6,
        max_tokens: int = 16000,
        daily_cap_usd: float | None = None,
        prompt: Prompt | None = None,
        trace_log: Path | None = None,
    ) -> None:
        """``context_factory(registry) -> AgentContext`` builds per-run tool state."""
        self.client = client
        self.model = model
        self.call_log = call_log
        self.context_factory = context_factory
        self.effort = effort
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.daily_cap_usd = daily_cap_usd
        self.prompt = prompt or load_prompt("agent")
        self.trace_log = trace_log
        self.tool_defs = [t.definition() for t in TOOLS]
        self.output_format = {
            "type": "json_schema",
            "schema": anthropic.transform_schema(AnswerDraft),
        }

    def _request(self, messages: list[dict[str, Any]], final: bool) -> Any:
        kwargs: dict[str, Any] = {}
        if self.model in FALLBACK_MODELS:
            kwargs = {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        if final:
            kwargs["tool_choice"] = {"type": "none"}
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.prompt.text,
            tools=self.tool_defs,
            messages=messages,
            output_config={"effort": self.effort, "format": self.output_format},
            **kwargs,
        )

    def _log_llm(self, response: Any, latency_ms: int, trace_id: str) -> float:
        usage = response.usage
        served = str(response.model)
        cost = cost_usd(
            served,
            int(usage.input_tokens or 0),
            int(usage.output_tokens or 0),
            int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        )
        self.call_log.record(
            CallRecord(
                model=served,
                input_tokens=int(usage.input_tokens or 0),
                output_tokens=int(usage.output_tokens or 0),
                latency_ms=latency_ms,
                cost_usd=cost,
                purpose="agent_step",
                prompt_version=self.prompt.id,
                stop_reason=response.stop_reason,
                trace_id=trace_id,
            )
        )
        return cost or 0.0

    def _run_tools(
        self, ctx: AgentContext, blocks: list[Any], steps: list[Step], pool: ThreadPoolExecutor
    ) -> list[dict[str, Any]]:
        """Run this turn's tool calls concurrently; return their tool_result blocks in order."""
        started = time.perf_counter()
        futures = [(b, pool.submit(run_tool, ctx, b.name, b.input)) for b in blocks]
        results = []
        for block, future in futures:
            remaining = max(0.1, TOOL_TIMEOUT_S - (time.perf_counter() - started))
            t0 = time.perf_counter()
            try:
                output, is_error = future.result(timeout=remaining)
            except FutureTimeout:
                output, is_error = {"error": f"Tool timed out after {TOOL_TIMEOUT_S:.0f}s."}, True
            except Exception as exc:  # unexpected failure inside a tool (e.g. DB error)
                output, is_error = {"error": f"{type(exc).__name__}: {exc}"}, True
            content = json.dumps(output, default=str)
            steps.append(
                Step(
                    index=len(steps),
                    kind="tool",
                    name=block.name,
                    args=dict(block.input) if isinstance(block.input, dict) else None,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    is_error=is_error,
                    result_preview=content[:300],
                )
            )
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            }
            if is_error:
                result["is_error"] = True
            results.append(result)
        return results

    def run(self, question: str) -> AgentResult:
        """Answer ``question`` with tools; returns the answer and trace."""
        trace_id = uuid.uuid4().hex
        registry = SourceRegistry()
        ctx = self.context_factory(registry)
        steps: list[Step] = []
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        total_cost = 0.0
        start = time.perf_counter()
        answer: Answer | None = None
        model_used: str | None = None

        # Not a `with` block: shutting down must not wait for a tool that timed out.
        pool = ThreadPoolExecutor(max_workers=4)
        try:
            for step in range(self.max_steps):
                if (
                    self.daily_cap_usd is not None
                    and self.call_log.spent_today_usd() >= self.daily_cap_usd
                ):
                    raise SpendCapExceededError(
                        f"Daily LLM spend cap of ${self.daily_cap_usd:.2f} reached."
                    )
                final = step == self.max_steps - 1
                t0 = time.perf_counter()
                response = self._request(messages, final)
                latency = int((time.perf_counter() - t0) * 1000)
                total_cost += self._log_llm(response, latency, trace_id)
                model_used = str(response.model)
                steps.append(
                    Step(
                        index=len(steps),
                        kind="llm",
                        name=model_used,
                        latency_ms=latency,
                        input_tokens=int(response.usage.input_tokens or 0),
                        output_tokens=int(response.usage.output_tokens or 0),
                        stop_reason=response.stop_reason,
                    )
                )
                stop = response.stop_reason
                if stop == "refusal":
                    answer = _insufficient("No answer was produced: the model declined.")
                    break
                if stop == "max_tokens":
                    answer = _insufficient("No answer was produced: the response was cut off.")
                    break
                messages.append({"role": "assistant", "content": response.content})
                if stop == "pause_turn":
                    continue
                tool_blocks = [b for b in response.content if b.type == "tool_use"]
                if stop == "tool_use" and tool_blocks:
                    results: list[dict[str, Any]] = self._run_tools(ctx, tool_blocks, steps, pool)
                    if step == self.max_steps - 2:
                        results.append({"type": "text", "text": FINAL_NUDGE})
                    messages.append({"role": "user", "content": results})
                    continue
                answer = self._final_answer(response, registry)
                break
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        if answer is None:
            answer = _insufficient("No answer was produced within the step limit.")
        sources = [s.text for s in registry.sources.values()]
        result = AgentResult(
            answer=answer,
            trace_id=trace_id,
            steps=steps,
            tools_called=[s.name for s in steps if s.kind == "tool"],
            unsupported_numbers=unsupported_numbers(answer.answer, sources),
            cost_usd=round(total_cost, 6),
            latency_ms=int((time.perf_counter() - start) * 1000),
            model=model_used,
        )
        if self.trace_log is not None:
            self._write_trace(question, result)
        return result

    def _final_answer(self, response: Any, registry: SourceRegistry) -> Answer:
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        try:
            draft = AnswerDraft.model_validate_json(text)
        except ValidationError:
            return _insufficient("The model's final answer could not be parsed.")
        return resolve_agent_citations(draft, registry)

    def _write_trace(self, question: str, result: AgentResult) -> None:
        assert self.trace_log is not None
        self.trace_log.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "trace_id": result.trace_id,
            "question": question,
            "model": result.model,
            "tools_called": result.tools_called,
            "steps": [asdict(s) for s in result.steps],
            "unsupported_numbers": result.unsupported_numbers,
            "citation_validity": result.answer.citation_validity,
            "insufficient_context": result.answer.insufficient_context,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
        }
        with self.trace_log.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
