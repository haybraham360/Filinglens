"""Rule-based agent for when no tool-calling LLM is available (local model or none).

The question is parsed with the same filters as retrieval plus keyword rules, which pick
the tools deterministically (``plan``). Tools run exactly as in the Claude agent. The answer
is then either templated from the tool results (no LLM: every number is a verbatim tool
value) or written by a small local model from those results in one structured call. Small
models are much better at writing from given facts than at choosing tools, which is why
only Claude plans its own tool calls.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from filinglens.agent.guards import unsupported_numbers
from filinglens.agent.loop import AgentResult, Step, resolve_agent_citations
from filinglens.agent.sources import SourceRegistry
from filinglens.agent.tools import AgentContext, run_tool
from filinglens.llm.prompts import load_prompt
from filinglens.llm.provider import StructuredLLM
from filinglens.rag.extractive import best_sentences
from filinglens.rag.schemas import Answer, AnswerDraft, DraftCitation
from filinglens.retrieval.filters import QueryFilters, parse_filters

# Longest phrases first, so "revenue growth" wins over "revenue".
RATIO_PHRASES: tuple[tuple[str, str], ...] = (
    ("operating cash flow margin", "operating_cash_flow_margin"),
    ("long-term debt to equity", "long_term_debt_to_equity"),
    ("net income growth", "net_income_growth_yoy"),
    ("return on equity", "roe"),
    ("return on assets", "roa"),
    ("debt-to-equity", "debt_to_equity"),
    ("debt to equity", "debt_to_equity"),
    ("operating margin", "operating_margin"),
    ("revenue growth", "revenue_growth_yoy"),
    ("sales growth", "revenue_growth_yoy"),
    ("current ratio", "current_ratio"),
    ("profit margin", "net_margin"),
    ("gross margin", "gross_margin"),
    ("net margin", "net_margin"),
    ("grew", "revenue_growth_yoy"),
    ("roe", "roe"),
    ("roa", "roa"),
)
CONCEPT_PHRASES: tuple[tuple[str, str], ...] = (
    ("operating cash flow", "operating_cash_flow"),
    ("cash from operations", "operating_cash_flow"),
    ("stockholders' equity", "stockholders_equity"),
    ("shareholders' equity", "stockholders_equity"),
    ("operating income", "operating_income"),
    ("long-term debt", "long_term_debt"),
    ("total assets", "total_assets"),
    ("gross profit", "gross_profit"),
    ("net income", "net_income"),
    ("net sales", "revenue"),
    ("revenue", "revenue"),
    ("earnings", "net_income"),
)
RATIO_LABELS = {r: r.replace("_yoy", " (YoY)").replace("_", " ") for _, r in RATIO_PHRASES}
_CHANGE = re.compile(
    r"\b(new|added|add|removed|dropped|deleted|changed|change[sd]?\s+(?:in|to|from)|"
    r"differ\w*|compared?\s+(?:to|with)|versus|vs\.?)\b",
    re.I,
)
_RISKY = re.compile(r"\b(risk|risks|risk factors?|item\s*1a|disclosure|language)\b", re.I)
_EXPLAIN = re.compile(
    r"\b(why|drove|driv\w+|explain\w*|reason\w*|cause[sd]?|attribut\w+|management\s+say\w*|"
    r"what\s+does\s+\w+\s+say)\b",
    re.I,
)
MAX_YEARS = 6


@dataclass(frozen=True)
class PlannedCall:
    """One tool call chosen by the rules."""

    name: str
    args: dict[str, Any]


def _years(filters: QueryFilters, latest: int | None) -> list[int]:
    ys = sorted(filters.fiscal_years)
    if len(ys) >= 2:
        return list(range(ys[0], ys[-1] + 1))[-MAX_YEARS:]
    if ys:
        return ys
    return [latest] if latest else []


def plan(
    question: str, filters: QueryFilters, latest_year: Callable[[], int | None]
) -> list[PlannedCall]:
    """Choose tool calls for ``question`` (deterministic)."""
    q = question.lower()
    ticker = filters.tickers[0] if filters.tickers else None
    calls: list[PlannedCall] = []
    ratio = next(
        (r for phrase, r in RATIO_PHRASES if re.search(rf"\b{re.escape(phrase)}\b", q)), None
    )
    concepts: list[str] = []
    if ratio is None:
        for phrase, concept in CONCEPT_PHRASES:
            if re.search(rf"\b{re.escape(phrase)}\b", q) and concept not in concepts:
                concepts.append(concept)
    numeric = bool(ratio or concepts)
    explain = bool(_EXPLAIN.search(q))
    change = bool(_CHANGE.search(q)) and (bool(_RISKY.search(q)) or not numeric)

    if ticker and change and not numeric:
        item = filters.items[0] if filters.items else ("7" if "md&a" in q else "1A")
        args: dict[str, Any] = {"ticker": ticker, "item": item}
        ys = sorted(filters.fiscal_years)
        if ys:
            args["year_b"] = ys[-1]
        if len(ys) >= 2:
            args["year_a"] = ys[0]
        calls.append(PlannedCall("compare_sections", args))
    if ticker and numeric:
        years = _years(filters, latest_year())
        if years:
            if ratio:
                calls.append(
                    PlannedCall(
                        "compute_ratio", {"ticker": ticker, "ratio": ratio, "fiscal_years": years}
                    )
                )
            else:
                calls.append(
                    PlannedCall(
                        "get_financials",
                        {"ticker": ticker, "concepts": concepts, "fiscal_years": years},
                    )
                )
    if explain or not calls:
        search: dict[str, Any] = {"query": question, "top_k": 6}
        if ticker:
            search["ticker"] = ticker
        if filters.fiscal_years:
            search["fiscal_year"] = max(filters.fiscal_years)
        if numeric and explain:
            search["item"] = "7"
        elif filters.items:
            search["item"] = filters.items[0]
        if len(filters.forms) == 1:
            search["form"] = filters.forms[0]
        calls.append(PlannedCall("search_filings", search))
    return calls


def _money(value: Decimal, unit: str) -> str:
    v = float(value)
    if unit == "USD":
        if abs(v) >= 1e9:
            return f"${v / 1e9:,.2f} billion"
        if abs(v) >= 1e6:
            return f"${v / 1e6:,.1f} million"
        return f"${v:,.0f}"
    if unit == "USD/shares":
        return f"${v:,.2f} per share"
    return f"{v:,.0f} {unit}"


def template_answer(
    question: str, filters: QueryFilters, results: list[tuple[PlannedCall, dict[str, Any], bool]]
) -> AnswerDraft:
    """Write the answer from tool results with no LLM; numbers are quoted from the tools."""
    lines: list[str] = []
    cites: list[DraftCitation] = []
    ok = False
    for call, out, is_error in results:
        if is_error:
            lines.append(f"Couldn't run {call.name}: {out.get('error')}")
            continue
        ref = out.get("ref")
        if call.name == "compute_ratio" and ref:
            parts = []
            for year, raw in out["values_by_year"].items():
                if raw is None:
                    parts.append(f"FY{year} not available")
                    continue
                parts.append(f"FY{year} {float(raw) * 100:.1f}%")
                cites.append(DraftCitation(ref=ref, quote=raw))
                ok = True
            label = RATIO_LABELS.get(out["ratio"], out["ratio"])
            lines.append(f"{out['ticker']} {label} ({out['formula']}): {'; '.join(parts)} [{ref}]")
        elif call.name == "get_financials" and ref:
            for v in out["values"]:
                lines.append(
                    f"{out['ticker']} {v['concept'].replace('_', ' ')} FY{v['fiscal_year']}: "
                    f"{_money(Decimal(v['value']), v['unit'])} [{ref}]"
                )
                cites.append(DraftCitation(ref=ref, quote=v["value"]))
                ok = True
            if out["missing"]:
                lines.append("Not available in XBRL facts: " + ", ".join(out["missing"]) + ".")
        elif call.name == "compare_sections" and ref:
            c = out["counts"]
            lines.append(
                f"{out['ticker']} Item {out['item']}, FY{out['filing_b']['fiscal_year']} vs "
                f"FY{out['filing_a']['fiscal_year']}: {c['headings_added']} risk headings added, "
                f"{c['headings_removed']} removed; {c['paragraphs_added']} paragraphs added, "
                f"{c['paragraphs_removed']} removed, {c['paragraphs_modified']} modified [{ref}]"
            )
            cites.append(DraftCitation(ref=ref, quote=f'"headings_added": {c["headings_added"]}'))
            for label, key in (
                ("Added", "added_risk_headings"),
                ("Removed", "removed_risk_headings"),
            ):
                for h in out[key][:8]:
                    lines.append(f"- {label}: {h['text']} [{h['ref']}]")
                    cites.append(DraftCitation(ref=h["ref"], quote=h["text"]))
            ok = True
        elif call.name == "search_filings":
            texts = [r["text"] for r in out["results"]]
            for idx, sentence in best_sentences(question, texts, filters):
                r = out["results"][idx]
                lines.append(
                    f"- {r['ticker']} {r['form']} FY{r['fiscal_year']} Item {r['item']}: "
                    f"{sentence} [{r['ref']}]"
                )
                cites.append(DraftCitation(ref=r["ref"], quote=sentence))
                ok = True
    if not ok:
        lines.append("The filings and XBRL data available don't answer this question.")
    return AnswerDraft(
        answer="\n".join(lines),
        citations=cites,
        insufficient_context=not ok,
        confidence="medium" if ok else "low",
    )


class RouterAgent:
    """Same inputs and outputs as :class:`~filinglens.agent.loop.Agent`, no tool-calling LLM."""

    def __init__(
        self,
        context_factory: Callable[[SourceRegistry], AgentContext],
        llm: StructuredLLM | None = None,
        *,
        trace_log: Any = None,
    ) -> None:
        """``llm`` (e.g. a local Ollama model) only writes the answer; ``None`` = template."""
        self.context_factory = context_factory
        self.llm = llm
        self.prompt = load_prompt("router_writer")
        self.trace_log = trace_log

    def run(self, question: str, on_step: Callable[[Step], None] | None = None) -> AgentResult:
        """Plan with rules, run the tools, write the answer (``on_step`` reports progress)."""
        trace_id = uuid.uuid4().hex
        start = time.perf_counter()
        registry = SourceRegistry()
        ctx = self.context_factory(registry)
        filters = parse_filters(question, ctx.directory)

        def latest_year() -> int | None:
            if not filters.ciks:
                return None
            years = ctx.sections.available_years(filters.ciks[0], "7")
            return years[-1] if years else None

        from filinglens.agent.loop import _NotifyingList

        steps: list[Step] = _NotifyingList(on_step)
        results: list[tuple[PlannedCall, dict[str, Any], bool]] = []
        for call in plan(question, filters, latest_year):
            t0 = time.perf_counter()
            try:
                out, is_error = run_tool(ctx, call.name, call.args)
            except Exception as exc:  # unexpected tool failure (e.g. database)
                out, is_error = {"error": f"{type(exc).__name__}: {exc}"}, True
            steps.append(
                Step(
                    index=len(steps),
                    kind="tool",
                    name=call.name,
                    args=call.args,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    is_error=is_error,
                    result_preview=json.dumps(out, default=str)[:300],
                )
            )
            results.append((call, out, is_error))

        draft = template_answer(question, filters, results)
        model = "router"
        in_tok = out_tok = 0
        if self.llm is not None and not draft.insufficient_context:
            payload = [{"tool": c.name, "result": o} for c, o, e in results if not e]
            llm_result = self.llm.generate(
                system=self.prompt.text,
                user=f"<question>\n{question}\n</question>\n\n<tool_results>\n"
                f"{json.dumps(payload, default=str)}\n</tool_results>",
                schema=AnswerDraft,
                purpose="router_writer",
                prompt_version=self.prompt.id,
                trace_id=trace_id,
            )
            steps.append(
                Step(
                    index=len(steps),
                    kind="llm",
                    name=llm_result.model,
                    latency_ms=llm_result.latency_ms,
                    input_tokens=llm_result.input_tokens,
                    output_tokens=llm_result.output_tokens,
                    stop_reason=llm_result.stop_reason,
                )
            )
            model, in_tok, out_tok = (
                llm_result.model,
                llm_result.input_tokens,
                llm_result.output_tokens,
            )
            if llm_result.parsed is not None:
                draft = llm_result.parsed

        cited: dict[int, str] = {}
        answer: Answer = resolve_agent_citations(draft, registry, texts_out=cited)
        if self.llm is not None and (answer.citation_validity or 0) < 0.5 and model != "router":
            # The local model's citations didn't check out: fall back to the template.
            cited = {}
            answer = resolve_agent_citations(
                template_answer(question, filters, results), registry, texts_out=cited
            )
        return AgentResult(
            answer=answer,
            trace_id=trace_id,
            steps=steps,
            tools_called=[s.name for s in steps if s.kind == "tool"],
            unsupported_numbers=unsupported_numbers(
                answer.answer, [s.text for s in registry.sources.values()]
            ),
            latency_ms=int((time.perf_counter() - start) * 1000),
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            final_stop_reason="end_turn",
            cited_texts=cited,
        )
