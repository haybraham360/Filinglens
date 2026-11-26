"""LLM-as-judge for answer correctness and faithfulness (specs/05).

One property per call (atomic checks), structured outputs for a deterministic parse,
candidate text wrapped as untrusted data, and an instruction not to reward length. The
judge model must differ from the model under test; the default is ``claude-opus-5-5``.
The server-side refusal fallback is deliberately *off* here: a judge that silently
switched model mid-run would make grades incomparable, so a refusal is a judge error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from filinglens.llm.prompts import load_prompt

JUDGE_PROMPT_VERSION = "v1"


class CorrectnessVerdict(BaseModel):
    """Structured correctness grade."""

    reasoning: str = Field(description="One or two sentences comparing key facts.")
    verdict: Literal["correct", "partially_correct", "incorrect"]


class Claim(BaseModel):
    """One factual claim and whether its citations support it."""

    claim: str
    supported: bool


class FaithfulnessVerdict(BaseModel):
    """Structured faithfulness check."""

    claims: list[Claim]


VERDICT_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


class JudgeError(RuntimeError):
    """The judge failed to produce a usable grade (refusal, truncation, model mismatch)."""


@dataclass
class JudgeCall:
    """One judge request, kept for traces and cost accounting."""

    property: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    output: dict[str, Any]
    user: str


class Judge:
    """Correctness and faithfulness grading with Claude."""

    def __init__(
        self,
        client: Any,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 8000,
        llm: Any = None,
    ) -> None:
        """``client`` is an ``anthropic.Anthropic`` (or a fake). Pass ``llm`` (a local
        :class:`~filinglens.llm.ollama.OllamaLLM`) instead to judge for free; small local
        judges are much less reliable, so check them with ``evals.label`` first."""
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.llm = llm

    def _call(
        self, prop: str, system: str, user: str, schema: type[BaseModel]
    ) -> tuple[Any, JudgeCall]:
        if self.llm is not None:
            res = self.llm.generate(
                system=system,
                user=user,
                schema=schema,
                purpose=f"judge_{prop}",
                prompt_version=JUDGE_PROMPT_VERSION,
            )
            if res.parsed is None:
                raise JudgeError(f"local judge produced no verdict ({res.stop_reason})")
            return res.parsed, JudgeCall(
                prop,
                res.model,
                res.input_tokens,
                res.output_tokens,
                res.latency_ms,
                res.parsed.model_dump(),
                user,
            )
        t0 = time.perf_counter()
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": self.effort},
        )
        latency = int((time.perf_counter() - t0) * 1000)
        if str(response.model) != self.model:
            raise JudgeError(f"judge served by {response.model}, expected {self.model}")
        if response.stop_reason in ("refusal", "max_tokens") or response.parsed_output is None:
            raise JudgeError(f"judge produced no verdict (stop_reason={response.stop_reason})")
        parsed = response.parsed_output
        call = JudgeCall(
            property=prop,
            model=str(response.model),
            input_tokens=int(response.usage.input_tokens or 0),
            output_tokens=int(response.usage.output_tokens or 0),
            latency_ms=latency,
            output=parsed.model_dump(),
            user=user,
        )
        return parsed, call

    def correctness(self, question: str, reference: str, candidate: str) -> tuple[float, JudgeCall]:
        """Score 1 / 0.5 / 0 for correct / partially correct / incorrect."""
        user = (
            f"<question>\n{question}\n</question>\n\n"
            f"<reference_answer>\n{reference}\n</reference_answer>\n\n"
            f"<candidate_answer>\n{candidate or '(empty response)'}\n</candidate_answer>"
        )
        verdict, call = self._call(
            "correctness",
            load_prompt("judge_correctness", JUDGE_PROMPT_VERSION).text,
            user,
            CorrectnessVerdict,
        )
        return VERDICT_SCORE[verdict.verdict], call

    def faithfulness(
        self, answer: str, sources: list[tuple[int, str]]
    ) -> tuple[float | None, JudgeCall]:
        """Share of claims supported by cited sources; ``None`` if the answer makes no claims."""
        rendered = "\n\n".join(f'<source n="{n}">\n{text}\n</source>' for n, text in sources)
        user = (
            f"<answer>\n{answer}\n</answer>\n\n"
            f"<cited_sources>\n{rendered or '(no sources cited)'}\n</cited_sources>"
        )
        verdict, call = self._call(
            "faithfulness",
            load_prompt("judge_faithfulness", JUDGE_PROMPT_VERSION).text,
            user,
            FaithfulnessVerdict,
        )
        if not verdict.claims:
            return None, call
        return sum(c.supported for c in verdict.claims) / len(verdict.claims), call


KNOWN_NEGATIVES = (
    ("empty", ""),
    ("dont_know", "I don't know."),
    (
        "wrong_question",
        "Microsoft's largest segment by revenue in fiscal 2023 was Intelligent "
        "Cloud, which grew 15% year over year.",
    ),
)


def judge_selfcheck(judge: Judge) -> dict[str, float]:
    """Known negatives must all grade "incorrect" (paid: 3 judge calls)."""
    question = "What was Apple's total net sales in fiscal 2023?"
    reference = "Apple's total net sales were $383.3 billion in fiscal 2023."
    return {name: judge.correctness(question, reference, text)[0] for name, text in KNOWN_NEGATIVES}
