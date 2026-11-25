"""LLM provider interface and the Claude implementation (structured outputs)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

from filinglens.llm.calllog import CallLog, CallRecord, SpendCapExceededError
from filinglens.llm.pricing import cost_usd

T = TypeVar("T", bound=BaseModel)

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
FALLBACK_MODELS = frozenset(
    {
        "claude-fable-5-2",
        "claude-fable-5-1",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
    }
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class LLMResult(Generic[T]):
    """Parsed output plus call metadata. ``parsed`` is ``None`` on refusal or truncation."""

    parsed: T | None
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float | None
    fallback_used: bool = False


class StructuredLLM(Protocol):
    """A model that returns schema-validated output."""

    model: str

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        purpose: str,
        prompt_version: str,
        trace_id: str | None = None,
    ) -> LLMResult[T]:
        """Run one request and return the parsed result."""
        ...


class AnthropicLLM:
    """Claude via the official SDK: ``beta.messages.parse`` with a Pydantic schema.

    Adaptive thinking is the model default and is left on; ``effort`` controls depth.
    The server-side refusal fallback is enabled on models that support it, so a
    classifier decline is retried on a fallback model inside the same call.
    """

    def __init__(
        self,
        model: str,
        call_log: CallLog,
        *,
        effort: str = "medium",
        max_tokens: int = 16000,
        daily_cap_usd: float | None = None,
        client: Any | None = None,
    ) -> None:
        """Create the provider. ``client`` is injectable for tests."""
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.daily_cap_usd = daily_cap_usd
        self._client: Any = client
        self._log = call_log

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        purpose: str,
        prompt_version: str,
        trace_id: str | None = None,
    ) -> LLMResult[T]:
        """Call Claude once, log the call, and return the validated output."""
        if self.daily_cap_usd is not None and self._log.spent_today_usd() >= self.daily_cap_usd:
            raise SpendCapExceededError(
                f"Daily LLM spend cap of ${self.daily_cap_usd:.2f} reached; try again tomorrow."
            )
        kwargs: dict[str, Any] = {}
        if self.model in FALLBACK_MODELS:
            kwargs = {"betas": [FALLBACK_BETA], "fallbacks": "default"}

        start = time.perf_counter()
        response = self._client.beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": self.effort},
            **kwargs,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        usage = response.usage
        served_by = str(response.model)
        input_tokens = int(usage.input_tokens or 0)
        output_tokens = int(usage.output_tokens or 0)
        cost = cost_usd(
            served_by,
            input_tokens,
            output_tokens,
            int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        )
        fallback_used = any(
            getattr(entry, "type", None) == "fallback_message"
            for entry in (getattr(usage, "iterations", None) or [])
        )
        stop = response.stop_reason
        self._log.record(
            CallRecord(
                model=served_by,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                purpose=purpose,
                prompt_version=prompt_version,
                stop_reason=stop,
                trace_id=trace_id,
            )
        )
        parsed = response.parsed_output if stop not in ("refusal", "max_tokens") else None
        return LLMResult(
            parsed=parsed,
            model=served_by,
            stop_reason=stop,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            fallback_used=fallback_used,
        )
