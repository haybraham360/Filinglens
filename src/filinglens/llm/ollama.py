"""Local LLM via Ollama (free): structured outputs through ``/api/chat`` with a JSON schema.

Implements :class:`~filinglens.llm.provider.StructuredLLM`, so it drops into the RAG
pipeline, the router agent's answer writer, the eval judge and the drafter. Small local
models are much weaker than Claude; FilingLens never relies on them for numbers (those
come from tools) and still verifies every citation.
"""

from __future__ import annotations

import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from filinglens.llm.calllog import CallLog, CallRecord
from filinglens.llm.pricing import LOCAL_PREFIX
from filinglens.llm.provider import LLMResult

T = TypeVar("T", bound=BaseModel)


class OllamaUnavailableError(RuntimeError):
    """Ollama isn't running, or the model hasn't been pulled."""


def ollama_available(url: str, model: str, timeout: float = 1.5) -> bool:
    """True if the Ollama server answers and ``model`` is pulled."""
    try:
        resp = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except (httpx.HTTPError, ValueError):
        return False
    names = {m.get("name", "") for m in resp.json().get("models", [])}
    return model in names or f"{model}:latest" in names


class OllamaLLM:
    """Structured generation with a local Ollama model."""

    def __init__(
        self,
        model: str,
        call_log: CallLog,
        *,
        url: str = "http://localhost:11434",
        timeout: float = 300.0,
        num_ctx: int = 8192,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """``transport`` is for tests (``httpx.MockTransport``)."""
        self.model = model
        self.served_name = f"{LOCAL_PREFIX}{model}"
        self.num_ctx = num_ctx
        self._log = call_log
        self._http = httpx.Client(base_url=url.rstrip("/"), timeout=timeout, transport=transport)

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
        """One chat request constrained to ``schema``; ``parsed`` is None if it doesn't validate."""
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": schema.model_json_schema(),
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        start = time.perf_counter()
        try:
            resp = self._http.post("/api/chat", json=body)
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                "Ollama isn't running. Start it with `ollama serve` (or the Ollama app)."
            ) from exc
        if resp.status_code == 404:
            raise OllamaUnavailableError(
                f"Model {self.model!r} isn't pulled. Run `ollama pull {self.model}`."
            )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = int((time.perf_counter() - start) * 1000)
        stop = "max_tokens" if data.get("done_reason") == "length" else "end_turn"
        parsed: T | None = None
        if stop == "end_turn":
            try:
                parsed = schema.model_validate_json(data.get("message", {}).get("content", ""))
            except ValidationError:
                stop = "invalid_output"
        input_tokens = int(data.get("prompt_eval_count") or 0)
        output_tokens = int(data.get("eval_count") or 0)
        self._log.record(
            CallRecord(
                model=self.served_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=0.0,
                purpose=purpose,
                prompt_version=prompt_version,
                stop_reason=stop,
                trace_id=trace_id,
            )
        )
        return LLMResult(
            parsed=parsed,
            model=self.served_name,
            stop_reason=stop,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=0.0,
        )
