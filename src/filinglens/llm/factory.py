"""Pick the LLM provider: Claude, a local Ollama model, or none (LLM-free mode)."""

from __future__ import annotations

from typing import Literal

from filinglens.config import Settings
from filinglens.llm.calllog import CallLog
from filinglens.llm.provider import StructuredLLM

Provider = Literal["anthropic", "ollama", "none"]


def resolve_provider(settings: Settings, requested: str | None = None) -> Provider:
    """Resolve ``auto``: Claude if a key is set, else Ollama if reachable, else none."""
    choice = requested or settings.llm_provider
    if choice in ("anthropic", "ollama", "none"):
        return choice  # type: ignore[return-value]
    if settings.anthropic_api_key is not None:
        return "anthropic"
    from filinglens.llm.ollama import ollama_available

    if ollama_available(settings.ollama_url, settings.ollama_model):
        return "ollama"
    return "none"


def make_llm(settings: Settings, provider: Provider, call_log: CallLog) -> StructuredLLM | None:
    """The structured LLM for ``provider`` (``None`` means LLM-free mode)."""
    if provider == "none":
        return None
    if provider == "ollama":
        from filinglens.llm.ollama import OllamaLLM

        return OllamaLLM(settings.ollama_model, call_log, url=settings.ollama_url)
    import anthropic

    from filinglens.llm.provider import AnthropicLLM

    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    return AnthropicLLM(
        settings.llm_model,
        call_log,
        effort=settings.llm_effort,
        max_tokens=settings.llm_max_tokens,
        daily_cap_usd=settings.llm_daily_spend_cap_usd,
        client=anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic(),
    )
