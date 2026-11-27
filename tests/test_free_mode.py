"""LLM-free and local-model paths: extractive answers, Ollama client, router agent."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from filinglens.agent.router import RouterAgent, plan
from filinglens.agent.sources import SourceRegistry
from filinglens.agent.tools import AgentContext
from filinglens.config import Settings
from filinglens.llm.calllog import MemoryCallLog
from filinglens.llm.factory import resolve_provider
from filinglens.llm.ollama import OllamaLLM, OllamaUnavailableError
from filinglens.llm.pricing import cost_usd
from filinglens.llm.provider import LLMResult
from filinglens.rag.answer import answer_question
from filinglens.rag.extractive import EXTRACTIVE_NOTE, extractive_answer
from filinglens.rag.schemas import AnswerDraft, DraftCitation
from filinglens.retrieval.filters import CompanyDirectory, QueryFilters, parse_filters
from tests.test_agent_tools import CIKS, make_ctx
from tests.test_rag import TEXT, FakeRetriever, chunk

# --- extractive answers ---------------------------------------------------------------------


def test_extractive_answer_quotes_relevant_sentences_verbatim() -> None:
    chunks = [
        chunk("a", "The Company sells phones worldwide. " + TEXT),
        chunk("b", "Unrelated text about office leases in Ohio and Texas for many years."),
    ]
    answer = extractive_answer(
        "What does Apple say about limited sources of components?",
        chunks,
        QueryFilters(tickers=("AAPL",)),
    )
    assert answer.answer.startswith(EXTRACTIVE_NOTE)
    assert len(answer.citations) == 1 and answer.citation_validity == 1.0
    quote = answer.citations[0].quote
    assert quote in chunks[0].text and "limited sources" in quote
    assert answer.citations[0].chunk_id == "a" and "[1]" in answer.answer


def test_extractive_answer_admits_no_match() -> None:
    answer = extractive_answer("dividend policy", [chunk("a", TEXT)], QueryFilters())
    assert answer.insufficient_context and answer.citations == []


def test_answer_question_without_llm_is_extractive() -> None:
    result = answer_question(
        "limited sources of components", FakeRetriever([chunk("a", TEXT)]), None
    )  # type: ignore[arg-type]
    assert result.llm_model == "extractive" and result.cost_usd == 0.0
    assert result.answer.citations and result.answer.citations[0].verified


# --- Ollama -------------------------------------------------------------------------------


def _ollama(handler: Any) -> tuple[OllamaLLM, MemoryCallLog]:
    log = MemoryCallLog()
    return OllamaLLM("qwen2.5:3b", log, transport=httpx.MockTransport(handler)), log


def _gen(llm: OllamaLLM) -> LLMResult[AnswerDraft]:
    return llm.generate(
        system="sys", user="u", schema=AnswerDraft, purpose="answer", prompt_version="answer_v1"
    )


DRAFT = {
    "answer": "x [C1]",
    "citations": [{"ref": "C1", "quote": "q"}],
    "insufficient_context": False,
    "confidence": "low",
}


def test_ollama_structured_request_and_logging() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(DRAFT)},
                "done_reason": "stop",
                "prompt_eval_count": 900,
                "eval_count": 60,
            },
        )

    llm, log = _ollama(handler)
    result = _gen(llm)
    assert seen["format"]["properties"].keys() >= {"answer", "citations"}
    assert seen["options"]["temperature"] == 0 and seen["stream"] is False
    assert seen["messages"][0] == {"role": "system", "content": "sys"}
    assert result.parsed is not None and result.parsed.answer == "x [C1]"
    assert result.model == "ollama/qwen2.5:3b" and result.cost_usd == 0.0
    assert log.calls[0].model == "ollama/qwen2.5:3b" and log.calls[0].input_tokens == 900
    assert cost_usd("ollama/qwen2.5:3b", 10**6, 10**6) == 0.0


@pytest.mark.parametrize(
    ("payload", "stop"),
    [
        ({"message": {"content": '{"answer": "trunc'}, "done_reason": "length"}, "max_tokens"),
        ({"message": {"content": '{"answer": 1}'}, "done_reason": "stop"}, "invalid_output"),
    ],
)
def test_ollama_truncation_and_invalid_output(payload: dict[str, Any], stop: str) -> None:
    llm, _ = _ollama(lambda r: httpx.Response(200, json=payload))
    result = _gen(llm)
    assert result.parsed is None and result.stop_reason == stop


def test_ollama_not_running_or_model_missing() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaUnavailableError, match="ollama serve"):
        _gen(_ollama(down)[0])
    with pytest.raises(OllamaUnavailableError, match=r"ollama pull qwen2\.5:3b"):
        _gen(_ollama(lambda r: httpx.Response(404, json={"error": "not found"}))[0])


def test_provider_auto_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    import filinglens.llm.ollama as ollama_mod

    base = Settings(_env_file=None)  # type: ignore[call-arg]
    with_key = base.model_copy(update={"anthropic_api_key": "sk-test"})
    assert resolve_provider(with_key) == "anthropic"
    monkeypatch.setattr(ollama_mod, "ollama_available", lambda url, model: False)
    assert resolve_provider(base) == "none"
    monkeypatch.setattr(ollama_mod, "ollama_available", lambda url, model: True)
    assert resolve_provider(base) == "ollama"
    assert resolve_provider(with_key, "none") == "none"


# --- router -------------------------------------------------------------------------------

DIRECTORY = CompanyDirectory(
    by_ticker={**CIKS, "ACME": "0000999001"},
    by_alias={"apple": CIKS["AAPL"], "microsoft": CIKS["MSFT"]},
)


def _plan(question: str, latest: int | None = 2023) -> list[tuple[str, dict[str, Any]]]:
    calls = plan(question, parse_filters(question, DIRECTORY), lambda: latest)
    return [(c.name, c.args) for c in calls]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What was Apple's operating margin from 2021 to 2023?",
            [
                (
                    "compute_ratio",
                    {
                        "ticker": "AAPL",
                        "ratio": "operating_margin",
                        "fiscal_years": [2021, 2022, 2023],
                    },
                )
            ],
        ),
        (
            "What was Microsoft's revenue in fiscal 2022?",
            [
                (
                    "get_financials",
                    {"ticker": "MSFT", "concepts": ["revenue"], "fiscal_years": [2022]},
                )
            ],
        ),
        (
            "What new risk factors did ACME add in its 2023 10-K?",
            [("compare_sections", {"ticker": "ACME", "item": "1A", "year_b": 2023})],
        ),
        (
            "What does Apple say about supply chain risk?",
            [
                (
                    "search_filings",
                    {
                        "query": "What does Apple say about supply chain risk?",
                        "top_k": 6,
                        "ticker": "AAPL",
                    },
                )
            ],
        ),
    ],
)
def test_router_plans(question: str, expected: list[tuple[str, dict[str, Any]]]) -> None:
    assert _plan(question) == expected


def test_router_numbers_plus_explanation_and_latest_year() -> None:
    calls = _plan("How did Apple's gross margin change and what drove it?")
    assert [n for n, _ in calls] == ["compute_ratio", "search_filings"]
    assert calls[0][1]["fiscal_years"] == [2023]  # latest available year
    assert calls[1][1]["item"] == "7"


def test_router_without_company_only_searches() -> None:
    assert [n for n, _ in _plan("What was the gross margin in 2023?")] == ["search_filings"]


def _router(tmp_path: Path, llm: Any = None) -> RouterAgent:
    def factory(registry: SourceRegistry) -> AgentContext:
        ctx = make_ctx(tmp_path)
        ctx.registry = registry
        return ctx

    return RouterAgent(factory, llm)


def test_router_agent_template_answer_is_cited_and_tool_backed(tmp_path: Path) -> None:
    result = _router(tmp_path).run("What was Apple's gross margin in fiscal 2023?")
    a = result.answer
    assert result.tools_called == ["compute_ratio"] and result.model == "router"
    assert "FY2023 44.1%" in a.answer and a.citation_validity == 1.0
    assert a.citations[0].source == "tool" and a.citations[0].quote == "0.441311"
    assert result.unsupported_numbers == []


def test_router_agent_financials_and_missing_data(tmp_path: Path) -> None:
    result = _router(tmp_path).run("What was MSFT net income in 2023 and 2019?")
    assert "$72.36 billion" in result.answer.answer
    assert "net_income FY2019" in result.answer.answer
    assert result.unsupported_numbers == []


class FakeWriter:
    model = "ollama/qwen2.5:3b"

    def __init__(self, draft: AnswerDraft) -> None:
        self.draft, self.calls = draft, []

    def generate(self, **kw: Any) -> LLMResult[AnswerDraft]:
        self.calls.append(kw)
        return LLMResult(self.draft, self.model, "end_turn", 800, 50, 1200, 0.0)


def test_router_agent_local_writer_used_when_citations_check_out(tmp_path: Path) -> None:
    good = AnswerDraft(
        answer="Apple's gross margin was 44.1% in fiscal 2023 [T1].",
        citations=[DraftCitation(ref="T1", quote="0.441311")],
        insufficient_context=False,
        confidence="high",
    )
    writer = FakeWriter(good)
    result = _router(tmp_path, writer).run("What was Apple's gross margin in fiscal 2023?")
    assert result.answer.answer == "Apple's gross margin was 44.1% in fiscal 2023 [1]."
    assert result.model == "ollama/qwen2.5:3b" and result.steps[-1].kind == "llm"
    assert "<tool_results>" in writer.calls[0]["user"]


def test_router_agent_falls_back_to_template_on_bad_local_citations(tmp_path: Path) -> None:
    bad = AnswerDraft(
        answer="Margin was 51% [T1].",
        citations=[DraftCitation(ref="T1", quote="0.51")],
        insufficient_context=False,
        confidence="high",
    )
    result = _router(tmp_path, FakeWriter(bad)).run("What was Apple's gross margin in 2023?")
    assert "FY2023 44.1%" in result.answer.answer and result.answer.citation_validity == 1.0


def test_heading_words_count_for_the_sentence_below() -> None:
    text = (
        "ITEM 3. LEGAL PROCEEDINGS\n\nIn June 2015, the State of Louisiana filed suit "
        "against Bluewater and other operators alleging damage to coastal wetlands."
    )
    answer = extractive_answer(
        "What legal proceedings does Bluewater face?",
        [chunk("a", text)],
        QueryFilters(tickers=("BLUE",)),
    )
    assert answer.citations and answer.citations[0].quote.startswith("In June 2015")
