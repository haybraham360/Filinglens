import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import filinglens.agent.loop as loop_mod
from filinglens.agent.loop import FINAL_NUDGE, Agent
from filinglens.agent.sources import SourceRegistry
from filinglens.agent.tools import AgentContext
from filinglens.llm.calllog import MemoryCallLog, SpendCapExceededError
from filinglens.llm.provider import FALLBACK_BETA
from tests.test_agent_tools import make_ctx


def usage(i: int = 1000, o: int = 100) -> Any:
    return SimpleNamespace(
        input_tokens=i,
        output_tokens=o,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        iterations=None,
    )


def tool_use(id_: str, name: str, args: dict[str, Any]) -> Any:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


def thinking() -> Any:
    return SimpleNamespace(type="thinking", thinking="", signature="sig")


def final(payload: dict[str, Any]) -> Any:
    return SimpleNamespace(type="text", text=json.dumps(payload))


def response(content: list[Any], stop: str) -> Any:
    return SimpleNamespace(
        content=content, stop_reason=stop, usage=usage(), model="claude-sonnet-5-5"
    )


class ScriptedClient:
    """Returns pre-baked responses and records every request."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        # snapshot messages: the loop appends to the same list later
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def make_agent(
    tmp_path: Path, client: ScriptedClient, **kw: Any
) -> tuple[Agent, list[AgentContext]]:
    made: list[AgentContext] = []

    def factory(registry: SourceRegistry) -> AgentContext:
        ctx = make_ctx(tmp_path)
        ctx.registry = registry
        made.append(ctx)
        return ctx

    agent = Agent(
        client,
        "claude-sonnet-5-5",
        kw.pop("log", MemoryCallLog()),
        factory,
        trace_log=tmp_path / "traces.jsonl",
        **kw,
    )
    return agent, made


ANSWER = {
    "answer": "Apple's gross margin was 44.1% in FY2023 [T1]. Apple states products margin "
    "fell due to the weaker U.S. dollar [C1].",
    "citations": [
        {"ref": "T1", "quote": "0.441311"},
        {"ref": "C1", "quote": "Products gross margin decreased due to the weaker U.S. dollar"},
    ],
    "insufficient_context": False,
    "confidence": "high",
}


def test_two_tools_in_parallel_then_cited_answer(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            response(
                [
                    thinking(),
                    tool_use(
                        "tu1",
                        "compute_ratio",
                        {"ticker": "AAPL", "ratio": "gross_margin", "fiscal_years": [2023]},
                    ),
                    tool_use(
                        "tu2",
                        "search_filings",
                        {"query": "gross margin drivers", "ticker": "AAPL", "item": "7"},
                    ),
                ],
                "tool_use",
            ),
            response([thinking(), final(ANSWER)], "end_turn"),
        ]
    )
    log = MemoryCallLog()
    agent, _ = make_agent(tmp_path, client, log=log)
    result = agent.run("How did Apple's gross margin change and why?")

    assert result.tools_called == ["compute_ratio", "search_filings"]
    a = result.answer
    assert a.answer == (
        "Apple's gross margin was 44.1% in FY2023 [1]. Apple states products "
        "margin fell due to the weaker U.S. dollar [2]."
    )
    assert [(c.n, c.source, c.verified) for c in a.citations] == [
        (1, "tool", True),
        (2, "chunk", True),
    ]
    assert a.citation_validity == 1.0
    assert result.unsupported_numbers == []

    first, second = client.requests
    # request shape: strict tools, structured final answer, fallbacks, no forced tool use
    assert all(t["strict"] for t in first["tools"])
    assert first["output_config"]["format"]["type"] == "json_schema"
    assert first["betas"] == [FALLBACK_BETA] and first["fallbacks"] == "default"
    assert "tool_choice" not in first and "thinking" not in first
    # history is append-only: full assistant content (thinking included), then ONE user
    # message carrying both tool results in order
    assert second["messages"][0] == first["messages"][0]
    assistant, results = second["messages"][1], second["messages"][2]
    assert assistant["role"] == "assistant" and assistant["content"][0].type == "thinking"
    assert results["role"] == "user"
    assert [r["tool_use_id"] for r in results["content"]] == ["tu1", "tu2"]
    assert second["tools"] == first["tools"] and second["system"] == first["system"]

    assert len(log.calls) == 2 and {c.purpose for c in log.calls} == {"agent_step"}
    trace = json.loads((tmp_path / "traces.jsonl").read_text())
    assert trace["tools_called"] == ["compute_ratio", "search_filings"]


def test_step_limit_nudges_then_disables_tools(tmp_path: Path) -> None:
    loop_call = tool_use(
        "x", "get_financials", {"ticker": "AAPL", "concepts": ["revenue"], "fiscal_years": [2023]}
    )
    client = ScriptedClient(
        [
            response([loop_call], "tool_use"),
            response([loop_call], "tool_use"),
            response([final({**ANSWER, "citations": []})], "end_turn"),
        ]
    )
    agent, _ = make_agent(tmp_path, client, max_steps=3)
    result = agent.run("q")
    assert len(client.requests) == 3
    assert "tool_choice" not in client.requests[1]
    assert client.requests[2]["tool_choice"] == {"type": "none"}
    nudge = client.requests[2]["messages"][-1]["content"][-1]
    assert nudge == {"type": "text", "text": FINAL_NUDGE}
    assert result.answer.answer.startswith("Apple's gross margin")


def test_invalid_args_and_errors_go_back_to_the_model(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            response([tool_use("a", "compute_ratio", {"ticker": "AAPL"})], "tool_use"),
            response(
                [final({**ANSWER, "citations": [], "insufficient_context": True})], "end_turn"
            ),
        ]
    )
    agent, _ = make_agent(tmp_path, client)
    result = agent.run("q")
    tool_result = client.requests[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True and "Invalid arguments" in tool_result["content"]
    assert result.steps[1].is_error


def test_tool_timeout_is_reported_not_hung(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop_mod, "TOOL_TIMEOUT_S", 0.2)

    def slow(ctx: Any, name: str, raw: Any) -> tuple[dict[str, Any], bool]:
        time.sleep(1.0)
        return {}, False

    monkeypatch.setattr(loop_mod, "run_tool", slow)
    client = ScriptedClient(
        [
            response([tool_use("a", "search_filings", {"query": "x"})], "tool_use"),
            response([final({**ANSWER, "citations": []})], "end_turn"),
        ]
    )
    agent, _ = make_agent(tmp_path, client)
    agent.run("q")
    content = client.requests[1]["messages"][2]["content"][0]["content"]
    assert "timed out" in content


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_refusal_and_truncation_stop_the_loop(tmp_path: Path, stop: str) -> None:
    client = ScriptedClient([response([], stop)])
    agent, _ = make_agent(tmp_path, client)
    result = agent.run("q")
    assert result.answer.insufficient_context and len(client.requests) == 1


def test_unparseable_final_answer_is_flagged(tmp_path: Path) -> None:
    client = ScriptedClient([response([SimpleNamespace(type="text", text="not json")], "end_turn")])
    agent, _ = make_agent(tmp_path, client)
    assert "could not be parsed" in agent.run("q").answer.answer


def test_llm_arithmetic_is_caught(tmp_path: Path) -> None:
    made_up = {**ANSWER, "answer": "Gross margin was 44.1% [T1], so profit rose $9.9 billion."}
    client = ScriptedClient(
        [
            response(
                [
                    tool_use(
                        "a",
                        "compute_ratio",
                        {"ticker": "AAPL", "ratio": "gross_margin", "fiscal_years": [2023]},
                    )
                ],
                "tool_use",
            ),
            response([final(made_up)], "end_turn"),
        ]
    )
    agent, _ = make_agent(tmp_path, client)
    assert agent.run("q").unsupported_numbers == ["$9.9 billion"]


def test_spend_cap_is_checked_before_each_request(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            response([tool_use("a", "search_filings", {"query": "x"})], "tool_use"),
            response([final(ANSWER)], "end_turn"),
        ]
    )
    agent, _ = make_agent(tmp_path, client, daily_cap_usd=0.0025)  # one step costs $0.003
    with pytest.raises(SpendCapExceededError):
        agent.run("q")
    assert len(client.requests) == 1
