"""API behaviour with a fake service layer (no database, no LLM)."""

import io
import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from filinglens import __version__
from filinglens.agent.loop import Step
from filinglens.api.guards import RateLimiter, TTLCache, cache_key
from filinglens.api.main import create_app
from filinglens.api.observability import JsonFormatter, request_id_var
from filinglens.api.schemas import AskRequest, AskResponse, CompanyOut
from filinglens.api.services import NotFoundError
from filinglens.config import Settings
from filinglens.rag.schemas import Answer, Citation


class FakeServices:
    provider = "none"

    def __init__(self, settings: Settings) -> None:
        self.asked: list[AskRequest] = []

    def ready(self) -> tuple[bool, int]:
        return True, 42

    def companies(self) -> list[CompanyOut]:
        return [
            CompanyOut(
                ticker="AAPL",
                cik="0000320193",
                name="Apple Inc.",
                years_10k=[2022, 2023],
                years_10q=[2024],
            )
        ]

    def ask(self, req: AskRequest, on_step: Any = None) -> AskResponse:
        self.asked.append(req)
        if req.question.startswith("boom"):
            raise RuntimeError("secret internal detail")
        if on_step:
            on_step(Step(index=0, kind="tool", name="search_filings", latency_ms=12))
        cite = Citation(
            n=1,
            chunk_id="c1",
            accession_no="0000320193-23-000106",
            item="1A",
            quote="limited sources",
            verified=True,
            match_score=1.0,
        )
        return AskResponse(
            answer=Answer(
                answer="Apple cites limited sources [1].",
                citations=[cite],
                insufficient_context=False,
                confidence="medium",
                citation_validity=1.0,
            ),
            trace_id="t123",
            mode=req.mode,
            provider="none",
            model="router",
            latency_ms=5,
            tools_called=["search_filings"],
        )

    def compare(self, req: Any) -> dict[str, Any]:
        raise NotFoundError("Need two 10-K filings with Item 1A; available years: [2023].")

    def chunk(self, accession: str, chunk_id: str) -> Any:
        raise NotFoundError(f"No chunk {chunk_id} in {accession}.")


def client(**overrides: Any) -> TestClient:
    settings = Settings(_env_file=None, **overrides)  # type: ignore[call-arg]
    app = create_app(settings, services_factory=FakeServices)
    return TestClient(app, raise_server_exceptions=False)


def test_health_and_ready() -> None:
    with client() as c:
        assert c.get("/health").json() == {"status": "ok", "version": __version__}
        assert c.get("/ready").json() == {
            "status": "ok",
            "database": True,
            "provider": "none",
            "chunks": 42,
        }
        assert c.get("/companies").json()[0]["ticker"] == "AAPL"
        assert "/ask" in c.get("/openapi.json").json()["paths"]


def test_ask_returns_answer_with_request_and_trace_ids() -> None:
    with client() as c:
        r = c.post(
            "/ask",
            json={"question": "Supply risks?", "ticker": "AAPL"},
            headers={"X-Request-ID": "req-abc"},
        )
        assert r.status_code == 200
        assert r.json()["answer"]["citations"][0]["verified"] is True
        assert r.headers["X-Request-ID"] == "req-abc"
        bad = c.post(
            "/ask", json={"question": "Supply risks?"}, headers={"X-Request-ID": "x" * 200 + "\n"}
        )
        assert bad.headers["X-Request-ID"] != "x" * 200  # unsafe ids are replaced


@pytest.mark.parametrize(
    "body",
    [
        {"question": "hi"},
        {"question": "ok question", "ticker": "AAPL; DROP"},
        {"question": "ok question", "fiscal_year": 1900},
        {"question": "ok question", "mode": "x"},
        {"question": "q" * 2001},
    ],
)
def test_ask_input_validation(body: dict[str, Any]) -> None:
    with client() as c:
        assert c.post("/ask", json=body).status_code == 422


def test_rate_limits_standard_and_high_quality() -> None:
    with client(api_rate_limit_per_minute=3, api_rate_limit_high_per_minute=1) as c:
        q = {"question": "Supply risks?"}
        assert c.post("/ask", json=q | {"quality": "high"}).status_code == 200
        r = c.post("/ask", json=q | {"quality": "high"})
        assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
        assert c.post("/ask", json=q).status_code == 200
        assert c.post("/ask", json=q).status_code == 429  # 3 standard slots used (incl. high)


def test_not_found_and_internal_errors_are_clean() -> None:
    with client() as c:
        r = c.post("/compare", json={"ticker": "AAPL", "item": "1A"})
        assert r.status_code == 404 and "available years" in r.json()["detail"]
        assert c.get("/filings/a/chunk/b").status_code == 404
        r = c.post("/ask", json={"question": "boom please"})
        assert r.status_code == 500
        assert "secret" not in r.text and r.json()["detail"].startswith("Internal error")
        assert c.post("/compare", json={"ticker": "AAPL", "item": "../etc"}).status_code == 422
        assert c.get("/financials/A$PL").status_code == 422


def test_ask_stream_sends_steps_then_answer() -> None:
    with client() as c, c.stream("POST", "/ask/stream", json={"question": "Supply risks?"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [line[7:] for line in body.splitlines() if line.startswith("event: ")]
    assert events == ["step", "answer", "done"]
    data = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert data[0]["name"] == "search_filings" and data[1]["trace_id"] == "t123"


def test_stream_reports_errors_as_events() -> None:
    with client() as c, c.stream("POST", "/ask/stream", json={"question": "boom now"}) as r:
        body = "".join(r.iter_text())
    assert "event: error" in body and "secret" not in body


def test_cors_restricted_to_configured_origin() -> None:
    with client(api_cors_origins=["https://app.example"]) as c:
        ok = c.options(
            "/ask",
            headers={"Origin": "https://app.example", "Access-Control-Request-Method": "POST"},
        )
        assert ok.headers.get("access-control-allow-origin") == "https://app.example"
        bad = c.options(
            "/ask",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in bad.headers


def test_cors_origin_regex_allows_preview_deployments() -> None:
    regex = r"^https://filinglens-[a-z0-9]+-team\.vercel\.app$"
    with client(api_cors_origins=["https://app.example"], api_cors_origin_regex=regex) as c:

        def allowed(origin: str) -> str | None:
            r = c.options(
                "/ask", headers={"Origin": origin, "Access-Control-Request-Method": "POST"}
            )
            return r.headers.get("access-control-allow-origin")

        assert (
            allowed("https://filinglens-abc123-team.vercel.app")
            == "https://filinglens-abc123-team.vercel.app"
        )
        assert allowed("https://app.example") == "https://app.example"
        assert allowed("https://filinglens-abc123-team.vercel.app.evil.com") is None


def test_proxy_header_only_trusted_when_configured() -> None:
    with client(api_rate_limit_per_minute=1) as c:
        h = {"X-Forwarded-For": "1.1.1.1"}
        assert c.post("/ask", json={"question": "Supply risks?"}, headers=h).status_code == 200
        # untrusted: the spoofed header doesn't buy a fresh bucket
        h2 = {"X-Forwarded-For": "2.2.2.2"}
        assert c.post("/ask", json={"question": "Supply risks?"}, headers=h2).status_code == 429
    with client(api_rate_limit_per_minute=1, api_trust_proxy=True) as c:
        assert (
            c.post(
                "/ask", json={"question": "Supply risks?"}, headers={"X-Forwarded-For": "1.1.1.1"}
            ).status_code
            == 200
        )
        assert (
            c.post(
                "/ask", json={"question": "Supply risks?"}, headers={"X-Forwarded-For": "2.2.2.2"}
            ).status_code
            == 200
        )


# --- guards and logging -------------------------------------------------------------------


def test_rate_limiter_sliding_window() -> None:
    now = [0.0]
    rl = RateLimiter(2, window_s=60, clock=lambda: now[0])
    assert rl.check("a") is None and rl.check("a") is None
    wait = rl.check("a")
    assert wait == pytest.approx(60.0)
    assert rl.check("b") is None
    now[0] = 60.5
    assert rl.check("a") is None


def test_ttl_cache_expiry_and_lru() -> None:
    now = [0.0]
    cache: TTLCache[int] = TTLCache(max_items=2, ttl_s=10, clock=lambda: now[0])
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)  # evicts b (least recently used)
    assert cache.get("b") is None and cache.get("a") == 1
    now[0] = 11
    assert cache.get("a") is None
    assert cache_key(" Apple  RISKS ", "AAPL") == cache_key("apple risks", "AAPL")
    assert cache_key("apple risks", "AAPL") != cache_key("apple risks", "MSFT")


def test_json_logs_carry_request_id() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    log = logging.getLogger("test.json")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    token = request_id_var.set("rid-1")
    try:
        log.info("hello", extra={"path": "/ask"})
    finally:
        request_id_var.reset(token)
        log.removeHandler(handler)
    record = json.loads(stream.getvalue())
    assert record["msg"] == "hello" and record["request_id"] == "rid-1"
    assert record["path"] == "/ask" and record["level"] == "INFO"
