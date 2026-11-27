"""FastAPI application (specs/06). OpenAPI docs at /docs."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from filinglens import __version__
from filinglens.agent.loop import Step
from filinglens.api.guards import RateLimiter
from filinglens.api.observability import configure_logging, request_context_middleware
from filinglens.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkOut,
    CompanyOut,
    CompareRequest,
    FinancialsResponse,
    Health,
    Ready,
    SectionSpanOut,
    TraceOut,
)
from filinglens.api.services import BadRequestError, NotFoundError, Services
from filinglens.config import Settings, get_settings

logger = logging.getLogger("filinglens.api")


def client_ip(request: Request, trust_proxy: bool) -> str:
    """Client address; ``X-Forwarded-For`` only when running behind a trusted proxy."""
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_app(
    settings: Settings | None = None,
    services_factory: Callable[[Settings], Any] | None = None,
) -> FastAPI:
    """Build the app. ``services_factory`` lets tests inject a fake service layer."""
    settings = settings or get_settings()
    factory = services_factory or Services
    limits = {
        "ask": RateLimiter(settings.api_rate_limit_per_minute),
        "ask_high": RateLimiter(settings.api_rate_limit_high_per_minute),
        "other": RateLimiter(settings.api_rate_limit_other_per_minute),
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.services = factory(settings)
        yield
        close = getattr(app.state.services, "close", None)
        if close:
            close()

    api = FastAPI(
        title="FilingLens API",
        version=__version__,
        lifespan=lifespan,
        description="Cited answers, risk-factor diffs and XBRL financials from SEC 10-K/10-Qs.",
    )
    api.middleware("http")(request_context_middleware)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Trace-ID"],
    )

    def services(request: Request) -> Any:
        return request.app.state.services

    def limit(bucket: str) -> Callable[[Request], None]:
        def dependency(request: Request) -> None:
            wait = limits[bucket].check(client_ip(request, settings.api_trust_proxy))
            if wait is not None:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded.",
                    headers={"Retry-After": str(int(wait) + 1)},
                )

        return dependency

    def limit_ask(request: Request, body: AskRequest) -> None:
        limit("ask")(request)
        if body.quality == "high":
            limit("ask_high")(request)

    @api.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @api.exception_handler(BadRequestError)
    async def bad_request(_: Request, exc: BadRequestError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @api.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal error. See server logs."})

    @api.get("/health", response_model=Health)
    def health() -> Health:
        """Liveness."""
        return Health(status="ok", version=__version__)

    @api.get("/ready", response_model=Ready)
    def ready(svc: Any = Depends(services)) -> Ready:
        """Readiness: database reachable and which LLM provider answers."""
        ok, chunks = svc.ready()
        return Ready(
            status="ok" if ok and chunks else "degraded",
            database=ok,
            provider=svc.provider,
            chunks=chunks,
        )

    @api.get("/companies", response_model=list[CompanyOut], dependencies=[Depends(limit("other"))])
    def companies(svc: Any = Depends(services)) -> list[CompanyOut]:
        """Ingested companies and the fiscal years available."""
        result: list[CompanyOut] = svc.companies()
        return result

    @api.post("/ask", response_model=AskResponse, dependencies=[Depends(limit_ask)])
    def ask(body: AskRequest, request: Request, svc: Any = Depends(services)) -> AskResponse:
        """Answer a question with citations (agent with tools, or retrieval + answer)."""
        result: AskResponse = svc.ask(body)
        # Sync endpoints run in a worker thread with a copied context, so hand the trace id
        # to the middleware through request.state (shared) rather than a contextvar.
        request.state.trace_id = result.trace_id
        return result

    @api.post("/ask/stream", dependencies=[Depends(limit_ask)])
    async def ask_stream(body: AskRequest, svc: Any = Depends(services)) -> StreamingResponse:
        """Server-Sent Events: a ``step`` event per tool call, then ``answer`` (or ``error``)."""
        events: queue.Queue[tuple[str, str] | None] = queue.Queue()

        def on_step(step: Step) -> None:
            events.put(("step", json.dumps(asdict(step), default=str)))

        def work() -> None:
            try:
                resp: AskResponse = svc.ask(body, on_step=on_step)
                events.put(("answer", resp.model_dump_json()))
            except (NotFoundError, BadRequestError) as exc:
                events.put(("error", json.dumps({"detail": str(exc)})))
            except Exception:
                logger.exception("stream failed")
                events.put(("error", json.dumps({"detail": "Internal error."})))
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()

        async def stream() -> AsyncIterator[str]:
            loop = asyncio.get_running_loop()
            while True:
                item = await loop.run_in_executor(None, events.get)
                if item is None:
                    yield "event: done\ndata: {}\n\n"
                    return
                name, data = item
                yield f"event: {name}\ndata: {data}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.post("/compare", dependencies=[Depends(limit("other"))])
    def compare(body: CompareRequest, svc: Any = Depends(services)) -> dict[str, Any]:
        """What changed in a 10-K item between two fiscal years."""
        result: dict[str, Any] = svc.compare(body)
        return result

    @api.get(
        "/financials/{ticker}",
        response_model=FinancialsResponse,
        dependencies=[Depends(limit("other"))],
    )
    def financials(
        ticker: str,
        years: list[int] | None = Query(default=None, description="Fiscal years (default: last 5)"),
        svc: Any = Depends(services),
    ) -> FinancialsResponse:
        """Key metrics from XBRL: revenue, income, margins, growth, ROE, current ratio."""
        if not ticker.replace(".", "").replace("-", "").isalpha() or len(ticker) > 10:
            raise HTTPException(status_code=422, detail="Invalid ticker.")
        result: FinancialsResponse = svc.financials(ticker, years)
        return result

    @api.get(
        "/filings/{accession_no}/chunk/{chunk_id}",
        response_model=ChunkOut,
        dependencies=[Depends(limit("other"))],
    )
    def chunk(accession_no: str, chunk_id: str, svc: Any = Depends(services)) -> ChunkOut:
        """Source text behind a citation."""
        result: ChunkOut = svc.chunk(accession_no, chunk_id)
        return result

    @api.get(
        "/filings/{accession_no}/section/{item}",
        response_model=SectionSpanOut,
        dependencies=[Depends(limit("other"))],
    )
    def section(
        accession_no: str,
        item: str,
        start: int = Query(ge=0),
        end: int = Query(gt=0),
        svc: Any = Depends(services),
    ) -> SectionSpanOut:
        """A passage from a parsed section with context (citations from section diffs)."""
        result: SectionSpanOut = svc.section_span(accession_no, item, start, end)
        return result

    @api.get("/traces/{trace_id}", response_model=TraceOut, dependencies=[Depends(limit("other"))])
    def trace(trace_id: str, svc: Any = Depends(services)) -> TraceOut:
        """Steps behind an answer ("How I got this answer")."""
        result: TraceOut = svc.trace(trace_id)
        return result

    return api


def _lazy_app() -> FastAPI:
    return create_app()


app = _lazy_app()
