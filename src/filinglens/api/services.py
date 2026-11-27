"""Everything the API endpoints do, behind one object built at startup."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from filinglens import __version__
from filinglens.agent.diff import SectionStore
from filinglens.agent.financials import PostgresFactStore
from filinglens.agent.loop import Agent, AgentResult, Step
from filinglens.agent.router import RouterAgent
from filinglens.agent.sources import SourceRegistry
from filinglens.agent.tools import AgentContext, run_tool
from filinglens.api.guards import TTLCache, cache_key
from filinglens.api.observability import trace_id_var
from filinglens.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkOut,
    CompanyOut,
    CompareRequest,
    FinancialsResponse,
    MetricRow,
    SectionSpanOut,
    StepOut,
    TraceOut,
)
from filinglens.config import Settings
from filinglens.db.repositories import schema_sql
from filinglens.llm.calllog import PostgresCallLog, SpendCapExceededError
from filinglens.llm.factory import Provider, make_llm, resolve_provider
from filinglens.llm.prompts import load_prompt
from filinglens.parsing.pipeline import PARSER_VERSION
from filinglens.rag.answer import answer_question, filing_index_url
from filinglens.retrieval.embedder import Embedder, load_embedder
from filinglens.retrieval.filters import CompanyDirectory, parse_filters
from filinglens.retrieval.retriever import RetrievalConfig, Retriever

logger = logging.getLogger("filinglens.api")


class NotFoundError(LookupError):
    """The requested company, filing, chunk or trace doesn't exist."""


class BadRequestError(ValueError):
    """The request is valid JSON but can't be served (e.g. question too long)."""


FINANCIAL_ROWS: list[tuple[str, str, str, str]] = [
    # (metric, label, kind, tool)
    ("revenue", "Revenue", "usd", "get_financials"),
    ("operating_income", "Operating income", "usd", "get_financials"),
    ("net_income", "Net income", "usd", "get_financials"),
    ("gross_margin", "Gross margin", "ratio", "compute_ratio"),
    ("operating_margin", "Operating margin", "ratio", "compute_ratio"),
    ("net_margin", "Net margin", "ratio", "compute_ratio"),
    ("revenue_growth_yoy", "Revenue growth (YoY)", "ratio", "compute_ratio"),
    ("roe", "Return on equity", "ratio", "compute_ratio"),
    ("current_ratio", "Current ratio", "ratio", "compute_ratio"),
]


class Services:
    """Shared state: connection pool, embedder, company directory, caches."""

    def __init__(
        self,
        settings: Settings,
        *,
        pool: ConnectionPool | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """Open the pool (schema applied once) and load the embedding model."""
        self.settings = settings
        self.pool = pool or ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=settings.db_pool_size,
            # prepare_threshold=None: no server-side prepared statements, so the pool also works
            # through transaction-mode poolers (Neon, PgBouncer) used by serverless hosts
            kwargs={"autocommit": True, "prepare_threshold": None},
            configure=register_vector,
            open=True,
        )
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(schema_sql())
        self.embedder = embedder or load_embedder(settings.embedding_model)
        self.sections = SectionStore(settings.processed_dir)
        self.cache: TTLCache[AskResponse] = TTLCache(ttl_s=settings.api_cache_ttl_s)
        self.provider: Provider = resolve_provider(settings)
        self._names = self._company_names()
        self.directory = self._directory()
        self.config_version = self._config_version()
        logger.info(
            "services ready",
            extra={"provider": self.provider, "config_version": self.config_version},
        )

    # -- setup -------------------------------------------------------------------------------

    def _universe_csv(self) -> Any:
        universe = self.settings.universe_file.stem.removesuffix("_universe")
        return self.settings.raw_dir / "universe" / f"{universe}_ciks.csv"

    def _company_names(self) -> dict[str, str]:
        path = self._universe_csv()
        if not path.exists():
            return {}
        with path.open(newline="") as fh:
            return {r["ticker"]: r["name"] for r in csv.DictReader(fh)}

    def _directory(self) -> CompanyDirectory:
        return CompanyDirectory.load(
            self._universe_csv(), self.settings.universe_file.parent / "company_aliases.yaml"
        )

    def _config_version(self) -> str:
        s = self.settings
        parts = [
            __version__,
            PARSER_VERSION,
            load_prompt("answer").id,
            load_prompt("agent").id,
            load_prompt("router_writer").id,
            self.provider,
            s.llm_model,
            s.demo_llm_model,
            s.ollama_model,
            s.embedding_model,
            s.retrieval_top_k,
            s.use_hybrid,
            s.use_rerank,
        ]
        return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:12]

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection[Any]]:
        """A pooled autocommit connection with pgvector registered."""
        with self.pool.connection() as conn:
            yield conn

    def close(self) -> None:
        """Close the pool."""
        self.pool.close()

    def ready(self) -> tuple[bool, int]:
        """(database reachable, chunk count)."""
        try:
            with self.connection() as conn:
                row = conn.execute("SELECT count(*) FROM chunks").fetchone()
            return True, int(row[0]) if row else 0
        except psycopg.Error:
            return False, 0

    # -- building blocks -----------------------------------------------------------------------

    def _retriever(self, conn: psycopg.Connection[Any]) -> Retriever:
        s = self.settings
        cfg = RetrievalConfig(
            top_k=s.retrieval_top_k,
            candidate_k=s.retrieval_candidate_k,
            use_hybrid=s.use_hybrid,
            rrf_k=s.rrf_k,
            use_rerank=False,
            rerank_top_n=s.rerank_top_n,
        )
        return Retriever(conn, self.embedder, self.directory, cfg)

    def _context_factory(
        self, conn: psycopg.Connection[Any]
    ) -> Callable[[SourceRegistry], AgentContext]:
        retriever = self._retriever(conn)
        facts = PostgresFactStore(conn)
        concept_map = str(self.settings.universe_file.parent / "concept_map.yaml")

        def section_vector(accession_no: str, item: str) -> np.ndarray | None:
            # mean of the chunk embeddings already stored for this filing's Item
            row = conn.execute(
                "SELECT avg(embedding) FROM chunks "
                "WHERE accession_no = %s AND item = %s AND embedding IS NOT NULL",
                (accession_no, item),
            ).fetchone()
            return None if row is None or row[0] is None else np.asarray(row[0], dtype=float)

        def factory(registry: SourceRegistry) -> AgentContext:
            return AgentContext(
                retriever=retriever,
                sections=self.sections,
                facts=facts,
                directory=self.directory,
                registry=registry,
                embedder=self.embedder,
                concept_map_path=concept_map,
                section_vector=section_vector,
            )

        return factory

    def cik(self, ticker: str) -> str:
        """Resolve a ticker, or raise NotFoundError."""
        cik = self.directory.by_ticker.get(ticker.upper().replace(".", "-"))
        if cik is None:
            raise NotFoundError(f"Unknown ticker {ticker!r}.")
        return cik

    # -- /ask ------------------------------------------------------------------------------------

    def ask(self, req: AskRequest, on_step: Callable[[Step], None] | None = None) -> AskResponse:
        """Answer a question (cached), falling back to free mode if the spend cap is hit."""
        if len(req.question) > self.settings.api_max_question_chars:
            raise BadRequestError(
                f"Question too long ({len(req.question)} chars; max "
                f"{self.settings.api_max_question_chars})."
            )
        if req.ticker:
            self.cik(req.ticker)
        key = cache_key(
            req.question, req.ticker, req.fiscal_year, req.mode, req.quality, self.config_version
        )
        cached = self.cache.get(key)
        if cached is not None:
            trace_id_var.set(cached.trace_id)
            return cached.model_copy(update={"cached": True})
        try:
            resp = self._answer(req, self.provider, on_step)
        except SpendCapExceededError:
            logger.warning("spend cap reached; answering in free mode")
            resp = self._answer(req, "none", on_step).model_copy(update={"degraded": True})
        self.cache.put(key, resp)
        return resp

    def _scoped_question(self, req: AskRequest) -> str:
        """Fold explicit selectors into the question, where the filter parser picks them up."""
        scope = [
            t
            for t in (
                req.ticker.upper() if req.ticker else None,
                f"fiscal year {req.fiscal_year}" if req.fiscal_year else None,
            )
            if t
        ]
        return f"{req.question} ({', '.join(scope)})" if scope else req.question

    def _answer(
        self, req: AskRequest, provider: Provider, on_step: Callable[[Step], None] | None
    ) -> AskResponse:
        settings = self.settings
        if provider == "anthropic":
            model = settings.llm_model if req.quality == "high" else settings.demo_llm_model
            settings = settings.model_copy(update={"llm_model": model})
        question = self._scoped_question(req)
        start = time.perf_counter()
        with self.connection() as conn:
            call_log = PostgresCallLog(conn)
            if req.mode == "rag":
                retriever = self._retriever(conn)
                filters = parse_filters(question, self.directory)
                retrieval = retriever.retrieve(question, filters=filters)
                if on_step:
                    on_step(
                        Step(
                            index=0,
                            kind="tool",
                            name="search_filings",
                            latency_ms=int(retrieval.timings_ms.get("total", 0)),
                            args={"query": req.question},
                        )
                    )
                res = answer_question(
                    question, retriever, make_llm(settings, provider, call_log), retrieval=retrieval
                )
                resp = AskResponse(
                    answer=res.answer,
                    trace_id=res.trace_id,
                    mode="rag",
                    provider=provider,
                    model=res.llm_model,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    tools_called=["search_filings"],
                )
                payload: dict[str, Any] = {
                    "retrieved": [
                        {"chunk_id": c.chunk_id, "ranks": c.ranks, "score": c.score}
                        for c in retrieval.chunks
                    ],
                    "filters": {
                        "tickers": retrieval.filters.tickers,
                        "fiscal_years": retrieval.filters.fiscal_years,
                    },
                    "relaxed": retrieval.relaxed,
                    "timings_ms": res.timings_ms,
                }
            else:
                result = self._run_agent(conn, settings, provider, question, on_step)
                resp = AskResponse(
                    answer=result.answer,
                    trace_id=result.trace_id,
                    mode="agent",
                    provider=provider,
                    model=result.model,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    tools_called=result.tools_called,
                    steps=[
                        StepOut(**{k: v for k, v in asdict(s).items() if k in StepOut.model_fields})
                        for s in result.steps
                    ],
                    unsupported_numbers=result.unsupported_numbers,
                )
                payload = {
                    "steps": [asdict(s) for s in result.steps],
                    "unsupported_numbers": result.unsupported_numbers,
                    "cost_usd": result.cost_usd,
                }
            trace_id_var.set(resp.trace_id)
            self._store_trace(conn, resp, req, payload)
        return resp

    def _run_agent(
        self,
        conn: psycopg.Connection[Any],
        settings: Settings,
        provider: Provider,
        question: str,
        on_step: Callable[[Step], None] | None,
    ) -> AgentResult:
        factory = self._context_factory(conn)
        if provider == "anthropic":
            import anthropic

            key = (
                settings.anthropic_api_key.get_secret_value()
                if settings.anthropic_api_key
                else None
            )
            agent = Agent(
                anthropic.Anthropic(api_key=key),
                settings.llm_model,
                PostgresCallLog(conn),
                factory,
                effort=settings.llm_effort,
                max_steps=settings.agent_max_steps,
                max_tokens=settings.llm_max_tokens,
                daily_cap_usd=settings.llm_daily_spend_cap_usd,
            )
            return agent.run(question, on_step=on_step)
        llm = make_llm(settings, "ollama", PostgresCallLog(conn)) if provider == "ollama" else None
        return RouterAgent(factory, llm).run(question, on_step=on_step)

    def _store_trace(
        self,
        conn: psycopg.Connection[Any],
        resp: AskResponse,
        req: AskRequest,
        payload: dict[str, Any],
    ) -> None:
        payload |= {
            "answer": resp.answer.model_dump(),
            "provider": resp.provider,
            "model": resp.model,
            "latency_ms": resp.latency_ms,
            "request": req.model_dump(),
            "config_version": self.config_version,
        }
        conn.execute(
            "INSERT INTO traces (trace_id, question, mode, payload) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (trace_id) DO NOTHING",
            (
                resp.trace_id,
                req.question,
                resp.mode,
                Jsonb(json.loads(json.dumps(payload, default=str))),
            ),
        )

    # -- other endpoints -------------------------------------------------------------------------

    def companies(self) -> list[CompanyOut]:
        """Ingested companies with the fiscal years that have chunks."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT ticker, cik, "
                "array_agg(DISTINCT fiscal_year) FILTER (WHERE form LIKE '10-K%%'), "
                "array_agg(DISTINCT fiscal_year) FILTER (WHERE form LIKE '10-Q%%') "
                "FROM chunks WHERE ticker IS NOT NULL GROUP BY ticker, cik ORDER BY ticker"
            ).fetchall()
        return [
            CompanyOut(
                ticker=t,
                cik=c,
                name=self._names.get(t),
                years_10k=sorted(y for y in (k or []) if y),
                years_10q=sorted(y for y in (q or []) if y),
            )
            for t, c, k, q in rows
        ]

    def compare(self, req: CompareRequest) -> dict[str, Any]:
        """``compare_sections`` result (added/removed/modified, similarity measures)."""
        with self.connection() as conn:
            ctx = self._context_factory(conn)(SourceRegistry())
            out, is_error = run_tool(ctx, "compare_sections", req.model_dump())
        if is_error:
            raise NotFoundError(str(out.get("error")))
        for key in (
            "added_risk_headings",
            "removed_risk_headings",
            "added_paragraphs",
            "removed_paragraphs",
        ):
            for entry in out[key]:
                self._attach_span(ctx, entry)
        for pair in out["modified_paragraphs"]:
            self._attach_span(ctx, pair["old"])
            self._attach_span(ctx, pair["new"])
        return out

    @staticmethod
    def _attach_span(ctx: AgentContext, entry: dict[str, Any]) -> None:
        src = ctx.registry.get(entry["ref"])
        if src is not None:
            entry |= {
                "accession_no": src.accession_no,
                "item": src.item,
                "char_start": src.char_start,
                "char_end": src.char_end,
            }

    def financials(self, ticker: str, years: list[int] | None = None) -> FinancialsResponse:
        """Key metrics for the last five fiscal years (or ``years``)."""
        cik = self.cik(ticker)
        with self.connection() as conn:
            if not years:
                rows = conn.execute(
                    "SELECT DISTINCT fy FROM financial_facts WHERE cik = %s AND form = '10-K' "
                    "AND fy IS NOT NULL ORDER BY fy DESC LIMIT 5",
                    (cik,),
                ).fetchall()
                years = sorted(int(r[0]) for r in rows)
            if not years:
                raise NotFoundError(f"No XBRL financials ingested for {ticker.upper()}.")
            ctx = self._context_factory(conn)(SourceRegistry())
            metric_rows: list[MetricRow] = []
            sources: list[dict[str, Any]] = []
            usd = [m for m, _, kind, _ in FINANCIAL_ROWS if kind == "usd"]
            fin, _ = run_tool(
                ctx, "get_financials", {"ticker": ticker, "concepts": usd, "fiscal_years": years}
            )
            by_concept: dict[str, dict[str, str | None]] = {
                m: {str(y): None for y in years} for m in usd
            }
            for v in fin.get("values", []):
                by_concept[v["concept"]][str(v["fiscal_year"])] = v["value"]
                sources.append(v)
            for metric, label, _kind, tool in FINANCIAL_ROWS:
                if tool == "get_financials":
                    metric_rows.append(
                        MetricRow(metric=metric, label=label, kind="usd", values=by_concept[metric])
                    )
                    continue
                out, is_error = run_tool(
                    ctx, "compute_ratio", {"ticker": ticker, "ratio": metric, "fiscal_years": years}
                )
                if is_error:
                    continue
                metric_rows.append(
                    MetricRow(
                        metric=metric,
                        label=label,
                        kind="ratio",
                        values=out["values_by_year"],
                        formula=out["formula"],
                    )
                )
        return FinancialsResponse(
            ticker=ticker.upper(), years=years, rows=metric_rows, sources=sources
        )

    def chunk(self, accession_no: str, chunk_id: str) -> ChunkOut:
        """Source text of one chunk."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT chunk_id, accession_no, cik, ticker, form, fiscal_year, item, heading, "
                "text, char_start, char_end FROM chunks WHERE accession_no = %s AND chunk_id = %s",
                (accession_no, chunk_id),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"No chunk {chunk_id} in {accession_no}.")
        cid, acc, cik, ticker, form, fy, item, heading, text, start, end = row
        return ChunkOut(
            chunk_id=cid,
            accession_no=acc,
            ticker=ticker,
            form=form,
            fiscal_year=fy,
            item=item,
            heading=heading,
            text=text,
            char_start=start,
            char_end=end,
            filing_url=filing_index_url(cik, acc),
        )

    def section_span(
        self, accession_no: str, item: str, start: int, end: int, context: int = 400
    ) -> SectionSpanOut:
        """A passage of a parsed section with surrounding text (for diff citations)."""
        idx = self.sections.index
        rows = idx[(idx["accession_no"] == accession_no) & (idx["item"] == item)]
        if rows.empty:
            raise NotFoundError(f"No Item {item} in {accession_no}.")
        import pandas as pd

        cik = str(rows.iloc[0]["cik"])
        df = pd.read_parquet(
            self.settings.processed_dir / "sections" / cik / f"{accession_no}.parquet"
        )
        text = str(df[df["item"] == item].iloc[0]["text"])
        if not (0 <= start < end <= len(text)):
            raise BadRequestError(f"Span {start}-{end} outside the section (length {len(text)}).")
        return SectionSpanOut(
            accession_no=accession_no,
            item=item,
            before=text[max(0, start - context) : start],
            text=text[start:end],
            after=text[end : end + context],
            filing_url=filing_index_url(cik, accession_no),
        )

    def trace(self, trace_id: str) -> TraceOut:
        """A stored trace."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT trace_id, created_at, question, mode, payload FROM traces "
                "WHERE trace_id = %s",
                (trace_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"No trace {trace_id}.")
        return TraceOut(
            trace_id=row[0],
            created_at=row[1].isoformat(),
            question=row[2],
            mode=row[3],
            payload=row[4],
        )
