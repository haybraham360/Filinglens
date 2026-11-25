"""Retrieval pipeline: filters -> dense (+ lexical, RRF) (+ rerank) -> top chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import psycopg

from filinglens.retrieval.embedder import Embedder
from filinglens.retrieval.filters import (
    CompanyDirectory,
    QueryFilters,
    lexical_query,
    parse_filters,
)
from filinglens.retrieval.rerank import Reranker, rerank
from filinglens.retrieval.search import (
    RetrievedChunk,
    dense_search,
    lexical_search,
    reciprocal_rank_fusion,
)


@dataclass(frozen=True)
class RetrievalConfig:
    """Eval-tunable retrieval knobs (specs/03)."""

    top_k: int = 8
    candidate_k: int = 40
    use_hybrid: bool = True
    rrf_k: int = 60
    use_rerank: bool = False
    rerank_top_n: int = 8
    use_filters: bool = True


@dataclass
class RetrievalResult:
    """Retrieved chunks plus what was done to get them."""

    chunks: list[RetrievedChunk]
    filters: QueryFilters
    relaxed: list[str] = field(default_factory=list)
    lexical_query: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)


# Filters are dropped in this order when nothing matches; the company is never dropped,
# since answering about the wrong company is worse than saying "not found".
_RELAX_ORDER = ("items", "forms", "fiscal_years")


class Retriever:
    """Runs one query end to end against Postgres."""

    def __init__(
        self,
        conn: psycopg.Connection[Any],
        embedder: Embedder,
        directory: CompanyDirectory,
        config: RetrievalConfig = RetrievalConfig(),
        reranker: Reranker | None = None,
    ) -> None:
        """Bind the retriever to a connection (with pgvector registered) and models."""
        if config.use_rerank and reranker is None:
            raise ValueError("use_rerank=True requires a reranker")
        self.conn = conn
        self.embedder = embedder
        self.directory = directory
        self.config = config
        self.reranker = reranker

    def _search(
        self, question: str, filters: QueryFilters, timings: dict[str, float]
    ) -> tuple[list[RetrievedChunk], str]:
        cfg = self.config
        t0 = time.perf_counter()
        query_vec = self.embedder.embed_query(question)
        timings["embed"] = timings.get("embed", 0.0) + (time.perf_counter() - t0) * 1000

        pool = cfg.candidate_k if (cfg.use_hybrid or cfg.use_rerank) else cfg.top_k
        t0 = time.perf_counter()
        dense = dense_search(self.conn, query_vec, self.embedder.model_name, filters, pool)
        timings["dense"] = timings.get("dense", 0.0) + (time.perf_counter() - t0) * 1000
        if not cfg.use_hybrid:
            return dense, ""

        tsq = lexical_query(question, filters)
        t0 = time.perf_counter()
        lexical = lexical_search(self.conn, tsq, filters, pool)
        timings["lexical"] = timings.get("lexical", 0.0) + (time.perf_counter() - t0) * 1000
        return reciprocal_rank_fusion([dense, lexical], k=cfg.rrf_k), tsq

    def retrieve(self, question: str, filters: QueryFilters | None = None) -> RetrievalResult:
        """Retrieve the chunks to answer ``question``.

        ``filters`` overrides rule-based extraction (the agent passes the filters it chose).
        """
        cfg = self.config
        timings: dict[str, float] = {}
        start = time.perf_counter()
        if filters is None:
            filters = parse_filters(question, self.directory) if cfg.use_filters else QueryFilters()

        candidates, tsq = self._search(question, filters, timings)
        relaxed: list[str] = []
        for name in _RELAX_ORDER:
            if candidates:
                break
            if getattr(filters, name):
                filters = replace(filters, **{name: ()})
                relaxed.append(name)
                candidates, tsq = self._search(question, filters, timings)

        if cfg.use_rerank and self.reranker is not None and candidates:
            t0 = time.perf_counter()
            chunks = rerank(self.reranker, question, candidates, cfg.rerank_top_n)
            timings["rerank"] = (time.perf_counter() - t0) * 1000
        else:
            chunks = candidates[: cfg.top_k]
        timings["total"] = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            chunks=chunks, filters=filters, relaxed=relaxed, lexical_query=tsq, timings_ms=timings
        )
