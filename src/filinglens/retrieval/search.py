"""Dense (pgvector HNSW), lexical (Postgres full-text) and fused search over ``chunks``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import psycopg
from psycopg import sql

from filinglens.retrieval.embedder import Vector
from filinglens.retrieval.filters import QueryFilters

_COLUMNS = (
    "chunk_id",
    "accession_no",
    "cik",
    "ticker",
    "form",
    "fiscal_year",
    "fiscal_period",
    "item",
    "heading",
    "text",
    "context_header",
    "char_start",
    "char_end",
)
_SELECT = sql.SQL(", ").join(sql.Identifier(c) for c in _COLUMNS)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk with its scores from each retriever that found it."""

    chunk_id: str
    accession_no: str
    cik: str
    ticker: str | None
    form: str
    fiscal_year: int | None
    fiscal_period: str | None
    item: str
    heading: str | None
    text: str
    context_header: str | None
    char_start: int | None
    char_end: int | None
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict, compare=False)

    @property
    def passage(self) -> str:
        """Context header + text: what was embedded, and what the reranker scores."""
        return f"{self.context_header}\n{self.text}" if self.context_header else self.text


def _row_to_chunk(row: Sequence[Any], score: float, source: str, rank: int) -> RetrievedChunk:
    values = dict(zip(_COLUMNS, row, strict=True))
    return RetrievedChunk(**values, score=score, ranks={source: rank})


def filter_clause(filters: QueryFilters) -> tuple[sql.Composable, dict[str, Any]]:
    """SQL ``AND ...`` conditions for ``filters`` (amendments match their base form)."""
    parts: list[sql.Composable] = []
    params: dict[str, Any] = {}
    if filters.ciks:
        parts.append(sql.SQL("cik = ANY(%(ciks)s)"))
        params["ciks"] = list(filters.ciks)
    if filters.fiscal_years:
        parts.append(sql.SQL("fiscal_year = ANY(%(years)s)"))
        params["years"] = list(filters.fiscal_years)
    if filters.forms:
        parts.append(sql.SQL("regexp_replace(form, '/A$', '') = ANY(%(forms)s)"))
        params["forms"] = list(filters.forms)
    if filters.items:
        # 10-Q items are part-qualified ("II-1A"); "Item 1A" should match either form.
        parts.append(
            sql.SQL("(item = ANY(%(items)s) OR split_part(item, '-', 2) = ANY(%(items)s))")
        )
        params["items"] = list(filters.items)
    if not parts:
        return sql.SQL(""), params
    return sql.SQL(" AND ") + sql.SQL(" AND ").join(parts), params


def dense_search(
    conn: psycopg.Connection[Any],
    query_vec: Vector,
    embedding_model: str,
    filters: QueryFilters,
    k: int,
) -> list[RetrievedChunk]:
    """Top-``k`` chunks by cosine similarity, filtered first.

    Uses pgvector's iterative HNSW scan so restrictive filters (one company, one year)
    still return ``k`` rows instead of whatever survived the first ``ef_search`` probe.
    """
    where, params = filter_clause(filters)
    query = sql.SQL(
        "SELECT {cols}, embedding <=> %(q)s AS distance FROM chunks "
        "WHERE embedding IS NOT NULL AND embedding_model = %(model)s{where} "
        "ORDER BY embedding <=> %(q)s LIMIT %(k)s"
    ).format(cols=_SELECT, where=where)
    params |= {"q": query_vec, "model": embedding_model, "k": k}
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        cur.execute(query, params)
        rows = cur.fetchall()
    rows.sort(key=lambda r: (float(r[-1]), r[0]))  # relaxed_order may be slightly unsorted
    return [_row_to_chunk(r[:-1], 1.0 - float(r[-1]), "dense", i + 1) for i, r in enumerate(rows)]


def lexical_search(
    conn: psycopg.Connection[Any],
    tsquery_text: str,
    filters: QueryFilters,
    k: int,
) -> list[RetrievedChunk]:
    """Top-``k`` chunks by ``ts_rank_cd`` over the generated ``tsv`` column (GIN-indexed)."""
    if not tsquery_text.strip():
        return []
    where, params = filter_clause(filters)
    query = sql.SQL(
        "SELECT {cols}, ts_rank_cd(tsv, q) AS rank "
        "FROM chunks, websearch_to_tsquery('english', %(tsq)s) AS q "
        "WHERE tsv @@ q{where} ORDER BY rank DESC, chunk_id LIMIT %(k)s"
    ).format(cols=_SELECT, where=where)
    params |= {"tsq": tsquery_text, "k": k}
    rows = conn.execute(query, params).fetchall()
    return [_row_to_chunk(r[:-1], float(r[-1]), "lexical", i + 1) for i, r in enumerate(rows)]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Fuse ranked lists: ``score = sum(1 / (k + rank))``. Ties break on chunk_id."""
    scores: dict[str, float] = {}
    merged: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk.chunk_id in merged:
                prev = merged[chunk.chunk_id]
                merged[chunk.chunk_id] = replace(prev, ranks={**prev.ranks, **chunk.ranks})
            else:
                merged[chunk.chunk_id] = chunk
    ordered = sorted(merged, key=lambda cid: (-scores[cid], cid))
    return [replace(merged[cid], score=scores[cid]) for cid in ordered]
