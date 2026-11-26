"""Build one isolated retrieval index per index config, in its own Postgres schema.

``eval_<fingerprint>.chunks`` is created ``LIKE public.chunks INCLUDING ALL`` (generated
``tsv`` column and HNSW/GIN indexes included), filled by re-chunking the cached sections
parquet, and embedded with or without context headers. Production ``public.chunks`` is
never touched. Connections use ``search_path = eval_x, public`` so the app's unqualified
SQL reads this index while ``financial_facts``/``llm_calls`` still resolve to ``public``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from evals.config import IndexConfig
from filinglens.parsing.chunker import DEFAULT_CHUNK_ITEMS, ChunkConfig, FilingInfo, chunk_section
from filinglens.parsing.items import base_form
from filinglens.parsing.pipeline import CHUNK_COLUMNS, chunks_frame, sections_from_frame
from filinglens.retrieval.embedder import Embedder
from filinglens.retrieval.indexer import embed_pending_chunks

logger = logging.getLogger(__name__)


def schema_for(index: IndexConfig) -> str:
    """Postgres schema name for an index config."""
    return f"eval_{index.fingerprint()}"


def connect_index(database_url: str, schema: str) -> psycopg.Connection[Any]:
    """Autocommit connection that resolves ``chunks`` to the eval schema."""
    conn = psycopg.connect(database_url, autocommit=True)
    conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
    register_vector(conn)
    return conn


def _is_built(conn: psycopg.Connection[Any], schema: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
        (schema, "_index_meta"),
    ).fetchone()
    if row is None:
        return False
    done = conn.execute(
        sql.SQL("SELECT complete FROM {}._index_meta LIMIT 1").format(sql.Identifier(schema))
    ).fetchone()
    return bool(done and done[0])


def iter_chunk_frames(processed_dir: Path, cfg: ChunkConfig) -> list[pd.DataFrame]:
    """Re-chunk every cached sections parquet with ``cfg`` (no HTML re-parsing)."""
    frames = []
    for path in sorted((processed_dir / "sections").glob("*/*.parquet")):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        first = df.iloc[0]
        fy = None if pd.isna(first["fiscal_year"]) else int(first["fiscal_year"])
        fp = (
            None
            if first["fiscal_period"] is None or pd.isna(first["fiscal_period"])
            else str(first["fiscal_period"])
        )
        info = FilingInfo(
            str(first["accession_no"]),
            str(first["cik"]),
            str(first["ticker"]),
            str(first["form"]),
            fy,
            fp,
        )
        wanted = set(DEFAULT_CHUNK_ITEMS.get(base_form(info.form), ()))
        chunks = [
            c
            for s in sections_from_frame(df)
            if s.item in wanted
            for c in chunk_section(s, info, cfg)
        ]
        if chunks:
            frames.append(chunks_frame(chunks))
    return frames


def ensure_index(
    database_url: str,
    processed_dir: Path,
    index: IndexConfig,
    embedder_factory: Callable[[str], Embedder],
    *,
    rebuild: bool = False,
) -> str:
    """Build (or reuse) the schema for ``index``; returns the schema name."""
    schema = schema_for(index)
    ident = sql.Identifier(schema)
    conn = connect_index(database_url, schema)
    try:
        if _is_built(conn, schema) and not rebuild:
            logger.info("reusing index %s", schema)
            return schema
        logger.info("building index %s (%s)", schema, index.model_dump())
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(ident))
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(ident))
        conn.execute(
            sql.SQL("CREATE TABLE {}.chunks (LIKE public.chunks INCLUDING ALL)").format(ident)
        )
        conn.execute(
            sql.SQL(
                "CREATE TABLE {}._index_meta (config jsonb, complete boolean, "
                "built_at timestamptz DEFAULT now())"
            ).format(ident)
        )
        conn.execute(
            sql.SQL("INSERT INTO {}._index_meta (config, complete) VALUES (%s, false)").format(
                ident
            ),
            (json.dumps(index.model_dump()),),
        )
        cfg = ChunkConfig(
            max_tokens=index.chunk_max_tokens,
            overlap_tokens=index.chunk_overlap_tokens,
            min_tokens=index.chunk_min_tokens,
            strategy=index.chunk_strategy,
        )
        total = 0
        with conn.cursor() as cur:
            for frame in iter_chunk_frames(processed_dir, cfg):
                with cur.copy(
                    sql.SQL("COPY {}.chunks ({}) FROM STDIN").format(
                        ident, sql.SQL(", ").join(sql.Identifier(c) for c in CHUNK_COLUMNS)
                    )
                ) as cp:
                    for row in frame[CHUNK_COLUMNS].itertuples(index=False, name=None):
                        cp.write_row(tuple(None if pd.isna(v) else v for v in row))
                total += len(frame)
        if total == 0:
            raise RuntimeError(
                f"No chunks built from {processed_dir}/sections. Run `filinglens parse` first."
            )
        embed_pending_chunks(
            conn,
            embedder_factory(index.embedding_model),
            expected_dim=_vector_dim(conn),
            include_header=index.context_header,
        )
        conn.execute(sql.SQL("UPDATE {}._index_meta SET complete = true").format(ident))
        logger.info("built index %s with %d chunks", schema, total)
        return schema
    finally:
        conn.close()


def _vector_dim(conn: psycopg.Connection[Any]) -> int:
    row = conn.execute(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass "
        "AND attname = 'embedding'"
    ).fetchone()
    if row is None:
        raise RuntimeError("chunks.embedding column not found")
    return int(row[0])
