"""Embed chunks that have no embedding (or one from a different model) and store them."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import psycopg

from filinglens.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


def pending_count(conn: psycopg.Connection[Any], model_name: str) -> int:
    """Chunks still needing an embedding from ``model_name``."""
    row = conn.execute(
        "SELECT count(*) FROM chunks WHERE embedding IS NULL "
        "OR embedding_model IS DISTINCT FROM %s",
        (model_name,),
    ).fetchone()
    return int(row[0]) if row else 0


def embed_pending_chunks(
    conn: psycopg.Connection[Any],
    embedder: Embedder,
    *,
    expected_dim: int,
    batch_size: int = 256,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed ``context_header + text`` for pending chunks, committing per batch.

    Resumable: an interrupted run keeps finished batches, and re-running only picks up what
    is left. Raises if the model's dimension doesn't match the ``vector(n)`` column.
    """
    if embedder.dim != expected_dim:
        raise ValueError(
            f"{embedder.model_name} produces {embedder.dim}-d vectors but EMBEDDING_DIM / the "
            f"chunks.embedding column is {expected_dim}-d. Change the model or migrate the column."
        )
    total = pending_count(conn, embedder.model_name)
    if limit is not None:
        total = min(total, limit)
    done = 0
    while done < total:
        rows = conn.execute(
            "SELECT chunk_id, coalesce(context_header, ''), text FROM chunks "
            "WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM %s "
            "ORDER BY chunk_id LIMIT %s",
            (embedder.model_name, min(batch_size, total - done)),
        ).fetchall()
        if not rows:
            break
        texts = [f"{header}\n{text}" if header else text for _, header, text in rows]
        vectors = embedder.embed_documents(texts)
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET embedding = %s, embedding_model = %s WHERE chunk_id = %s",
                [
                    (vec, embedder.model_name, cid)
                    for (cid, _, _), vec in zip(rows, vectors, strict=True)
                ],
            )
        done += len(rows)
        if progress is not None:
            progress(done, total)
    return done
