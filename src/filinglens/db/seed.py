"""Seed bundles: move a pre-ingested corpus into a deployed database (specs/06).

Production never ingests from EDGAR on a user request. Instead, ingest/parse/embed
locally, ``seed-export`` a tarball, and ``seed-import`` it on the server. The bundle holds
the ``filings``, ``chunks`` (with embeddings) and ``financial_facts`` tables as Postgres
``COPY`` text (portable across Postgres/pgvector versions; no ``pg_dump`` needed) plus the
parsed sections and universe snapshot that ``compare`` and the company directory read.
"""

from __future__ import annotations

import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from filinglens import __version__
from filinglens.db.repositories import schema_sql

# Generated/serial columns are left out so the target regenerates them.
TABLES: dict[str, list[str]] = {
    "filings": [
        "accession_no",
        "cik",
        "ticker",
        "form",
        "filing_date",
        "acceptance_datetime",
        "period_of_report",
        "primary_doc_url",
        "local_path",
        "is_amendment",
    ],
    "chunks": [
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
        "token_count",
        "char_start",
        "char_end",
        "embedding",
        "embedding_model",
        "context_header",
    ],
    "financial_facts": [
        "cik",
        "taxonomy",
        "concept",
        "unit",
        "value",
        "period_start",
        "period_end",
        "fy",
        "fp",
        "form",
        "accession_no",
        "filed_date",
        "frame",
    ],
}
DATA_DIRS = ("processed/sections", "raw/universe")


def _copy_sql(table: str, direction: str) -> sql.Composed:
    cols = sql.SQL(", ").join(sql.Identifier(c) for c in TABLES[table])
    return sql.SQL("COPY {} ({}) " + direction).format(sql.Identifier(table), cols)


def export_seed(conn: psycopg.Connection[Any], data_dir: Path, out: Path) -> dict[str, Any]:
    """Write the bundle to ``out``; returns its manifest."""
    manifest: dict[str, Any] = {
        "filinglens": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "tables": {},
    }
    with tarfile.open(out, "w:gz") as tar:
        for table in TABLES:
            buf = io.BytesIO()
            with conn.cursor() as cur, cur.copy(_copy_sql(table, "TO STDOUT")) as cp:
                for block in cp:
                    buf.write(block)
            data = buf.getvalue()
            manifest["tables"][table] = data.count(b"\n")
            info = tarfile.TarInfo(f"db/{table}.copy")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for rel in DATA_DIRS:
            path = data_dir / rel
            if path.exists():
                tar.add(path, arcname=f"data/{rel}")
        models = conn.execute("SELECT embedding_model, count(*) FROM chunks GROUP BY 1").fetchall()
        manifest["embedding_models"] = {m or "none": n for m, n in models}
        blob = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    return manifest


def import_seed(
    conn: psycopg.Connection[Any],
    bundle: Path,
    data_dir: Path,
    *,
    replace: bool = False,
    if_empty: bool = False,
) -> dict[str, Any]:
    """Load a bundle. Refuses to overwrite non-empty tables unless ``replace``.

    ``if_empty``: always extract the data files, but load the tables only when the
    database is empty (for containers whose filesystem is wiped on restart).
    """
    with conn.transaction():
        conn.execute(schema_sql())
    existing = {
        t: conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(t))).fetchone()
        for t in TABLES
    }
    non_empty = [t for t, row in existing.items() if row and row[0]]
    load_tables = not (if_empty and non_empty)
    if non_empty and not replace and not if_empty:
        raise RuntimeError(f"Tables already have data: {non_empty}. Pass --replace to overwrite.")
    with tarfile.open(bundle, "r:gz") as tar:
        manifest: dict[str, Any] = json.loads(_read(tar, "manifest.json"))
        members = [m for m in tar.getmembers() if m.name.startswith("data/")]
        for m in members:
            m.name = m.name.removeprefix("data/")
        tar.extractall(data_dir, members=members, filter="data")  # blocks path traversal
        if load_tables:
            with conn.transaction():
                conn.execute("TRUNCATE chunks, financial_facts, filings")
                with conn.cursor() as cur:
                    for table in TABLES:
                        payload = _read(tar, f"db/{table}.copy")
                        with cur.copy(_copy_sql(table, "FROM STDIN")) as cp:
                            cp.write(payload)
    counts = {
        t: (
            conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(t))).fetchone()
            or (0,)
        )[0]
        for t in TABLES
    }
    manifest["imported"] = counts
    manifest["tables_loaded"] = load_tables
    return manifest


def _read(tar: tarfile.TarFile, name: str) -> bytes:
    member = tar.extractfile(name)
    if member is None:
        raise RuntimeError(f"{name} missing from seed bundle")
    return member.read()
