"""Postgres writers for ingestion output (``filings`` and ``financial_facts``)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib import resources
from types import TracebackType
from typing import Any

import pandas as pd
import psycopg

from filinglens.ingestion.facts import FACT_COLUMNS
from filinglens.ingestion.filings import FILING_COLUMNS, FilingRecord
from filinglens.parsing.pipeline import CHUNK_COLUMNS

_UPSERT_FILINGS = f"""
INSERT INTO filings ({", ".join(FILING_COLUMNS)})
VALUES ({", ".join(["%s"] * len(FILING_COLUMNS))})
ON CONFLICT (accession_no) DO UPDATE SET
    {", ".join(f"{c} = EXCLUDED.{c}" for c in FILING_COLUMNS[1:] if c != "local_path")},
    local_path = COALESCE(EXCLUDED.local_path, filings.local_path)
"""


def schema_sql() -> str:
    """Return the bundled ``schema.sql`` (idempotent ``CREATE ... IF NOT EXISTS``)."""
    return resources.files("filinglens.db").joinpath("schema.sql").read_text()


def _clean(value: Any) -> Any:
    """Map pandas missing values (NaN/NaT/NA) to None for the driver."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


class PostgresSink:
    """Idempotent loader: upserts filings by accession and replaces facts per CIK."""

    def __init__(self, database_url: str) -> None:
        """Connect and make sure the schema exists."""
        self._conn = psycopg.connect(database_url)
        with self._conn.transaction():
            self._conn.execute(schema_sql())

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    def __enter__(self) -> PostgresSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def upsert_filings(self, records: Sequence[FilingRecord]) -> int:
        """Insert or update filing metadata. Returns the number of rows written."""
        if not records:
            return 0
        rows = [tuple(r.to_row()[c] for c in FILING_COLUMNS) for r in records]
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(_UPSERT_FILINGS, rows)
        return len(rows)

    def replace_facts(self, cik: str, facts: pd.DataFrame) -> int:
        """Replace all facts for ``cik`` in one transaction.

        companyfacts is a full snapshot per company, so delete-then-COPY keeps re-runs
        idempotent without a natural key (several columns are nullable).
        """
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute("DELETE FROM financial_facts WHERE cik = %s", (cik,))
            with cur.copy(f"COPY financial_facts ({', '.join(FACT_COLUMNS)}) FROM STDIN") as cp:
                for row in facts[FACT_COLUMNS].itertuples(index=False, name=None):
                    cp.write_row(tuple(_clean(v) for v in row))
        return len(facts)

    def filing_exists(self, accession_no: str) -> bool:
        """True if ``accession_no`` is in ``filings`` (chunks reference it)."""
        row = self._conn.execute(
            "SELECT 1 FROM filings WHERE accession_no = %s", (accession_no,)
        ).fetchone()
        return row is not None

    def replace_chunks(self, accession_no: str, chunks: pd.DataFrame) -> bool:
        """Replace a filing's chunks unless they are unchanged. Returns True if rewritten.

        Unchanged chunk sets are left alone so embeddings computed later (specs/03) are
        not wiped by re-running the parser.
        """
        existing = self._conn.execute(
            "SELECT chunk_id, md5(text), coalesce(context_header, '') FROM chunks "
            "WHERE accession_no = %s",
            (accession_no,),
        ).fetchall()
        new = {
            (
                str(r["chunk_id"]),
                hashlib.md5(str(r["text"]).encode()).hexdigest(),
                str(r["context_header"]),
            )
            for r in chunks.to_dict("records")
        }
        if set(existing) == new:
            return False
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE accession_no = %s", (accession_no,))
            with cur.copy(f"COPY chunks ({', '.join(CHUNK_COLUMNS)}) FROM STDIN") as cp:
                for row in chunks[CHUNK_COLUMNS].itertuples(index=False, name=None):
                    cp.write_row(tuple(_clean(v) for v in row))
        return True

    def count(self, table: str, cik: str | None = None) -> int:
        """Row count for ``filings``, ``financial_facts`` or ``chunks``, optionally for one CIK."""
        if table not in {"filings", "financial_facts", "chunks"}:
            raise ValueError(f"unknown table {table}")
        query = f"SELECT count(*) FROM {table}" + (" WHERE cik = %s" if cik else "")
        row = self._conn.execute(query, (cik,) if cik else ()).fetchone()
        return int(row[0]) if row else 0
