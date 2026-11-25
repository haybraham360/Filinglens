"""End-to-end ingestion: universe -> filings -> documents -> facts -> parquet -> Postgres."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd

from filinglens.ingestion.client import EdgarClient, EdgarError, EdgarForbiddenError
from filinglens.ingestion.facts import FACT_COLUMNS, fetch_company_facts, flatten_company_facts
from filinglens.ingestion.filings import (
    FILING_COLUMNS,
    FilingRecord,
    download_filings,
    list_filings,
)
from filinglens.ingestion.universe import Company, resolve_universe

logger = logging.getLogger(__name__)


class IngestSink(Protocol):
    """Destination for ingested rows (Postgres in production, a fake in tests)."""

    def upsert_filings(self, records: Sequence[FilingRecord]) -> int:
        """Insert or update filing metadata."""
        ...

    def replace_facts(self, cik: str, facts: pd.DataFrame) -> int:
        """Replace all facts for one company."""
        ...


@dataclass
class CompanyResult:
    """Per-company outcome."""

    ticker: str
    cik: str
    filings: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed_docs: int = 0
    fact_rows: int = 0
    error: str | None = None


@dataclass
class IngestSummary:
    """Totals for one ``ingest`` run."""

    companies: list[CompanyResult] = field(default_factory=list)
    unresolved_tickers: list[str] = field(default_factory=list)
    network_requests: int = 0
    cache_hits: int = 0
    bytes_downloaded: int = 0

    @property
    def ok(self) -> bool:
        """True when every ticker resolved and no company or document failed."""
        return not self.unresolved_tickers and all(
            c.error is None and c.failed_docs == 0 for c in self.companies
        )

    def total(self, attr: str) -> int:
        """Sum an integer attribute across companies."""
        return sum(int(getattr(c, attr)) for c in self.companies)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def filings_frame(records: Sequence[FilingRecord]) -> pd.DataFrame:
    """Filing records as a DataFrame with the ``filings`` table's columns."""
    return pd.DataFrame([r.to_row() for r in records], columns=FILING_COLUMNS)


def ingest_company(
    client: EdgarClient,
    company: Company,
    *,
    forms: Sequence[str],
    since: int,
    raw_dir: Path,
    processed_dir: Path,
    sink: IngestSink | None,
    workers: int,
    refresh: bool,
    include_amendments: bool,
) -> CompanyResult:
    """Ingest one company. EDGAR errors are captured on the result; 403s propagate."""
    result = CompanyResult(ticker=company.ticker, cik=company.cik)
    try:
        records = list_filings(
            client,
            company,
            forms,
            since,
            include_amendments=include_amendments,
            refresh=refresh,
        )
        report = download_filings(client, records, raw_dir, workers=workers)
        result.filings = len(report.records)
        result.downloaded = report.downloaded
        result.skipped = report.skipped
        result.failed_docs = len(report.failed)
        _write_parquet(
            filings_frame(report.records), processed_dir / "filings" / f"{company.cik}.parquet"
        )

        payload = fetch_company_facts(client, company.cik, refresh=refresh)
        if payload is None:
            logger.warning("%s (%s): no XBRL companyfacts", company.ticker, company.cik)
            facts = pd.DataFrame(columns=FACT_COLUMNS)
        else:
            facts = flatten_company_facts(payload, company.cik)
        result.fact_rows = len(facts)
        _write_parquet(facts, processed_dir / "facts" / f"{company.cik}.parquet")

        if sink is not None:
            sink.upsert_filings(report.records)
            sink.replace_facts(company.cik, facts)
    except EdgarForbiddenError:
        raise
    except EdgarError as exc:
        result.error = str(exc)
        logger.error("%s (%s): %s", company.ticker, company.cik, exc)
    return result


def run_ingest(
    client: EdgarClient,
    universe_file: Path,
    *,
    forms: Sequence[str],
    since: int,
    raw_dir: Path,
    processed_dir: Path,
    sink: IngestSink | None = None,
    tickers: Sequence[str] | None = None,
    workers: int = 4,
    refresh: bool = False,
    include_amendments: bool = True,
) -> IngestSummary:
    """Run ingestion for a universe. Safe to re-run: cached responses and files are reused.

    ``refresh`` re-fetches the mutable JSON endpoints (tickers, submissions, companyfacts)
    to pick up new filings; documents already on disk are never downloaded again.
    """
    summary = IngestSummary()
    companies, summary.unresolved_tickers = resolve_universe(
        client, universe_file, snapshot_dir=raw_dir / "universe", refresh=refresh
    )
    if tickers:
        wanted = {t.upper() for t in tickers}
        companies = [c for c in companies if c.ticker in wanted]

    for i, company in enumerate(companies, 1):
        logger.info("[%d/%d] %s (CIK %s)", i, len(companies), company.ticker, company.cik)
        result = ingest_company(
            client,
            company,
            forms=forms,
            since=since,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            sink=sink,
            workers=workers,
            refresh=refresh,
            include_amendments=include_amendments,
        )
        logger.info(
            "  %s: %d filings (%d new, %d cached, %d failed), %d fact rows",
            company.ticker,
            result.filings,
            result.downloaded,
            result.skipped,
            result.failed_docs,
            result.fact_rows,
        )
        summary.companies.append(result)

    summary.network_requests = client.stats.network_requests
    summary.cache_hits = client.stats.cache_hits
    summary.bytes_downloaded = client.stats.bytes_downloaded
    return summary
