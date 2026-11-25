from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import respx

from filinglens.ingestion.facts import FACT_COLUMNS, flatten_company_facts
from filinglens.ingestion.filings import FilingRecord, list_filings, parse_acceptance
from filinglens.ingestion.pipeline import IngestSummary, run_ingest
from filinglens.ingestion.universe import (
    Company,
    archive_cik,
    normalize_ticker,
    pad_cik,
    resolve_universe,
    resolve_universe_file,
)
from tests.edgar_fakes import AAPL_CIK, COMPANYFACTS_JSON, make_client, mount_sec

AAPL = Company(ticker="AAPL", cik=AAPL_CIK, name="Apple Inc.")


class FakeSink:
    def __init__(self) -> None:
        self.filings: dict[str, FilingRecord] = {}
        self.facts: dict[str, pd.DataFrame] = {}

    def upsert_filings(self, records: Sequence[FilingRecord]) -> int:
        self.filings.update({r.accession_no: r for r in records})
        return len(records)

    def replace_facts(self, cik: str, facts: pd.DataFrame) -> int:
        self.facts[cik] = facts
        return len(facts)


def test_cik_padding() -> None:
    assert pad_cik(320193) == "0000320193"
    assert pad_cik("0000320193") == "0000320193"
    assert archive_cik("0000320193") == "320193"


def test_ticker_normalization_and_universe_paths(tmp_path: Path) -> None:
    assert normalize_ticker(" brk.b ") == "BRK-B"
    assert resolve_universe_file("pilot", Path("config")) == Path("config/pilot_universe.csv")
    assert resolve_universe_file("custom.csv") == Path("custom.csv")


def test_acceptance_time_is_eastern_converted_to_utc() -> None:
    # 18:08 ET on 2023-11-02 (EDT, UTC-4), despite the trailing "Z".
    assert parse_acceptance("2023-11-02T18:08:27.000Z") == datetime(
        2023, 11, 2, 22, 8, 27, tzinfo=UTC
    )
    # Winter: EST, UTC-5.
    assert parse_acceptance("2024-01-10T16:00:00.000Z") == datetime(2024, 1, 10, 21, 0, tzinfo=UTC)
    assert parse_acceptance("") is None


@respx.mock
def test_universe_resolution_reports_unknown_tickers(tmp_path: Path) -> None:
    mount_sec(respx.mock)
    universe = tmp_path / "test_universe.csv"
    universe.write_text("ticker,sector\nAAPL,Tech\nBRK.B,Financials\nZZZZ,Nope\n")
    with make_client(tmp_path) as client:
        companies, unresolved = resolve_universe(client, universe, snapshot_dir=tmp_path / "u")
    assert [(c.ticker, c.cik, c.sector) for c in companies] == [
        ("AAPL", "0000320193", "Tech"),
        ("BRK-B", "0001067983", "Financials"),
    ]
    assert unresolved == ["ZZZZ"]
    assert (tmp_path / "u" / "test_ciks.csv").read_text().startswith("ticker,cik,name,sector")


@respx.mock
def test_list_filings_filters_forms_years_and_follows_pages(tmp_path: Path) -> None:
    mount_sec(respx.mock)
    with make_client(tmp_path) as client:
        records = list_filings(client, AAPL, ["10-K", "10-Q"], since=2016)

    by_acc = {r.accession_no: r for r in records}
    assert set(by_acc) == {
        "0001628280-16-020309",  # from the older page
        "0000320193-23-000077",
        "0000320193-23-000106",
        "0000320193-23-000110",  # amendment, flagged
    }
    tenk = by_acc["0000320193-23-000106"]
    assert tenk.form == "10-K"
    assert tenk.period_of_report == date(2023, 9, 30)
    assert tenk.primary_doc_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    )
    assert not tenk.is_amendment
    assert by_acc["0000320193-23-000110"].is_amendment
    # The 1994-2001 page ends before `since` and must not be requested.
    assert not any("submissions-002" in str(c.request.url) for c in respx.calls)


@respx.mock
def test_list_filings_can_exclude_amendments(tmp_path: Path) -> None:
    mount_sec(respx.mock)
    with make_client(tmp_path) as client:
        records = list_filings(client, AAPL, ["10-K"], since=2023, include_amendments=False)
    assert [r.accession_no for r in records] == ["0000320193-23-000106"]


def test_flatten_company_facts_keeps_restatements_and_types() -> None:
    df = flatten_company_facts(COMPANYFACTS_JSON, AAPL_CIK)
    assert list(df.columns) == FACT_COLUMNS
    assert len(df) == 4
    revenue = df[df.concept == "Revenues"].sort_values("filed_date")
    assert list(revenue.filed_date) == [date(2023, 11, 3), date(2024, 11, 1)]
    assert revenue.iloc[0].value == 383285000000.0
    assert revenue.iloc[0].frame == "CY2023"
    shares = df[df.concept == "EntityCommonStockSharesOutstanding"].iloc[0]
    assert shares.taxonomy == "dei" and pd.isna(shares.period_start)


@respx.mock
def test_run_ingest_end_to_end_and_rerun_downloads_nothing(tmp_path: Path) -> None:
    mount_sec(respx.mock)
    universe = tmp_path / "t_universe.csv"
    universe.write_text("ticker\nAAPL\n")
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    sink = FakeSink()

    def run() -> IngestSummary:
        with make_client(tmp_path) as client:
            return run_ingest(
                client,
                universe,
                forms=["10-K", "10-Q"],
                since=2015,
                raw_dir=raw,
                processed_dir=processed,
                sink=sink,
            )

    first = run()
    assert first.ok
    assert first.total("filings") == 5
    assert first.total("downloaded") == 5
    assert first.total("fact_rows") == 4
    assert len(sink.filings) == 5
    assert all(r.local_path and Path(r.local_path).exists() for r in sink.filings.values())
    assert len(pd.read_parquet(processed / "filings" / f"{AAPL_CIK}.parquet")) == 5
    assert len(pd.read_parquet(processed / "facts" / f"{AAPL_CIK}.parquet")) == 4

    calls_after_first = respx.mock.calls.call_count
    second = run()
    assert second.network_requests == 0
    assert second.total("downloaded") == 0
    assert second.total("skipped") == 5
    assert respx.mock.calls.call_count == calls_after_first


@respx.mock
def test_company_without_xbrl_facts_still_ingests_filings(tmp_path: Path) -> None:
    mount_sec(respx.mock)
    respx.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{AAPL_CIK}.json").respond(404)
    universe = tmp_path / "t_universe.csv"
    universe.write_text("ticker\nAAPL\n")
    with make_client(tmp_path) as client:
        summary = run_ingest(
            client,
            universe,
            forms=["10-K"],
            since=2023,
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
        )
    assert summary.ok
    assert summary.companies[0].fact_rows == 0
    assert summary.companies[0].filings == 2


@respx.mock
def test_since_in_future_yields_no_filings(tmp_path: Path) -> None:
    mount_sec(respx.mock)
    with make_client(tmp_path) as client:
        assert list_filings(client, AAPL, ["10-K"], since=2030) == []
