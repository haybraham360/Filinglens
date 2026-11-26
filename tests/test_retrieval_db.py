"""Retrieval against real Postgres + pgvector; skipped unless FILINGLENS_TEST_DATABASE_URL is set.

Use a dedicated database: these tests delete all rows in ``chunks`` and ``filings``.
"""

import os
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest

from filinglens.db.connection import connect
from filinglens.ingestion.filings import FilingRecord
from filinglens.parsing.chunker import FilingInfo, chunk_section
from filinglens.parsing.pipeline import chunks_frame
from filinglens.parsing.sections import extract_sections
from filinglens.retrieval.embedder import HashingEmbedder
from filinglens.retrieval.filters import CompanyDirectory
from filinglens.retrieval.indexer import embed_pending_chunks, pending_count
from filinglens.retrieval.retriever import RetrievalConfig, Retriever

URL = os.environ.get("FILINGLENS_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not URL, reason="set FILINGLENS_TEST_DATABASE_URL to run DB tests"),
]
FIXTURES = Path(__file__).parent / "fixtures" / "filings"
FILINGS = [
    ("tenk_ixbrl.htm", "10-K", "0000999001-23-000010", "0000999001", "ACME", 2023, "FY"),
    ("tenq.htm", "10-Q", "0000999001-24-000020", "0000999001", "ACME", 2024, "Q2"),
    ("tenk_legacy.htm", "10-K", "0000999002-16-000005", "0000999002", "BLUE", 2015, "FY"),
]


@pytest.fixture(scope="module")
def conn() -> Iterator[psycopg.Connection[Any]]:
    from filinglens.db.connection import connect
    from filinglens.db.repositories import PostgresSink

    assert URL
    c = connect(URL)
    c.execute("DELETE FROM chunks")
    c.execute("DELETE FROM filings")
    sink = PostgresSink(URL)
    for name, form, acc, cik, ticker, fy, fp in FILINGS:
        sink.upsert_filings(
            [FilingRecord(acc, cik, ticker, form, date(fy, 12, 1), None, None, None, None, False)]
        )
        secs = extract_sections((FIXTURES / name).read_bytes(), form).sections
        info = FilingInfo(acc, cik, ticker, form, fy, fp)
        chunks = [ch for s in secs.values() for ch in chunk_section(s, info)]
        sink.replace_chunks(acc, chunks_frame(chunks))
    sink.close()
    embed_pending_chunks(c, HashingEmbedder(), expected_dim=384, batch_size=7)
    yield c
    c.execute("DELETE FROM chunks")
    c.execute("DELETE FROM filings")
    c.close()


@pytest.fixture
def directory(tmp_path: Path) -> CompanyDirectory:
    csv = tmp_path / "t_ciks.csv"
    csv.write_text(
        "ticker,cik,name,sector\nACME,0000999001,Acme Devices Inc,IT\n"
        "BLUE,0000999002,Bluewater Energy Corp,Energy\n"
    )
    return CompanyDirectory.load(csv)


def make(conn: psycopg.Connection[Any], directory: CompanyDirectory, **cfg: Any) -> Retriever:
    return Retriever(conn, HashingEmbedder(), directory, RetrievalConfig(**cfg))


def test_all_chunks_embedded_and_rerun_is_noop(conn: psycopg.Connection[Any]) -> None:
    assert pending_count(conn, "hashing-bow-v1") == 0
    assert embed_pending_chunks(conn, HashingEmbedder(), expected_dim=384) == 0
    with pytest.raises(ValueError, match="384"):
        embed_pending_chunks(conn, HashingEmbedder(dim=8), expected_dim=384)


def test_company_filter_keeps_results_on_that_company(
    conn: psycopg.Connection[Any], directory: CompanyDirectory
) -> None:
    result = make(conn, directory).retrieve("What legal proceedings does Bluewater face?")
    assert result.filters.tickers == ("BLUE",)
    assert result.chunks and {c.ticker for c in result.chunks} == {"BLUE"}
    assert any("Louisiana" in c.text for c in result.chunks[:3])


def test_dense_only_vs_hybrid(conn: psycopg.Connection[Any], directory: CompanyDirectory) -> None:
    q = "ACME semiconductor shortages and contract manufacturers in Malaysia"
    dense = make(conn, directory, use_hybrid=False, top_k=3).retrieve(q)
    hybrid = make(conn, directory, use_hybrid=True, top_k=3).retrieve(q)
    assert len(dense.chunks) == 3 and all(set(c.ranks) == {"dense"} for c in dense.chunks)
    assert "lexical" in hybrid.chunks[0].ranks
    assert "Malaysia" in hybrid.chunks[0].text
    assert "OR" in hybrid.lexical_query
    assert {"embed", "dense", "lexical", "total"} <= set(hybrid.timings_ms)


def test_year_form_and_item_filters(
    conn: psycopg.Connection[Any], directory: CompanyDirectory
) -> None:
    r = make(conn, directory).retrieve("ACME export restrictions risk factors 10-Q fiscal 2024")
    assert {(c.form, c.fiscal_year) for c in r.chunks} == {("10-Q", 2024)}
    r = make(conn, directory).retrieve("ACME Item 1A risks in the 2024 quarterly report")
    assert {c.item for c in r.chunks} == {"II-1A"}


def test_impossible_filters_are_relaxed_but_company_is_kept(
    conn: psycopg.Connection[Any], directory: CompanyDirectory
) -> None:
    r = make(conn, directory).retrieve("Bluewater hedging risk in fiscal 2021")
    assert r.relaxed == ["fiscal_years"]
    assert r.chunks and {c.ticker for c in r.chunks} == {"BLUE"}


def test_sink_writes_are_committed_and_visible_to_other_connections() -> None:
    """Regression: filing_exists() before replace_chunks() used to leave writes uncommitted."""
    from filinglens.db.connection import connect
    from filinglens.db.repositories import PostgresSink

    assert URL
    acc = "0000999003-20-000001"
    info = FilingInfo(acc, "0000999003", "REGR", "10-K", 2020, "FY")
    secs = extract_sections((FIXTURES / "tenk_ixbrl.htm").read_bytes(), "10-K").sections
    frame = chunks_frame(chunk_section(secs["1A"], info))
    with PostgresSink(URL) as sink:
        sink.upsert_filings(
            [
                FilingRecord(
                    acc,
                    "0000999003",
                    "REGR",
                    "10-K",
                    date(2020, 12, 1),
                    None,
                    None,
                    None,
                    None,
                    False,
                )
            ]
        )
        assert sink.filing_exists(acc)
        assert sink.replace_chunks(acc, frame)
        other = connect(URL, apply_schema=False)
        try:
            row = other.execute("SELECT count(*) FROM chunks WHERE accession_no = %s", (acc,))
            assert row.fetchone() == (len(frame),)
        finally:
            other.execute("DELETE FROM chunks WHERE accession_no = %s", (acc,))
            other.execute("DELETE FROM filings WHERE accession_no = %s", (acc,))
            other.close()


def test_get_financials_from_postgres_matches_frame_store() -> None:
    """financial_facts round trip: numeric -> Decimal, dates intact, same selections."""
    from filinglens.agent.financials import FrameFactStore, PostgresFactStore, get_financials
    from filinglens.db.connection import connect
    from filinglens.db.repositories import PostgresSink
    from tests.agent_fakes import CIKS, facts_frame

    assert URL
    df = facts_frame()
    cik = CIKS["AAPL"]
    with PostgresSink(URL) as sink:
        sink.replace_facts(cik, df[df["cik"] == cik])
    c = connect(URL, apply_schema=False)
    try:
        args = (cik, ["revenue", "net_income", "current_assets"], [2022, 2023])
        db_vals, db_missing = get_financials(PostgresFactStore(c), *args, as_of="latest")
        fr_vals, fr_missing = get_financials(FrameFactStore(df), *args, as_of="latest")
        assert [v.to_dict() for v in db_vals] == [v.to_dict() for v in fr_vals]
        assert db_missing == fr_missing == []
    finally:
        c.execute("DELETE FROM financial_facts WHERE cik = %s", (cik,))
        c.close()


def test_eval_runner_retrieval_mode_against_isolated_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evals.run end to end: builds eval_<fp>.chunks from parsed sections, never public.chunks."""
    import json

    import evals.run as run_mod
    import yaml
    from evals.dataset import Case, GoldSpan, write_jsonl

    from filinglens.config import get_settings
    from filinglens.parsing.chunker import ChunkConfig
    from filinglens.parsing.pipeline import run_parse
    from tests.test_parsing_chunks import _setup_processed

    assert URL
    processed = _setup_processed(tmp_path)
    run_parse(processed, ChunkConfig())
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(run_mod, "REPORTS_DIR", tmp_path / "reports")
    get_settings.cache_clear()
    sec = pd.read_parquet(processed / "sections" / "0000999001" / "0000999001-23-000010.parquet")
    text = sec.set_index("item").loc["1A", "text"]
    start = text.index("Many components")
    cases = [
        Case(
            id="r1",
            question="ACME semiconductor shortages and single-source components",
            type="factual",
            gold_answer="x",
            ticker="ACME",
            split="dev",
            verified_by="t",
            gold_spans=[
                GoldSpan(
                    accession_no="0000999001-23-000010",
                    item="1A",
                    char_start=start,
                    char_end=start + 200,
                )
            ],
        ),
    ]
    data = tmp_path / "golden_t.jsonl"
    write_jsonl(data, cases)
    cfg_path = tmp_path / "r.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "r",
                "mode": "retrieval",
                "concurrency": 1,
                "index": {"embedding_model": "hashing-bow-v1"},
                "retrieval": {"use_filters": False},
            }
        )
    )
    pub_before = connect(URL, apply_schema=False).execute("SELECT count(*) FROM chunks").fetchone()
    try:
        code = run_mod.main(
            ["--dataset", str(data), "--split", "dev", "--config", str(cfg_path), "--run-id", "t1"]
        )
        assert code == 0
        summary = json.loads((tmp_path / "reports" / "t1.json").read_text())
        assert summary["counts"]["ok"] == 1 and summary["counts"]["open_errors"] == 0
        assert summary["metrics"]["recall_at_10"]["mean"] == 1.0
        schema = summary["meta"]["index_schema"]
        c = connect(URL, apply_schema=False)
        n = c.execute(f'SELECT count(*), count(embedding) FROM "{schema}".chunks').fetchone()
        assert n is not None and n[0] > 0 and n[0] == n[1]
        assert c.execute("SELECT count(*) FROM chunks").fetchone() == pub_before
        # second run reuses the built index and resumes (nothing re-scored)
        assert (
            run_mod.main(
                [
                    "--dataset",
                    str(data),
                    "--split",
                    "dev",
                    "--config",
                    str(cfg_path),
                    "--run-id",
                    "t1",
                ]
            )
            == 0
        )
        rows = (tmp_path / "reports" / "runs" / "t1" / "results.jsonl").read_text().splitlines()
        assert len(rows) == 1
        c.execute(f'DROP SCHEMA "{schema}" CASCADE')
        c.close()
    finally:
        get_settings.cache_clear()
