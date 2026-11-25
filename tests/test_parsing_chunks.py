import itertools
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from filinglens.parsing.chunker import ChunkConfig, FilingInfo, chunk_section, context_header
from filinglens.parsing.pipeline import (
    PARSER_VERSION,
    chunks_path,
    coverage_table,
    quality_frame,
    run_parse,
    sections_path,
)
from filinglens.parsing.sections import extract_sections
from filinglens.parsing.text import count_tokens

FIXTURES = Path(__file__).parent / "fixtures" / "filings"
INFO = FilingInfo("0000999001-23-000010", "0000999001", "ACME", "10-K", 2023, "FY")


def risk_section():  # type: ignore[no-untyped-def]
    return extract_sections((FIXTURES / "tenk_ixbrl.htm").read_bytes(), "10-K").sections["1A"]


def test_default_config_gives_one_chunk_per_risk_factor() -> None:
    chunks = chunk_section(risk_section(), INFO)
    assert [c.heading for c in chunks] == [
        None,  # Item heading + intro
        "The Company depends on a limited number of suppliers for key components, and supply "
        "disruptions could materially harm its business.",
        "Global economic conditions could materially adversely affect the Company.",
        "The Company is subject to complex and changing laws and regulations worldwide.",
    ]
    assert chunks[1].text.startswith("Macroeconomic and Industry Risks\n\nThe Company depends")
    assert chunks[1].context_header == (
        "[ACME | 10-K | FY2023 | Item 1A Risk Factors | The Company depends on a limited "
        "number of suppliers for key components, and supply disruptions could materially "
        "harm its business.]"
    )


@pytest.mark.parametrize(
    "cfg",
    [
        ChunkConfig(),
        ChunkConfig(max_tokens=60, overlap_tokens=15, min_tokens=10),
        ChunkConfig(max_tokens=40, overlap_tokens=8, min_tokens=0),
        ChunkConfig(max_tokens=120, overlap_tokens=15, strategy="fixed"),
    ],
)
def test_chunk_invariants(cfg: ChunkConfig) -> None:
    section = risk_section()
    chunks = chunk_section(section, INFO, cfg)
    heading_tokens = max(
        count_tokens(section.text[s.start : s.end]) for s in section.spans if s.kind == "heading"
    )
    covered = set()
    for c in chunks:
        assert section.text[c.char_start : c.char_end] == c.text  # exact offsets
        assert c.token_count == count_tokens(c.text)
        # Only a heading glued to its first piece may exceed the limit.
        assert c.token_count <= cfg.max_tokens + heading_tokens
        assert c.chunk_id.startswith(f"{INFO.accession_no}_1A_")
        covered.update(range(c.char_start, c.char_end))
    # Every non-separator character of the section is in some chunk.
    assert all(i in covered for i, ch in enumerate(section.text) if not ch.isspace())
    assert [c.char_start for c in chunks] == sorted(c.char_start for c in chunks)


def test_long_sections_are_split_with_overlap() -> None:
    cfg = ChunkConfig(max_tokens=60, overlap_tokens=15, min_tokens=10)
    chunks = chunk_section(risk_section(), INFO, cfg)
    overlaps = [a.char_end > b.char_start for a, b in itertools.pairwise(chunks)]
    assert any(overlaps)
    supplier = [c for c in chunks if c.heading and c.heading.startswith("The Company depends")]
    assert len(supplier) == 2  # continuation chunk keeps its risk heading


def test_chunking_is_deterministic() -> None:
    a = chunk_section(risk_section(), INFO)
    b = chunk_section(risk_section(), INFO)
    assert a == b


def test_context_header_for_10q() -> None:
    info = FilingInfo("x", "1", "ACME", "10-Q", 2024, "Q2")
    assert (
        context_header(info, "II-1A", None)
        == "[ACME | 10-Q | FY2024 Q2 | Part II Item 1A Risk Factors]"
    )


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        ChunkConfig(max_tokens=50, overlap_tokens=50)


# --- pipeline ----------------------------------------------------------------------------


def _setup_processed(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    (processed / "filings").mkdir(parents=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for name, acc, form, period in [
        ("tenk_ixbrl.htm", "0000999001-23-000010", "10-K", date(2023, 9, 30)),
        ("tenq.htm", "0000999001-24-000020", "10-Q", date(2024, 3, 30)),
        ("tenk_legacy.htm", "0000999002-16-000005", "10-K", date(2015, 12, 31)),
    ]:
        shutil.copy(FIXTURES / name, raw / name)
        cik = acc[:10]
        rows.append(
            {
                "accession_no": acc,
                "cik": cik,
                "ticker": "ACME" if cik == "0000999001" else "BLUE",
                "form": form,
                "filing_date": period,
                "period_of_report": period,
                "local_path": str(raw / name),
                "is_amendment": False,
            }
        )
    pd.DataFrame(rows).to_parquet(processed / "filings" / "all.parquet")
    return processed


def test_run_parse_end_to_end_and_rerun_is_cached_and_identical(tmp_path: Path) -> None:
    processed = _setup_processed(tmp_path)
    first = run_parse(processed, ChunkConfig())
    assert [r.error for r in first] == [None, None, None]
    assert not any(r.cached for r in first)

    sections = pd.read_parquet(sections_path(processed, "0000999001", "0000999001-23-000010"))
    assert set(sections["parser_version"]) == {PARSER_VERSION}
    assert sections.set_index("item").loc["1A", "fiscal_year"] == 2023
    legacy = pd.read_parquet(sections_path(processed, "0000999002", "0000999002-16-000005"))
    assert set(legacy["fiscal_year"]) == {2015}  # period-of-report fallback

    chunks = pd.read_parquet(chunks_path(processed, "0000999001", "0000999001-24-000020"))
    assert set(chunks["item"]) == {"I-2", "I-3", "II-1", "II-1A"}
    assert set(chunks["fiscal_period"]) == {"Q2"}

    second = run_parse(processed, ChunkConfig())
    assert all(r.cached for r in second)
    again = pd.read_parquet(chunks_path(processed, "0000999001", "0000999001-24-000020"))
    pd.testing.assert_frame_equal(chunks, again)

    forced = run_parse(processed, ChunkConfig(), force=True, workers=2)
    assert not any(r.cached for r in forced)
    assert [r.n_chunks for r in forced] == [r.n_chunks for r in first]


def test_quality_and_coverage(tmp_path: Path) -> None:
    results = run_parse(_setup_processed(tmp_path), ChunkConfig())
    coverage = coverage_table(quality_frame(results)).set_index(["base_form", "item"])
    assert coverage.loc[("10-K", "1A"), "found_pct"] == 100.0
    assert coverage.loc[("10-K", "7"), "filings"] == 2
    assert coverage.loc[("10-Q", "II-1A"), "found_pct"] == 100.0
    # Fixtures are tiny, so Item 7 is flagged as suspiciously short (< 1000 chars).
    assert coverage.loc[("10-K", "7"), "short_pct"] == 100.0


def test_missing_document_is_reported_not_raised(tmp_path: Path) -> None:
    processed = _setup_processed(tmp_path)
    (tmp_path / "raw" / "tenq.htm").unlink()
    results = run_parse(processed, ChunkConfig())
    errors = {r.accession_no: r.error for r in results if r.error}
    assert list(errors) == ["0000999001-24-000020"]
    assert "not on disk" in errors["0000999001-24-000020"]
