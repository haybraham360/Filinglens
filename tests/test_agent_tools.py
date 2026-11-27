from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from filinglens.agent.diff import SectionStore, align, units
from filinglens.agent.financials import FrameFactStore, fiscal_year_of, get_financials
from filinglens.agent.guards import unsupported_numbers
from filinglens.agent.ratios import RATIOS
from filinglens.agent.sources import SourceRegistry
from filinglens.agent.tools import TOOLS, AgentContext, run_tool
from filinglens.parsing.blocks import Block
from filinglens.parsing.chunker import FilingInfo
from filinglens.parsing.pipeline import sections_frame
from filinglens.parsing.sections import Section
from filinglens.research.similarity import (
    all_measures,
    change_ratio,
    cosine_tf,
    feature_scale_simple,
    jaccard,
    min_edit,
    simple,
)
from filinglens.retrieval.embedder import HashingEmbedder
from filinglens.retrieval.filters import CompanyDirectory, QueryFilters
from filinglens.retrieval.retriever import RetrievalResult
from filinglens.retrieval.search import RetrievedChunk
from tests.agent_fakes import CIKS, RESTATED_AAPL_NI_2022, facts_frame

CONCEPT_MAP = Path(__file__).resolve().parents[1] / "config" / "concept_map.yaml"


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, QueryFilters]] = []

    def retrieve(self, question: str, filters: QueryFilters | None = None) -> RetrievalResult:
        assert filters is not None
        self.calls.append((question, filters))
        chunk = RetrievedChunk(
            chunk_id="0000320193-23-000106_7_0003",
            accession_no="0000320193-23-000106",
            cik="0000320193",
            ticker="AAPL",
            form="10-K",
            fiscal_year=2023,
            fiscal_period="FY",
            item="7",
            heading="Gross Margin",
            text="Products gross margin decreased due to the weaker U.S. dollar.",
            context_header="[AAPL]",
            char_start=10,
            char_end=80,
        )
        return RetrievalResult(chunks=[chunk], filters=filters)


def make_ctx(tmp_path: Path, sections: SectionStore | None = None) -> AgentContext:
    directory = CompanyDirectory(
        by_ticker={**CIKS, "ACME": "0000999001"}, by_alias={"apple": CIKS["AAPL"]}
    )
    return AgentContext(
        retriever=FakeRetriever(),  # type: ignore[arg-type]
        sections=sections or SectionStore(tmp_path),
        facts=FrameFactStore(facts_frame()),
        directory=directory,
        registry=SourceRegistry(),
        embedder=HashingEmbedder(),
        concept_map_path=str(CONCEPT_MAP),
    )


# --- compute_ratio: hand-computed values, 3 companies x 3 years ------------------------------
# Expected values computed by hand from the fixture figures (USD millions), 6 dp half-even.

HAND = {
    ("AAPL", "gross_margin"): {2021: "0.417794", 2022: "0.433096", 2023: "0.441311"},
    ("AAPL", "operating_margin"): {2021: "0.297824", 2022: "0.302887", 2023: "0.298214"},
    ("AAPL", "net_margin"): {2021: "0.258818", 2022: "0.253096", 2023: "0.253062"},
    ("MSFT", "gross_margin"): {2021: "0.689258", 2022: "0.684017", 2023: "0.689201"},
    ("MSFT", "operating_margin"): {2021: "0.415949", 2022: "0.420553", 2023: "0.417729"},
    ("MSFT", "net_margin"): {2021: "0.364517", 2022: "0.366863", 2023: "0.341462"},
    ("NVDA", "gross_margin"): {2022: "0.649290", 2023: "0.569289", 2024: "0.727176"},
    ("NVDA", "operating_margin"): {2022: "0.373077", 2023: "0.156595", 2024: "0.541217"},
    ("NVDA", "net_margin"): {2022: "0.362339", 2023: "0.161934", 2024: "0.488493"},
    ("AAPL", "revenue_growth_yoy"): {2022: "0.077938", 2023: "-0.028005"},
    ("MSFT", "revenue_growth_yoy"): {2022: "0.179561", 2023: "0.068820"},
    ("NVDA", "revenue_growth_yoy"): {2023: "0.002229", 2024: "1.258545"},
    ("AAPL", "current_ratio"): {2022: "0.879356", 2023: "0.988012"},
    ("AAPL", "roe"): {2023: "1.560760"},
}


@pytest.mark.parametrize(("ticker", "ratio"), sorted(HAND))
def test_compute_ratio_matches_hand_calculation(tmp_path: Path, ticker: str, ratio: str) -> None:
    expected = HAND[(ticker, ratio)]
    out, is_error = run_tool(
        make_ctx(tmp_path),
        "compute_ratio",
        {"ticker": ticker, "ratio": ratio, "fiscal_years": sorted(expected)},
    )
    assert not is_error, out
    assert out["values_by_year"] == {str(y): v for y, v in expected.items()}
    assert out["formula"] == RATIOS[ratio].formula
    assert out["ref"].startswith("T")


def test_restatement_original_vs_latest() -> None:
    store = FrameFactStore(facts_frame())
    original, _ = get_financials(store, CIKS["AAPL"], ["net_income"], [2022], as_of="original")
    latest, _ = get_financials(store, CIKS["AAPL"], ["net_income"], [2022], as_of="latest")
    assert original[0].value == Decimal("99803000000.0")
    assert original[0].accession_no == "0000320193-22-000010"
    assert latest[0].value == Decimal(f"{RESTATED_AAPL_NI_2022 * 1_000_000}.0")
    assert latest[0].accession_no == "0000320193-23-000010"


def test_comparatives_get_period_fiscal_year_not_filing_fy() -> None:
    values, missing = get_financials(
        FrameFactStore(facts_frame()), CIKS["NVDA"], ["revenue"], [2022, 2023, 2024, 2019]
    )
    assert {v.fiscal_year: v.period_end for v in values} == {
        2022: date(2022, 1, 30),
        2023: date(2023, 1, 29),
        2024: date(2024, 1, 28),
    }
    assert all(v.xbrl_tag == "RevenueFromContractWithCustomerExcludingAssessedTax" for v in values)
    assert missing == ["revenue FY2019"]


def test_quarterly_values_keep_only_the_filings_own_quarter() -> None:
    values, _ = get_financials(
        FrameFactStore(facts_frame()), CIKS["AAPL"], ["revenue"], [2024], period="Q"
    )
    assert [(v.fiscal_period, v.value, v.period_end) for v in values] == [
        ("Q2", Decimal("90753000000.0"), date(2024, 3, 30))
    ]


def test_fiscal_year_of_52_53_week_calendars() -> None:
    assert fiscal_year_of(date(2022, 1, 1)) == 2021
    assert fiscal_year_of(date(2024, 1, 28)) == 2024
    assert fiscal_year_of(date(2023, 9, 30)) == 2023


def test_missing_inputs_give_none_not_a_guess(tmp_path: Path) -> None:
    out, is_error = run_tool(
        make_ctx(tmp_path),
        "compute_ratio",
        {"ticker": "MSFT", "ratio": "current_ratio", "fiscal_years": [2023]},
    )
    assert not is_error
    assert out["values_by_year"] == {"2023": None} and out["unavailable_years"] == [2023]


# --- validation and errors ----------------------------------------------------------------


def test_tool_definitions_are_strict_with_closed_schemas() -> None:
    for tool in TOOLS:
        d = tool.definition()
        assert d["strict"] is True
        assert d["input_schema"]["additionalProperties"] is False


@pytest.mark.parametrize(
    ("name", "args", "needle"),
    [
        (
            "compute_ratio",
            {"ticker": "AAPL", "ratio": "magic", "fiscal_years": [2023]},
            "Unknown ratio",
        ),
        (
            "compute_ratio",
            {"ticker": "ZZZZ", "ratio": "roe", "fiscal_years": [2023]},
            "Unknown ticker",
        ),
        (
            "get_financials",
            {"ticker": "AAPL", "concepts": ["vibes"], "fiscal_years": [2023]},
            "unknown concepts",
        ),
        ("get_financials", {"ticker": "AAPL", "fiscal_years": "2023"}, "Invalid arguments"),
        ("no_such_tool", {}, "Unknown tool"),
    ],
)
def test_bad_calls_return_helpful_errors(tmp_path: Path, name: str, args: Any, needle: str) -> None:
    out, is_error = run_tool(make_ctx(tmp_path), name, args)
    assert is_error and needle in str(out)


def test_search_filings_passes_filters_and_registers_handles(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    out, is_error = run_tool(
        ctx,
        "search_filings",
        {"query": "gross margin drivers", "ticker": "aapl", "fiscal_year": 2023, "item": "7"},
    )
    assert not is_error
    _, filters = ctx.retriever.calls[0]  # type: ignore[attr-defined]
    assert filters.ciks == (CIKS["AAPL"],) and filters.fiscal_years == (2023,)
    assert filters.items == ("7",)
    assert out["results"][0]["ref"] == "C1"
    src = ctx.registry.get("C1")
    assert src is not None and src.chunk_id == "0000320193-23-000106_7_0003"
    # same chunk again -> same handle
    run_tool(ctx, "search_filings", {"query": "again", "ticker": "AAPL"})
    assert list(ctx.registry.sources) == ["C1"]


# --- compare_sections -----------------------------------------------------------------------


def _section(item: str, blocks: list[tuple[str, str]]) -> Section:
    return Section.from_blocks(item, "Risk Factors", [Block(k, t) for k, t in blocks], "native")  # type: ignore[arg-type]


OLD_1A = [
    ("heading", "Item 1A. Risk Factors"),
    ("heading", "We depend on a limited number of suppliers."),
    (
        "para",
        "Many components are obtained from single or limited sources, and supply "
        "disruptions could harm our business and results of operations.",
    ),
    ("heading", "Global economic conditions could adversely affect us."),
    (
        "para",
        "Demand for industrial equipment is cyclical and sensitive to interest rates "
        "and inflation across our major markets.",
    ),
    ("heading", "We face risks from our former joint venture."),
    (
        "para",
        "Our former joint venture in Brazil could expose us to indemnification claims "
        "related to tax matters that predate our exit.",
    ),
]
NEW_1A = [
    ("heading", "Item 1A. Risk Factors"),
    ("heading", "We depend on a limited number of suppliers."),
    (
        "para",
        "Many components are obtained from single or limited sources, and supply "
        "disruptions could harm our business and results of operations.",
    ),
    ("heading", "Global economic conditions could adversely affect us."),
    (
        "para",
        "Demand for industrial equipment is cyclical and highly sensitive to interest "
        "rates, inflation and tariffs across our major markets.",
    ),
    ("heading", "New export restrictions could limit sales in China."),
    (
        "para",
        "In January the U.S. government announced export restrictions on advanced "
        "industrial controllers that could prevent sales to customers in China.",
    ),
]


def test_align_classifies_added_removed_modified_unchanged() -> None:
    old, new = _section("1A", OLD_1A), _section("1A", NEW_1A)
    paras = align(units(old, ("para",)), units(new, ("para",)))
    assert paras.unchanged == 1
    assert [m[1].text.split()[0] for m in paras.modified] == ["Demand"]
    assert [u.text.split()[0] for u in paras.added] == ["In"]
    assert [u.text.split()[0] for u in paras.removed] == ["Our"]
    heads = align(units(old, ("heading",)), units(new, ("heading",)))
    assert [u.text for u in heads.added] == ["New export restrictions could limit sales in China."]
    assert [u.text for u in heads.removed] == ["We face risks from our former joint venture."]
    assert heads.unchanged == 2


def test_align_is_deterministic_with_embeddings() -> None:
    old, new = _section("1A", OLD_1A), _section("1A", NEW_1A)
    runs = [align(units(old, ("para",)), units(new, ("para",)), HashingEmbedder()) for _ in "ab"]
    assert runs[0] == runs[1]


def _write_sections(
    processed: Path, cik: str, ticker: str, acc: str, fy: int, section: Section
) -> None:
    info = FilingInfo(acc, cik, ticker, "10-K", fy, "FY")
    path = processed / "sections" / cik / f"{acc}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    sections_frame([section], info).to_parquet(path)


def test_compare_sections_tool_defaults_to_latest_two_years(tmp_path: Path) -> None:
    cik = "0000999001"
    _write_sections(tmp_path, cik, "ACME", "0000999001-22-000001", 2022, _section("1A", OLD_1A))
    _write_sections(tmp_path, cik, "ACME", "0000999001-23-000001", 2023, _section("1A", NEW_1A))
    ctx = make_ctx(tmp_path, SectionStore(tmp_path))
    out, is_error = run_tool(ctx, "compare_sections", {"ticker": "ACME", "item": "1A"})
    assert not is_error, out
    assert out["filing_a"]["fiscal_year"] == 2022 and out["filing_b"]["fiscal_year"] == 2023
    assert [h["text"] for h in out["added_risk_headings"]] == [
        "New export restrictions could limit sales in China."
    ]
    assert out["counts"]["paragraphs_modified"] == 1
    sims = out["similarity"]
    assert set(sims) == {"cosine_tf", "jaccard", "min_edit", "simple", "embedding_cosine"}
    assert all(0 < v < 1 for v in sims.values())
    ref = out["added_risk_headings"][0]["ref"]
    src = ctx.registry.get(ref)
    assert src is not None and src.accession_no == "0000999001-23-000001" and src.kind == "section"

    err, is_error = run_tool(
        ctx, "compare_sections", {"ticker": "ACME", "item": "1A", "year_a": 2019, "year_b": 2023}
    )
    assert is_error and "available years: [2022, 2023]" in err["error"]


def test_compare_uses_stored_section_vectors_and_never_reembeds(tmp_path: Path) -> None:
    """With a database hook, embedding similarity comes from stored vectors (fast)."""
    cik = "0000999001"
    _write_sections(tmp_path, cik, "ACME", "0000999001-22-000001", 2022, _section("1A", OLD_1A))
    _write_sections(tmp_path, cik, "ACME", "0000999001-23-000001", 2023, _section("1A", NEW_1A))
    ctx = make_ctx(tmp_path, SectionStore(tmp_path))
    vectors = {
        "0000999001-22-000001": np.array([1.0, 0.0]),
        "0000999001-23-000001": np.array([0.6, 0.8]),
    }
    ctx.section_vector = lambda acc, item: vectors.get(acc)
    out, is_error = run_tool(ctx, "compare_sections", {"ticker": "ACME", "item": "1A"})
    assert not is_error, out
    assert out["similarity"]["embedding_cosine"] == pytest.approx(0.6)
    assert out["counts"]["paragraphs_modified"] == 1  # fuzzy paragraph alignment still finds it

    # a filing that isn't embedded yet: the score is omitted, nothing is embedded per request
    class Boom(HashingEmbedder):
        def embed_documents(self, texts):  # type: ignore[no-untyped-def,override]
            assert len(texts) < 10, "sections must not be re-embedded inside the API"
            return super().embed_documents(texts)

    ctx = make_ctx(tmp_path, SectionStore(tmp_path))
    ctx.embedder = Boom()
    ctx.section_vector = lambda acc, item: None
    out, _ = run_tool(ctx, "compare_sections", {"ticker": "ACME", "item": "1A"})
    assert out["similarity"]["embedding_cosine"] is None


# --- similarity ---------------------------------------------------------------------------


def test_similarity_measures_bounds_and_known_values() -> None:
    a, b = ["the", "cat", "sat"], ["the", "cat", "ran"]
    assert jaccard(a, b) == pytest.approx(2 / 4)
    assert cosine_tf(a, b) == pytest.approx(2 / 3)
    assert min_edit(a, b) == pytest.approx(1 - 2 / 6)  # a substitution = delete + insert
    # the paper's example: D_A -> D_B adds one word; D_A -> D_C deletes 3 and adds 3
    da = ["we", "expect", "demand", "to", "increase"]
    assert min_edit(da, ["we", "expect", "worldwide", "demand", "to", "increase"]) == pytest.approx(
        1 - 1 / 11
    )
    assert min_edit(da, ["we", "expect", "weakness", "in", "sales"]) == pytest.approx(1 - 6 / 10)
    # simple works on lines: every word in a changed line counts as changed
    old = [["risk", "one"], ["risk", "two", "here"]]
    new = [["risk", "one"], ["risk", "two", "there"]]
    assert change_ratio(old, new) == pytest.approx(6 / 10)
    assert simple(old, new) == pytest.approx(1 - 6 / 10)
    scaled = feature_scale_simple(np.array([0.0, 0.3, 0.6]), c_max=0.6)
    assert scaled.tolist() == pytest.approx([1.0, 0.5, 0.0])
    same = all_measures("Risk factors apply.", "Risk factors apply.")
    assert set(same.values()) == {1.0}
    assert all_measures("alpha beta", "gamma delta") == {
        "cosine_tf": 0.0,
        "jaccard": 0.0,
        "min_edit": 0.0,
        "simple": 0.0,
    }


# --- numbers guard ------------------------------------------------------------------------


def test_unsupported_numbers() -> None:
    tool = '{"values_by_year": {"2023": "0.441311"}, "value": "383285000000.0"}'
    chunk = "Net sales decreased 3% or $11.0 billion during 2023."
    ok = (
        "Apple's gross margin was 44.1% in FY2023 [1], on revenue of $383.3 billion [2]. "
        "Net sales decreased 3% [3]. Item 7 of the 10-K covers 2 segments."
    )
    assert unsupported_numbers(ok, [tool, chunk]) == []
    bad = "Gross margin rose to 46.2%, about $12.5 billion more than in 2022."
    assert unsupported_numbers(bad, [tool, chunk]) == ["46.2%", "$12.5 billion"]


def test_sections_frame_roundtrip_still_works(tmp_path: Path) -> None:
    sec = _section("1A", OLD_1A)
    df = sections_frame([sec], FilingInfo("a", "1", "T", "10-K", 2022, "FY"))
    assert isinstance(df, pd.DataFrame) and df.iloc[0]["item"] == "1A"
