from datetime import date
from pathlib import Path

import pytest

from filinglens.parsing import sections as sections_mod
from filinglens.parsing.blocks import Block, html_to_blocks
from filinglens.parsing.clean import remove_page_furniture
from filinglens.parsing.metadata import fiscal_from_ixbrl, resolve_fiscal
from filinglens.parsing.sections import extract_sections, markdown_to_blocks

FIXTURES = Path(__file__).parent / "fixtures" / "filings"


def load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


# --- HTML -> blocks ---------------------------------------------------------------------


def test_hidden_ixbrl_header_and_toc_table_are_dropped() -> None:
    texts = [b.text for b in html_to_blocks(load("tenk_ixbrl.htm"))]
    assert not any("0000999001" in t for t in texts)  # dei:EntityCentralIndexKey (hidden)
    assert not any(t.startswith("| Item 1A") or "Risk Factors | 5" in t for t in texts)


def test_bold_detection_respects_font_weight_override() -> None:
    blocks = html_to_blocks(
        b"<div><b>Bold heading</b></div>"
        b"<div><b>Mostly <span style='font-weight:normal'>not bold at all here</span></b></div>"
        b"<div><span style='font-weight:700;'>Styled heading</span></div>"
        b"<div style='display:none'>hidden</div>"
    )
    assert [(b.kind, b.text) for b in blocks] == [
        ("heading", "Bold heading"),
        ("para", "Mostly not bold at all here"),
        ("heading", "Styled heading"),
    ]


def test_legacy_cp1252_punctuation_is_normalized() -> None:
    texts = [b.text for b in html_to_blocks(load("tenk_legacy.htm"))]
    assert "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION" in " ".join(texts)
    assert not any("\u0092" in t or "\u00a0" in t for t in texts)


def test_financial_table_cells_are_merged() -> None:
    tables = [b.text for b in html_to_blocks(load("tenk_ixbrl.htm")) if b.kind == "table"]
    assert len(tables) == 1
    assert "| Net sales | $3,105 | $3,200 | (3)% |" in tables[0]


def test_single_row_layout_table_becomes_heading() -> None:
    blocks = html_to_blocks(
        b"<table><tr><td><b>Item 7.</b></td><td><b>MD&amp;A</b></td></tr></table>"
    )
    assert blocks == [Block("heading", "Item 7. MD&A")]


# --- page furniture -----------------------------------------------------------------------


def test_page_furniture_removed_but_real_repeats_kept() -> None:
    blocks = [
        Block("para", "Acme | 2023 Form 10-K | 3"),
        Block("heading", "Fiscal 2023 Compared to Fiscal 2022"),
        Block("para", "None."),
        Block("para", "Acme | 2023 Form 10-K | 4"),
        Block("para", "- 12 -"),
        Block("heading", "Fiscal 2023 Compared to Fiscal 2022"),
        Block("para", "Table of Contents"),
        Block("para", "None."),
        Block("para", "Acme | 2023 Form 10-K | 5"),
        Block("heading", "Fiscal 2023 Compared to Fiscal 2022"),
        Block("para", "None."),
        Block("para", "F-3"),
    ]
    assert [b.text for b in remove_page_furniture(blocks)] == [
        "Fiscal 2023 Compared to Fiscal 2022",
        "None.",
    ] * 3


# --- section boundaries on fixtures ---------------------------------------------------------

EXPECTED = {
    "tenk_ixbrl.htm": (
        "10-K",
        ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8"],
        {
            "1": ("Item 1. Business", "is a key competitive advantage."),
            "1A": ("Item 1A. Risk Factors", "fines, penalties or reputational harm."),
            "4": ("Item 4. Mine Safety Disclosures", "Not applicable."),
            "7": ("Item 7. Management's Discussion", "over the next 12 months."),
            "7A": ("Item 7A. Quantitative", "$41 million in 2023."),
        },
    ),
    "tenk_legacy.htm": (
        "10-K",
        ["1", "1A", "3", "7", "7A", "8"],
        {
            "1A": ("ITEM 1A. RISK FACTORS", "significant liabilities and regulatory penalties."),
            "3": ("ITEM 3. LEGAL PROCEEDINGS", "vigorously defend this action."),
            "7": ("ITEM 7. MANAGEMENT'S DISCUSSION", "lower service costs."),
        },
    ),
    "tenq.htm": (
        "10-Q",
        ["I-1", "I-2", "I-3", "I-4", "II-1", "II-1A", "II-2", "II-6"],
        {
            "I-2": ("Item 2. Management's Discussion", "industrial demand in Germany."),
            "I-4": ("Item 4. Controls and Procedures", "as of March 30, 2024."),
            "II-1A": ("Item 1A. Risk Factors", "approximately 7% of net sales in 2023."),
        },
    ),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_item_boundaries(name: str) -> None:
    form, keys, bounds = EXPECTED[name]
    result = extract_sections(load(name), form)
    assert list(result.sections) == keys
    assert all(s.source == "native" for s in result.sections.values())
    for key, (first, last) in bounds.items():
        text = result.sections[key].text
        assert text.startswith(first), key
        assert text.endswith(last), key


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_no_toc_or_page_junk_in_sections(name: str) -> None:
    form, _, _ = EXPECTED[name]
    for section in extract_sections(load(name), form).sections.values():
        for line in section.text.split("\n\n"):
            assert "Table of Contents" not in line
            assert "Form 10-K |" not in line and "Form 10-Q |" not in line
            assert not line.strip().isdigit()
            assert not line.upper().startswith("PART I")


def test_cross_reference_does_not_split_item() -> None:
    item1 = extract_sections(load("tenk_ixbrl.htm"), "10-K").sections["1"].text
    assert "see Part I, Item 1A of this Form 10-K" in item1
    assert "Human Capital" in item1


def test_risk_factor_headings_are_preserved_as_heading_blocks() -> None:
    section = extract_sections(load("tenk_legacy.htm"), "10-K").sections["1A"]
    headings = [section.text[s.start : s.end] for s in section.spans if s.kind == "heading"]
    assert headings == [
        "ITEM 1A. RISK FACTORS",
        "Oil and natural gas prices are volatile, and a sustained decline in prices could "
        "adversely affect our financial condition.",
        "Our hedging activities may limit the benefit of price increases and expose us to "
        "counterparty risk.",
        "Offshore operations involve special risks that could affect our operations adversely.",
    ]


def test_spans_index_section_text_exactly() -> None:
    for section in extract_sections(load("tenk_ixbrl.htm"), "10-K").sections.values():
        rebuilt = "\n\n".join(section.text[s.start : s.end] for s in section.spans)
        assert rebuilt == section.text


def test_edgartools_fills_required_items_native_missed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sections_mod, "split_items", lambda blocks, form: {})
    result = extract_sections(load("tenk_ixbrl.htm"), "10-K")
    assert list(result.sections) == ["1A", "7"]  # only required items are filled
    assert all(s.source == "edgartools" for s in result.sections.values())
    assert "Acme Devices, Inc. | 2023" not in result.sections["1A"].text  # footer removed
    assert result.edgartools_chars["1A"] > 1000


def test_edgartools_cross_check_lengths_agree_on_fixture() -> None:
    result = extract_sections(load("tenk_ixbrl.htm"), "10-K")
    for key in ("1A", "7", "7A"):
        ratio = len(result.sections[key].text) / result.edgartools_chars[key]
        assert 0.8 <= ratio <= 1.25, key


def test_markdown_to_blocks_unescapes() -> None:
    blocks = markdown_to_blocks(
        "# Item 1A\\. Risk Factors\n\n**A heading\\.**\n\nU\\.S\\. text \\- with **bold**\\.\n\n"
        "| a | b |\n| --- | --- |"
    )
    assert blocks == [
        Block("heading", "Item 1A. Risk Factors"),
        Block("heading", "A heading."),
        Block("para", "U.S. text - with bold."),
        Block("table", "| a | b |\n| --- | --- |"),
    ]


# --- fiscal metadata ----------------------------------------------------------------------


def test_fiscal_year_from_ixbrl() -> None:
    assert fiscal_from_ixbrl(load("tenq.htm").decode()) == (2024, "Q2")
    assert fiscal_from_ixbrl(load("tenk_ixbrl.htm").decode()) == (2023, "FY")


def test_fiscal_fallbacks_for_legacy_filings() -> None:
    html = load("tenk_legacy.htm").decode("cp1252")
    assert resolve_fiscal(html, "10-K", date(2015, 12, 31)) == (2015, "FY")
    assert resolve_fiscal(html, "10-K", date(2015, 12, 31), (2016, "FY")) == (2016, "FY")
    assert resolve_fiscal("", "10-Q", None) == (None, None)
