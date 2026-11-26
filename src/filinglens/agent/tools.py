"""The agent's tools: Pydantic-validated arguments, deterministic Python implementations.

Every tool returns JSON-serializable data. Text and numbers the model may cite are
registered in the :class:`SourceRegistry` and returned with their handles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError

from filinglens.agent.diff import SectionStore, align, section_similarity, units
from filinglens.agent.financials import FactStore, get_financials, load_concept_map
from filinglens.agent.ratios import RATIOS, compute
from filinglens.agent.sources import SourceRegistry
from filinglens.retrieval.embedder import Embedder
from filinglens.retrieval.filters import CompanyDirectory, QueryFilters
from filinglens.retrieval.retriever import Retriever

MAX_DIFF_ITEMS = 25
MAX_TEXT_CHARS = 700


class ToolError(Exception):
    """An error the model should see (bad ticker, missing year, ...)."""


@dataclass
class AgentContext:
    """Everything tools need for one agent run."""

    retriever: Retriever
    sections: SectionStore
    facts: FactStore
    directory: CompanyDirectory
    registry: SourceRegistry
    embedder: Embedder | None = None
    concept_map_path: str = "config/concept_map.yaml"

    def cik(self, ticker: str) -> str:
        """Resolve a ticker (or a company name) to a CIK, or raise a helpful error."""
        key = ticker.strip().upper().replace(".", "-")
        cik = self.directory.by_ticker.get(key) or self.directory.by_alias.get(ticker.lower())
        if cik is None:
            known = ", ".join(sorted(self.directory.by_ticker))
            raise ToolError(f"Unknown ticker {ticker!r}. Companies available: {known}.")
        return cik


# --- argument models (also the strict JSON schemas the model sees) ------------------------


class SearchFilingsArgs(BaseModel):
    """Arguments for ``search_filings``."""

    query: str = Field(description="What to look for, in plain words.")
    ticker: str | None = Field(default=None, description="Restrict to one company, e.g. AAPL.")
    fiscal_year: int | None = Field(default=None, description="Restrict to one fiscal year.")
    form: Literal["10-K", "10-Q"] | None = None
    item: str | None = Field(
        default=None, description='Item code, e.g. "1A", "7"; 10-Q items like "II-1A".'
    )
    top_k: int = Field(default=8, description="Number of chunks to return (1-15).")


class CompareSectionsArgs(BaseModel):
    """Arguments for ``compare_sections``."""

    ticker: str
    item: str = Field(default="1A", description='10-K item code, e.g. "1A", "7", "3".')
    year_a: int | None = Field(default=None, description="Older fiscal year (default b-1).")
    year_b: int | None = Field(default=None, description="Newer fiscal year (default latest).")


class GetFinancialsArgs(BaseModel):
    """Arguments for ``get_financials``."""

    ticker: str
    concepts: list[str] = Field(description="Canonical concept names, e.g. revenue, net_income.")
    fiscal_years: list[int]
    period: Literal["FY", "Q"] = "FY"
    as_of: Literal["original", "latest"] = Field(
        default="original",
        description='"original" = value as first filed; "latest" = most recent restatement.',
    )


class ComputeRatioArgs(BaseModel):
    """Arguments for ``compute_ratio``."""

    ticker: str
    ratio: str = Field(description="One of: " + ", ".join(sorted(RATIOS)))
    fiscal_years: list[int]
    as_of: Literal["original", "latest"] = "original"


# --- implementations ----------------------------------------------------------------------


def _clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def search_filings(ctx: AgentContext, args: SearchFilingsArgs) -> dict[str, Any]:
    """Hybrid retrieval with explicit filters; returns chunks with citable handles."""
    filters = QueryFilters(
        ciks=(ctx.cik(args.ticker),) if args.ticker else (),
        tickers=(args.ticker.upper(),) if args.ticker else (),
        fiscal_years=(args.fiscal_year,) if args.fiscal_year else (),
        forms=(args.form,) if args.form else (),
        items=(args.item.upper(),) if args.item else (),
    )
    result = ctx.retriever.retrieve(args.query, filters=filters)
    k = max(1, min(15, args.top_k))
    chunks = []
    for c in result.chunks[:k]:
        handle = ctx.registry.add(
            "chunk",
            ("chunk", c.chunk_id),
            text=c.text,
            cik=c.cik,
            ticker=c.ticker,
            accession_no=c.accession_no,
            form=c.form,
            fiscal_year=c.fiscal_year,
            item=c.item,
            chunk_id=c.chunk_id,
            char_start=c.char_start,
            char_end=c.char_end,
        )
        chunks.append(
            {
                "ref": handle,
                "ticker": c.ticker,
                "form": c.form,
                "fiscal_year": c.fiscal_year,
                "item": c.item,
                "heading": c.heading,
                "text": c.text,
            }
        )
    out: dict[str, Any] = {"results": chunks}
    if result.relaxed:
        out["note"] = f"No exact match; dropped filters: {', '.join(result.relaxed)}."
    if not chunks:
        out["note"] = "No matching filing text."
    return out


def compare_sections(ctx: AgentContext, args: CompareSectionsArgs) -> dict[str, Any]:
    """Diff one 10-K item between two fiscal years."""
    cik = ctx.cik(args.ticker)
    item = args.item.upper().removeprefix("ITEM").strip()
    years = ctx.sections.available_years(cik, item)
    if len(years) < 2 and not (args.year_a and args.year_b):
        raise ToolError(f"Need two 10-K filings with Item {item}; available years: {years}.")
    year_b = args.year_b or years[-1]
    year_a = args.year_a or max((y for y in years if y < year_b), default=year_b - 1)
    got_a = ctx.sections.get(cik, item, year_a)
    got_b = ctx.sections.get(cik, item, year_b)
    missing = [y for y, g in ((year_a, got_a), (year_b, got_b)) if g is None]
    if missing or got_a is None or got_b is None:
        raise ToolError(f"No 10-K Item {item} for FY{missing}; available years: {years}.")
    (ref_a, sec_a), (ref_b, sec_b) = got_a, got_b

    paragraphs = align(
        units(sec_a, ("para", "table")), units(sec_b, ("para", "table")), ctx.embedder
    )
    headings = align(units(sec_a, ("heading",)), units(sec_b, ("heading",)), ctx.embedder)

    def cite(ref: Any, unit: Any) -> str:
        return ctx.registry.add(
            "section",
            ("section", ref.accession_no, ref.item, unit.start),
            text=unit.text,
            cik=ref.cik,
            ticker=ref.ticker,
            accession_no=ref.accession_no,
            form=ref.form,
            fiscal_year=ref.fiscal_year,
            item=ref.item,
            char_start=unit.start,
            char_end=unit.end,
        )

    def listing(ref: Any, items: list[Any]) -> list[dict[str, str]]:
        return [{"ref": cite(ref, u), "text": _clip(u.text)} for u in items[:MAX_DIFF_ITEMS]]

    return {
        "ticker": ref_b.ticker,
        "item": item,
        "filing_a": {"fiscal_year": year_a, "accession_no": ref_a.accession_no},
        "filing_b": {"fiscal_year": year_b, "accession_no": ref_b.accession_no},
        "similarity": section_similarity(sec_a, sec_b, ctx.embedder),
        "counts": {
            "paragraphs_added": len(paragraphs.added),
            "paragraphs_removed": len(paragraphs.removed),
            "paragraphs_modified": len(paragraphs.modified),
            "paragraphs_unchanged": paragraphs.unchanged,
            "headings_added": len(headings.added),
            "headings_removed": len(headings.removed),
        },
        "added_risk_headings": listing(ref_b, headings.added),
        "removed_risk_headings": listing(ref_a, headings.removed),
        "added_paragraphs": listing(ref_b, paragraphs.added),
        "removed_paragraphs": listing(ref_a, paragraphs.removed),
        "modified_paragraphs": [
            {
                "old": {"ref": cite(ref_a, o), "text": _clip(o.text)},
                "new": {"ref": cite(ref_b, n), "text": _clip(n.text)},
            }
            for o, n, _ in paragraphs.modified[:MAX_DIFF_ITEMS]
        ],
        "truncated": any(
            len(x) > MAX_DIFF_ITEMS
            for x in (paragraphs.added, paragraphs.removed, paragraphs.modified, headings.added)
        ),
    }


def _financials(ctx: AgentContext, args: GetFinancialsArgs) -> dict[str, Any]:
    from pathlib import Path

    cmap = load_concept_map(Path(ctx.concept_map_path))
    try:
        values, missing = get_financials(
            ctx.facts,
            ctx.cik(args.ticker),
            args.concepts,
            args.fiscal_years,
            period=args.period,
            as_of=args.as_of,
            concept_map=cmap,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "ticker": args.ticker.upper(),
        "as_of": args.as_of,
        "values": [v.to_dict() for v in values],
        "missing": missing,
    }


def _ratio(ctx: AgentContext, args: ComputeRatioArgs) -> dict[str, Any]:
    from pathlib import Path

    if args.ratio not in RATIOS:
        raise ToolError(f"Unknown ratio {args.ratio!r}; choose from {sorted(RATIOS)}.")
    spec = RATIOS[args.ratio]
    years = sorted(set(args.fiscal_years))
    fetch_years = sorted(set(years) | ({y - 1 for y in years} if spec.needs_prior_year else set()))
    cmap = load_concept_map(Path(ctx.concept_map_path))
    values, _ = get_financials(
        ctx.facts,
        ctx.cik(args.ticker),
        list(spec.concepts),
        fetch_years,
        as_of=args.as_of,
        concept_map=cmap,
    )
    by_year: dict[int, dict[str, Any]] = {}
    for v in values:
        by_year.setdefault(v.fiscal_year, {})[v.concept] = v.value
    result = compute(args.ratio, by_year, years)
    return {
        "ticker": args.ticker.upper(),
        "ratio": args.ratio,
        "formula": spec.formula,
        "as_of": args.as_of,
        "values_by_year": {str(y): (None if r is None else str(r)) for y, r in result.items()},
        "inputs_used": [v.to_dict() for v in values],
        "unavailable_years": [y for y, r in result.items() if r is None],
    }


# --- registry -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    """A tool definition plus its implementation."""

    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[AgentContext, Any], dict[str, Any]]
    cites_result: bool = False

    def definition(self) -> dict[str, Any]:
        """The API tool definition (strict schema built from the args model)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": anthropic.transform_schema(self.args_model),
            "strict": True,
        }


TOOLS: tuple[Tool, ...] = (
    Tool(
        "search_filings",
        "Search 10-K/10-Q text (risk factors, MD&A, business, legal proceedings) with "
        "hybrid retrieval. Use it for what a company says: drivers, risks, strategy, "
        "litigation. Filter by ticker, fiscal_year, form and item whenever the question "
        "names them. Each result has a ref (C#) to cite.",
        SearchFilingsArgs,
        search_filings,
    ),
    Tool(
        "compare_sections",
        "Diff one 10-K Item between two fiscal years: similarity scores, and added, removed "
        "and modified paragraphs and risk-factor headings, each with a ref (P#) to cite. "
        "Use it for 'what changed / what's new / what was removed' questions. Omit years "
        "to compare the latest 10-K with the one before.",
        CompareSectionsArgs,
        compare_sections,
        cites_result=True,
    ),
    Tool(
        "get_financials",
        "Reported XBRL values (revenue, net_income, operating_income, gross_profit, "
        "total_assets, stockholders_equity, operating_cash_flow, ...) by fiscal year, with "
        "source accession and filed date. Use it for any reported figure.",
        GetFinancialsArgs,
        _financials,
        cites_result=True,
    ),
    Tool(
        "compute_ratio",
        "Compute a financial ratio (margins, ROE, ROA, current ratio, leverage, YoY growth) "
        "exactly from XBRL inputs. Returns the formula and the inputs used. Use it for any "
        "derived number: never calculate figures yourself.",
        ComputeRatioArgs,
        _ratio,
        cites_result=True,
    ),
)


def run_tool(ctx: AgentContext, name: str, raw_input: Any) -> tuple[dict[str, Any], bool]:
    """Validate and run one tool call. Returns ``(result, is_error)``; never raises."""
    tool = next((t for t in TOOLS if t.name == name), None)
    if tool is None:
        return {"error": f"Unknown tool {name!r}."}, True
    try:
        args = tool.args_model.model_validate(raw_input)
    except ValidationError as exc:
        return {"error": "Invalid arguments", "details": exc.errors(include_url=False)}, True
    try:
        result = tool.fn(ctx, args)
    except ToolError as exc:
        return {"error": str(exc)}, True
    if tool.cites_result:
        result = {"ref": ctx.registry.add_tool_result(name, args.model_dump(), result), **result}
    return result, False
