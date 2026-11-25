"""Parse ingested filings into section and chunk parquet files, plus a quality report."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from filinglens.parsing.blocks import decode_html
from filinglens.parsing.chunker import (
    DEFAULT_CHUNK_ITEMS,
    Chunk,
    ChunkConfig,
    FilingInfo,
    chunk_section,
)
from filinglens.parsing.items import MIN_CHARS, QUALITY_ITEMS, base_form, canonical_items
from filinglens.parsing.metadata import resolve_fiscal
from filinglens.parsing.sections import Section, Span, extract_sections

logger = logging.getLogger(__name__)

# Bump when parsing output changes, so cached sections are regenerated.
PARSER_VERSION = "1"

SECTION_COLUMNS = [
    "accession_no",
    "cik",
    "ticker",
    "form",
    "fiscal_year",
    "fiscal_period",
    "item",
    "title",
    "text",
    "char_length",
    "source",
    "parser_version",
    "block_kinds",
    "block_starts",
    "block_ends",
]
CHUNK_COLUMNS = [
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
    "context_header",
]


@dataclass
class FilingParseResult:
    """Outcome for one filing; ``quality`` feeds the coverage report."""

    accession_no: str
    cik: str
    ticker: str
    form: str
    is_amendment: bool
    fiscal_year: int | None = None
    n_sections: int = 0
    n_chunks: int = 0
    cached: bool = False
    error: str | None = None
    quality: dict[str, Any] = field(default_factory=dict)


def sections_path(processed_dir: Path, cik: str, accession_no: str) -> Path:
    """``processed/sections/<cik>/<accession>.parquet`` (one row per Item; used by research)."""
    return processed_dir / "sections" / cik / f"{accession_no}.parquet"


def chunks_path(processed_dir: Path, cik: str, accession_no: str) -> Path:
    """``processed/chunks/<cik>/<accession>.parquet``."""
    return processed_dir / "chunks" / cik / f"{accession_no}.parquet"


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_filings_index(processed_dir: Path) -> pd.DataFrame:
    """All filing metadata written by ingestion (``processed/filings/*.parquet``)."""
    files = sorted((processed_dir / "filings").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _facts_hint(processed_dir: Path, cik: str, accession_no: str) -> tuple[int | None, str | None]:
    """Fiscal year/period that companyfacts attributes to this accession (pre-iXBRL fallback)."""
    path = processed_dir / "facts" / f"{cik}.parquet"
    if not path.exists():
        return None, None
    facts = pd.read_parquet(path, columns=["accession_no", "fy", "fp"])
    rows = facts[facts["accession_no"] == accession_no]
    if rows.empty:
        return None, None
    fy = rows["fy"].dropna().mode()
    fp = rows["fp"].dropna().mode()
    return (int(fy.iloc[0]) if not fy.empty else None), (str(fp.iloc[0]) if not fp.empty else None)


def sections_frame(sections: Iterable[Section], info: FilingInfo) -> pd.DataFrame:
    """Sections as rows, including block spans so chunking can rerun without re-parsing."""
    rows = [
        {
            "accession_no": info.accession_no,
            "cik": info.cik,
            "ticker": info.ticker,
            "form": info.form,
            "fiscal_year": info.fiscal_year,
            "fiscal_period": info.fiscal_period,
            "item": s.item,
            "title": s.title,
            "text": s.text,
            "char_length": len(s.text),
            "source": s.source,
            "parser_version": PARSER_VERSION,
            "block_kinds": [sp.kind for sp in s.spans],
            "block_starts": [sp.start for sp in s.spans],
            "block_ends": [sp.end for sp in s.spans],
        }
        for s in sections
    ]
    df = pd.DataFrame(rows, columns=SECTION_COLUMNS)
    df["fiscal_year"] = df["fiscal_year"].astype("Int64")
    return df


def sections_from_frame(df: pd.DataFrame) -> list[Section]:
    """Rebuild :class:`Section` objects from a sections parquet."""
    sections = []
    for row in df.to_dict("records"):
        spans = zip(row["block_kinds"], row["block_starts"], row["block_ends"], strict=True)
        sections.append(
            Section(
                item=str(row["item"]),
                title=str(row["title"]),
                text=str(row["text"]),
                spans=tuple(Span(kind, int(a), int(b)) for kind, a, b in spans),
                source=str(row["source"]),
            )
        )
    return sections


def chunks_frame(chunks: Sequence[Chunk]) -> pd.DataFrame:
    """Chunks as rows with the ``chunks`` table's columns (minus embedding fields)."""
    df = pd.DataFrame([asdict(c) for c in chunks], columns=CHUNK_COLUMNS)
    df["fiscal_year"] = df["fiscal_year"].astype("Int64")
    return df


def _quality(
    form: str,
    sections: dict[str, Section],
    et_chars: dict[str, int],
    et_error: str | None,
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key in QUALITY_ITEMS[base_form(form)]:
        section = sections.get(key)
        chars = len(section.text) if section else 0
        et = et_chars.get(key)
        items[key] = {
            "found": section is not None,
            "chars": chars,
            "source": section.source if section else None,
            "short": section is not None and chars < MIN_CHARS.get(key, 0),
            "edgartools_chars": et,
            "agrees": None if not (section and et) else 0.8 <= chars / et <= 1.25,
        }
    return {"items": items, "edgartools_error": et_error}


def parse_filing(
    row: dict[str, Any],
    processed_dir: Path,
    cfg: ChunkConfig,
    *,
    force: bool = False,
    use_edgartools: bool = True,
    chunk_items: dict[str, tuple[str, ...]] = DEFAULT_CHUNK_ITEMS,
) -> FilingParseResult:
    """Parse one filing (or reuse its cached sections) and (re)write its chunks.

    Sections are cached per ``PARSER_VERSION``; chunks are always regenerated from them,
    which is cheap and lets evals sweep chunk sizes without re-parsing HTML.
    """
    acc, cik, form = str(row["accession_no"]), str(row["cik"]), str(row["form"])
    result = FilingParseResult(
        accession_no=acc,
        cik=cik,
        ticker=str(row["ticker"]),
        form=form,
        is_amendment=bool(row.get("is_amendment", False)),
    )
    s_path = sections_path(processed_dir, cik, acc)
    q_path = s_path.with_suffix(".quality.json")
    try:
        cached = None
        if s_path.exists() and q_path.exists() and not force:
            cached = pd.read_parquet(s_path)
            quality = json.loads(q_path.read_text())
            if quality.get("parser_version") != PARSER_VERSION:
                cached = None
        if cached is not None:
            sections = sections_from_frame(cached)
            fy, fp = quality["fiscal_year"], quality["fiscal_period"]
            result.cached = True
        else:
            local = row.get("local_path")
            if not local or not Path(local).exists():
                raise FileNotFoundError(f"primary document not on disk: {local}")
            raw = Path(local).read_bytes()
            fy, fp = _fiscal_for(row, processed_dir, decode_html(raw))
            extracted = extract_sections(raw, form, use_edgartools=use_edgartools)
            sections = list(extracted.sections.values())
            quality = _quality(
                form, extracted.sections, extracted.edgartools_chars, extracted.edgartools_error
            )
            quality.update(parser_version=PARSER_VERSION, fiscal_year=fy, fiscal_period=fp)
            info = FilingInfo(acc, cik, result.ticker, form, fy, fp)
            _write_parquet(sections_frame(sections, info), s_path)
            q_path.write_text(json.dumps(quality, sort_keys=True))

        info = FilingInfo(acc, cik, result.ticker, form, fy, fp)
        wanted = set(chunk_items.get(base_form(form), ()))
        chunks = [c for s in sections if s.item in wanted for c in chunk_section(s, info, cfg)]
        _write_parquet(chunks_frame(chunks), chunks_path(processed_dir, cik, acc))
        result.fiscal_year = fy
        result.n_sections = len(sections)
        result.n_chunks = len(chunks)
        result.quality = quality
    except Exception as exc:  # one bad filing must not stop the run
        result.error = f"{type(exc).__name__}: {exc}"
        logger.error("%s %s: %s", result.ticker, acc, result.error)
    return result


def _fiscal_for(
    row: dict[str, Any], processed_dir: Path, html: str
) -> tuple[int | None, str | None]:
    period = row.get("period_of_report")
    if isinstance(period, pd.Timestamp):
        period = period.date()
    hint = _facts_hint(processed_dir, str(row["cik"]), str(row["accession_no"]))
    return resolve_fiscal(
        html, str(row["form"]), period if isinstance(period, date) else None, hint
    )


def run_parse(
    processed_dir: Path,
    cfg: ChunkConfig,
    *,
    tickers: Sequence[str] | None = None,
    forms: Sequence[str] | None = None,
    force: bool = False,
    use_edgartools: bool = True,
    workers: int = 1,
) -> list[FilingParseResult]:
    """Parse every ingested filing that has a local primary document."""
    index = load_filings_index(processed_dir)
    if index.empty:
        return []
    index = index[index["local_path"].notna()]
    if tickers:
        index = index[index["ticker"].isin({t.upper() for t in tickers})]
    if forms:
        wanted = {f.upper() for f in forms}
        index = index[index["form"].map(lambda f: base_form(f) in wanted or f in wanted)]
    ordered = index.sort_values(["cik", "filing_date", "accession_no"])
    rows: list[dict[str, Any]] = [
        {str(k): v for k, v in r.items()} for r in ordered.to_dict("records")
    ]

    if workers <= 1:
        return [
            parse_filing(r, processed_dir, cfg, force=force, use_edgartools=use_edgartools)
            for r in rows
        ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                parse_filing, r, processed_dir, cfg, force=force, use_edgartools=use_edgartools
            )
            for r in rows
        ]
        return [f.result() for f in futures]


def quality_frame(results: Sequence[FilingParseResult]) -> pd.DataFrame:
    """One row per (filing, quality item)."""
    rows = []
    for r in results:
        for item, q in r.quality.get("items", {}).items():
            rows.append(
                {
                    "ticker": r.ticker,
                    "form": r.form,
                    "accession_no": r.accession_no,
                    "fiscal_year": r.fiscal_year,
                    "is_amendment": r.is_amendment,
                    "item": item,
                    **q,
                }
            )
        if r.error:
            rows.append(
                {
                    "ticker": r.ticker,
                    "form": r.form,
                    "accession_no": r.accession_no,
                    "is_amendment": r.is_amendment,
                    "item": None,
                    "found": False,
                    "error": r.error,
                }
            )
    return pd.DataFrame(rows)


def _pct_equal(values: pd.Series, target: object) -> float:
    """Percentage of non-null ``values`` equal to ``target`` (0 when all are null)."""
    known = values.dropna()
    return float((known == target).mean() * 100) if len(known) else 0.0


def coverage_table(quality: pd.DataFrame) -> pd.DataFrame:
    """Per form x item coverage over original (non-amended) filings."""
    if quality.empty:
        return pd.DataFrame()
    q = quality[~quality["is_amendment"].astype(bool) & quality["item"].notna()].copy()
    q["base_form"] = q["form"].map(base_form)
    order = {k: i for form in ("10-K", "10-Q") for i, k in enumerate(canonical_items(form))}
    q["order"] = q["item"].map(order)
    grouped = q.groupby(["base_form", "order", "item"], sort=True)
    table = pd.DataFrame(
        {
            "filings": grouped.size(),
            "found_pct": grouped["found"].mean() * 100,
            "short_pct": grouped["short"].mean() * 100,
            "from_edgartools_pct": grouped["source"].apply(
                lambda s: (s == "edgartools").mean() * 100
            ),
            "disagree_pct": grouped["agrees"].apply(
                lambda s: (s.dropna() == False).mean() * 100 if s.notna().any() else 0.0  # noqa: E712
            ),
        }
    ).reset_index()
    return table.drop(columns="order").round(1)
