"""Year-over-year change signals per filing (specs/07, step 1).

Each original 10-K is compared with the same firm's previous 10-K, and each 10-Q with the
same fiscal quarter's 10-Q a year earlier (Cohen, Malloy & Nguyen 2020, p. 16). Amendments
are excluded. Every row carries the *new* filing's acceptance timestamp and the calendar
month it was released in (US Eastern, which is EDGAR's clock), so portfolio formation can
enter the month after. A filing accepted on March 31 at 9 pm Eastern is a March release.

Output: one row per (filing, section). Sections are ``full`` (all extracted Items, in
document order, which approximates the paper's "main text"), plus Risk Factors, Legal
Proceedings and MD&A.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from filinglens.agent.diff import align, units
from filinglens.parsing.items import base_form
from filinglens.parsing.pipeline import load_filings_index, sections_from_frame
from filinglens.parsing.sections import Section
from filinglens.research.similarity import (
    change_ratio,
    cosine_tf,
    embedding_cosine,
    jaccard,
    min_edit,
    tokenize,
    tokenize_lines,
)
from filinglens.research.text_features import LMDictionary, officer_change_flags
from filinglens.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)

# Bump when a signal definition changes, so cached rows are recomputed.
SIGNALS_VERSION = "1"
EASTERN = "America/New_York"

SECTIONS: dict[str, dict[str, str | None]] = {
    "10-K": {"full": None, "1A": "1A", "3": "3", "7": "7"},
    "10-Q": {"full": None, "1A": "II-1A", "3": "II-1", "7": "I-2"},
}
RISK_SECTION = "1A"

# A "previous year" filing ends 300-430 days before the new one (fiscal year-end changes
# and 52/53-week years move period ends by a few weeks).
MIN_GAP_DAYS, MAX_GAP_DAYS = 300, 430

SIGNAL_COLUMNS = [
    "sim_cosine_tf",
    "sim_jaccard",
    "sim_minedit",
    "sim_simple",
    "sim_embed",
    "sim_embed_aligned",
    "n_new_risks",
    "n_removed_risks",
    "lm_neg_delta",
]


@dataclass
class SignalConfig:
    """What to compute. Off by default: embeddings and the LM dictionary need extra inputs."""

    min_words: int = 50
    with_min_edit: bool = True
    embedder: Embedder | None = None
    lm: LMDictionary | None = None
    max_embed_paragraphs: int = 400
    sections: dict[str, dict[str, str | None]] = field(default_factory=lambda: SECTIONS)


def eligible_filings(index: pd.DataFrame, forms: Sequence[str] = ("10-K",)) -> pd.DataFrame:
    """Original filings of the requested base forms with an acceptance timestamp."""
    if index.empty:
        return index
    df = index.copy()
    df["base_form"] = df["form"].map(base_form)
    is_amend = df["form"].str.endswith("/A")
    if "is_amendment" in df:
        is_amend |= df["is_amendment"].fillna(False).astype(bool)
    df = df[~is_amend & df["base_form"].isin(set(forms)) & df["acceptance_datetime"].notna()]
    df["acceptance_datetime"] = pd.to_datetime(df["acceptance_datetime"], utc=True)
    period = pd.to_datetime(df["period_of_report"], errors="coerce")
    df["period_end"] = period.fillna(pd.to_datetime(df["filing_date"]))
    df = df.sort_values(["cik", "base_form", "period_end", "acceptance_datetime"])
    # several originals for one period (rare re-files): keep the first one released
    return df.drop_duplicates(["cik", "base_form", "period_end"], keep="first")


def pair_filings(index: pd.DataFrame, forms: Sequence[str] = ("10-K",)) -> pd.DataFrame:
    """Each eligible filing with its year-earlier counterpart (closest to 365 days back)."""
    df = eligible_filings(index, forms)
    rows = []
    for _, grp in df.groupby(["cik", "base_form"], sort=False):
        ends = grp["period_end"].tolist()
        recs = grp.to_dict("records")
        for i, new in enumerate(recs):
            best, best_err = None, None
            for j in range(i - 1, -1, -1):
                gap = (ends[i] - ends[j]).days
                if gap > MAX_GAP_DAYS:
                    break
                if gap >= MIN_GAP_DAYS and (best_err is None or abs(gap - 365) < best_err):
                    best, best_err = recs[j], abs(gap - 365)
            if best is None:
                continue
            rows.append(
                {
                    "cik": new["cik"],
                    "ticker": new.get("ticker"),
                    "form": new["base_form"],
                    "accession_no": new["accession_no"],
                    "prev_accession_no": best["accession_no"],
                    "period_end": new["period_end"],
                    "prev_period_end": best["period_end"],
                    "acceptance_datetime": new["acceptance_datetime"],
                    "prev_acceptance_datetime": best["acceptance_datetime"],
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["filing_month"] = release_month(out["acceptance_datetime"])
    return out


def release_month(acceptance_utc: pd.Series) -> pd.Series:
    """Calendar month of release on EDGAR's (US Eastern) clock, as a month Period."""
    local = pd.to_datetime(acceptance_utc, utc=True).dt.tz_convert(EASTERN)
    return local.dt.tz_localize(None).dt.to_period("M")


# --- per-filing document cache ----------------------------------------------------------


@dataclass
class _Doc:
    """Tokenized sections of one filing, computed once and reused across its two pairs."""

    sections: dict[str, Section]
    tokens: dict[str, list[str]] = field(default_factory=dict)
    lines: dict[str, list[list[str]]] = field(default_factory=dict)
    embeds: dict[str, np.ndarray] = field(default_factory=dict)


def load_sections(processed_dir: Path, cik: str, accession_no: str) -> dict[str, Section] | None:
    """Parsed sections of one filing keyed by item, in document order (or ``None``)."""
    path = processed_dir / "sections" / str(cik) / f"{accession_no}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    return {s.item: s for s in sections_from_frame(frame)}


def _section_text(doc: _Doc, item: str | None) -> Section | None:
    if item is not None:
        return doc.sections.get(item)
    if not doc.sections:
        return None
    text = "\n\n".join(s.text for s in doc.sections.values())
    return Section(item="full", title="Full document", text=text, spans=(), source="joined")


def _tokens(doc: _Doc, key: str, sec: Section) -> tuple[list[str], list[list[str]]]:
    if key not in doc.tokens:
        doc.tokens[key] = tokenize(sec.text)
        doc.lines[key] = tokenize_lines(sec.text)
    return doc.tokens[key], doc.lines[key]


def _paragraph_embeddings(doc: _Doc, key: str, sec: Section, cfg: SignalConfig) -> np.ndarray:
    if key not in doc.embeds:
        assert cfg.embedder is not None
        if key == "full":
            paras = [u.text for s in doc.sections.values() for u in units(s, ("para",))]
        else:
            paras = [u.text for u in units(sec, ("para",))]
        paras = paras[: cfg.max_embed_paragraphs] or [sec.text[:4000]]
        doc.embeds[key] = cfg.embedder.embed_documents(paras)
    return doc.embeds[key]


def section_signals(
    old: _Doc, new: _Doc, key: str, item: str | None, cfg: SignalConfig
) -> dict[str, Any] | None:
    """All signals for one section of one filing pair, or ``None`` if either side is missing."""
    so, sn = _section_text(old, item), _section_text(new, item)
    if so is None or sn is None:
        return None
    to, lo = _tokens(old, key, so)
    tn, ln = _tokens(new, key, sn)
    row: dict[str, Any] = {"section": key, "words_old": len(to), "words_new": len(tn)}
    nan = math.nan
    if min(len(to), len(tn)) < cfg.min_words:
        row.update({c: nan for c in SIGNAL_COLUMNS})
        row["change_simple"] = nan
        row["too_short"] = True
        return row
    c = change_ratio(lo, ln)
    row.update(
        too_short=False,
        sim_cosine_tf=cosine_tf(to, tn),
        sim_jaccard=jaccard(to, tn),
        sim_minedit=min_edit(to, tn) if cfg.with_min_edit else nan,
        change_simple=c,
        sim_simple=1.0 - c,
        sim_embed=nan,
        sim_embed_aligned=nan,
        n_new_risks=nan,
        n_removed_risks=nan,
        lm_neg_old=nan,
        lm_neg_new=nan,
        lm_neg_delta=nan,
    )
    if cfg.embedder is not None:
        eo = _paragraph_embeddings(old, key, so, cfg)
        en = _paragraph_embeddings(new, key, sn, cfg)
        row["sim_embed"] = embedding_cosine(eo.mean(axis=0), en.mean(axis=0))
        # paragraph-aligned: each new paragraph's best match in the old filing, averaged
        row["sim_embed_aligned"] = float(np.clip(en @ eo.T, 0, 1).max(axis=1).mean())
    if key == RISK_SECTION and item is not None:
        diff = align(units(so, ("heading",)), units(sn, ("heading",)))
        row["n_new_risks"], row["n_removed_risks"] = len(diff.added), len(diff.removed)
    if cfg.lm is not None:
        row["lm_neg_old"] = cfg.lm.share(to)
        row["lm_neg_new"] = cfg.lm.share(tn)
        row["lm_neg_delta"] = row["lm_neg_new"] - row["lm_neg_old"]
    if key == "full":
        row.update(officer_change_flags(tn))
    return row


def iter_signals(
    processed_dir: Path, pairs: pd.DataFrame, cfg: SignalConfig, skip: set[tuple[str, str]]
) -> Iterator[dict[str, Any]]:
    """Signal rows for ``pairs``; ``skip`` holds (accession_no, section) already computed."""
    cache: dict[str, _Doc | None] = {}
    records = pairs.sort_values(["cik", "form", "period_end"]).to_dict("records")
    for rec in records:
        pair = {str(k): v for k, v in rec.items()}
        form = str(pair["form"])
        wanted = {
            k: v for k, v in cfg.sections[form].items() if (pair["accession_no"], k) not in skip
        }
        if not wanted:
            continue
        # keep only the two documents this firm's next pair can reuse
        for acc in [a for a in cache if a not in (pair["accession_no"], pair["prev_accession_no"])]:
            del cache[acc]
        docs = []
        for acc in (pair["prev_accession_no"], pair["accession_no"]):
            if acc not in cache:
                secs = load_sections(processed_dir, str(pair["cik"]), str(acc))
                cache[acc] = _Doc(secs) if secs else None
            docs.append(cache[acc])
        old, new = docs
        if old is None or new is None:
            logger.info("skip %s: sections not parsed", pair["accession_no"])
            continue
        for key, item in wanted.items():
            row = section_signals(old, new, key, item, cfg)
            if row is not None:
                yield {**pair, **row, "signals_version": SIGNALS_VERSION}


def run_signals(
    processed_dir: Path,
    out_path: Path,
    cfg: SignalConfig,
    *,
    forms: Sequence[str] = ("10-K",),
    tickers: Sequence[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Compute (incrementally) and save the signals table; returns the full table."""
    index = load_filings_index(processed_dir)
    if index.empty:
        raise FileNotFoundError(
            f"No filings index under {processed_dir}/filings; run `filinglens ingest` first."
        )
    if tickers:
        index = index[index["ticker"].isin({t.upper() for t in tickers})]
    pairs = pair_filings(index, forms)
    existing = pd.DataFrame()
    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path)
        existing = existing[existing["signals_version"] == SIGNALS_VERSION]
    skip = set(zip(existing.get("accession_no", []), existing.get("section", []), strict=False))
    new_rows = pd.DataFrame(list(iter_signals(processed_dir, pairs, cfg, skip)))
    table = pd.concat([existing, new_rows], ignore_index=True) if not new_rows.empty else existing
    if table.empty:
        return table
    table["filing_month"] = pd.PeriodIndex(table["filing_month"].astype(str), freq="M")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    table.assign(filing_month=table["filing_month"].astype(str)).to_parquet(tmp, index=False)
    tmp.replace(out_path)
    logger.info("signals: %d new rows, %d total", len(new_rows), len(table))
    return table


def load_signals(path: Path) -> pd.DataFrame:
    """Read a signals table written by :func:`run_signals`."""
    df = pd.read_parquet(path)
    df["filing_month"] = pd.PeriodIndex(df["filing_month"], freq="M")
    df["acceptance_datetime"] = pd.to_datetime(df["acceptance_datetime"], utc=True)
    return df
