"""`compare_sections`: what changed in an Item between two fiscal years (specs/04).

Paragraphs are aligned deterministically: exact (normalized) matches first, then a greedy
one-to-one matching on a score combining fuzzy text similarity and, when an embedder is
given, embedding cosine. Matched pairs below the "unchanged" bar are *modified*; the rest
are *added* or *removed*. Risk-factor headings get the same treatment, so "3 new risks"
comes from code, not from the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from filinglens.parsing.pipeline import sections_from_frame
from filinglens.parsing.sections import Section
from filinglens.research.similarity import all_measures, embedding_cosine
from filinglens.retrieval.embedder import Embedder

UNCHANGED_FUZZ = 95.0
MATCH_SCORE = 0.62
MIN_PARAGRAPH_CHARS = 40


@dataclass(frozen=True)
class Unit:
    """A paragraph or heading with its location in the section text."""

    text: str
    start: int
    end: int


@dataclass
class Alignment:
    """Diff of two lists of units."""

    added: list[Unit] = field(default_factory=list)
    removed: list[Unit] = field(default_factory=list)
    modified: list[tuple[Unit, Unit, float]] = field(default_factory=list)
    unchanged: int = 0


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def units(section: Section, kinds: tuple[str, ...]) -> list[Unit]:
    """Blocks of the given kinds (e.g. ``("para", "table")`` or ``("heading",)``).

    The Item's own heading (first block) is skipped; very short paragraphs ("None.") are
    skipped for paragraph diffs.
    """
    out = []
    for i, span in enumerate(section.spans):
        if i == 0 and span.kind == "heading":
            continue
        if span.kind not in kinds:
            continue
        text = section.text[span.start : span.end]
        if span.kind != "heading" and len(text) < MIN_PARAGRAPH_CHARS:
            continue
        out.append(Unit(text, span.start, span.end))
    return out


def align(old: list[Unit], new: list[Unit], embedder: Embedder | None = None) -> Alignment:
    """Align two unit lists; deterministic for identical input."""
    result = Alignment()
    old_left = list(range(len(old)))
    new_left = list(range(len(new)))

    # 1) exact normalized matches, in order
    exact: dict[str, list[int]] = {}
    for i in old_left:
        exact.setdefault(_norm(old[i].text), []).append(i)
    still_new = []
    matched_old: set[int] = set()
    for j in new_left:
        bucket = exact.get(_norm(new[j].text))
        if bucket:
            matched_old.add(bucket.pop(0))
            result.unchanged += 1
        else:
            still_new.append(j)
    old_left = [i for i in old_left if i not in matched_old]
    new_left = still_new

    # 2) greedy matching on fuzzy (+ embedding) score
    if old_left and new_left:
        fuzzy = np.array(
            [
                [fuzz.ratio(_norm(old[i].text), _norm(new[j].text)) / 100 for j in new_left]
                for i in old_left
            ]
        )
        score = fuzzy
        if embedder is not None:
            ea = embedder.embed_documents([old[i].text for i in old_left])
            eb = embedder.embed_documents([new[j].text for j in new_left])
            score = 0.5 * fuzzy + 0.5 * np.clip(ea @ eb.T, 0, 1)
        pairs = sorted(
            ((score[a, b], a, b) for a in range(len(old_left)) for b in range(len(new_left))),
            key=lambda t: (-t[0], t[1], t[2]),
        )
        used_a: set[int] = set()
        used_b: set[int] = set()
        for s, a, b in pairs:
            if s < MATCH_SCORE:
                break
            if a in used_a or b in used_b:
                continue
            used_a.add(a)
            used_b.add(b)
            o, n = old[old_left[a]], new[new_left[b]]
            if fuzzy[a, b] * 100 >= UNCHANGED_FUZZ:
                result.unchanged += 1
            else:
                result.modified.append((o, n, round(float(s), 3)))
        result.removed = [old[old_left[a]] for a in range(len(old_left)) if a not in used_a]
        result.added = [new[new_left[b]] for b in range(len(new_left)) if b not in used_b]
    else:
        result.removed = [old[i] for i in old_left]
        result.added = [new[j] for j in new_left]
    result.modified.sort(key=lambda t: t[1].start)
    return result


@dataclass(frozen=True)
class SectionRef:
    """Which filing a section came from."""

    cik: str
    ticker: str
    accession_no: str
    form: str
    fiscal_year: int
    item: str


class SectionStore:
    """Finds parsed sections (``processed/sections/<cik>/<accession>.parquet``)."""

    def __init__(self, processed_dir: Path) -> None:
        """Index sections lazily from ``processed_dir``."""
        self.processed_dir = processed_dir

    @cached_property
    def index(self) -> pd.DataFrame:
        """One row per (filing, item): cik, ticker, accession_no, form, fiscal_year, item."""
        cols = ["accession_no", "cik", "ticker", "form", "fiscal_year", "item"]
        files = sorted((self.processed_dir / "sections").glob("*/*.parquet"))
        frames = [pd.read_parquet(f, columns=cols) for f in files]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)

    def available_years(self, cik: str, item: str, form: str = "10-K") -> list[int]:
        """Fiscal years with this item, originals only (amendments are often partial)."""
        idx = self.index
        rows = idx[(idx["cik"] == cik) & (idx["item"] == item) & (idx["form"] == form)]
        return sorted({int(y) for y in rows["fiscal_year"].dropna()})

    def get(
        self, cik: str, item: str, fiscal_year: int, form: str = "10-K"
    ) -> tuple[SectionRef, Section] | None:
        """The section for a company/item/year, or ``None``."""
        idx = self.index
        rows = idx[
            (idx["cik"] == cik)
            & (idx["item"] == item)
            & (idx["form"] == form)
            & (idx["fiscal_year"] == fiscal_year)
        ].sort_values("accession_no")
        if rows.empty:
            return None
        row = rows.iloc[-1]
        path = self.processed_dir / "sections" / cik / f"{row['accession_no']}.parquet"
        frame = pd.read_parquet(path)
        frame = frame[frame["item"] == item]
        section = sections_from_frame(frame)[0]
        ref = SectionRef(
            cik, str(row["ticker"]), str(row["accession_no"]), form, int(fiscal_year), item
        )
        return ref, section


def section_similarity(
    old: Section, new: Section, embedder: Embedder | None = None
) -> dict[str, float | None]:
    """All similarity measures between two section texts."""
    measures: dict[str, float | None] = {**all_measures(old.text, new.text)}
    measures["embedding_cosine"] = None
    if embedder is not None:
        po = [u.text for u in units(old, ("para",))] or [old.text]
        pn = [u.text for u in units(new, ("para",))] or [new.text]
        mo = embedder.embed_documents(po).mean(axis=0)
        mn = embedder.embed_documents(pn).mean(axis=0)
        measures["embedding_cosine"] = round(embedding_cosine(mo, mn), 4)
    return measures
