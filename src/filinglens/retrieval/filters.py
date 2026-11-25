"""Rule-based extraction of company / fiscal year / form / item filters from a question.

Filtering before vector search matters for filings: every company's Risk Factors read
alike, so an unfiltered search for "Apple supply chain risks" happily returns Dell's.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|plc|ltd|llc|holdings|group|the|"
    r"platforms|systems|technologies)\b\.?",
    re.I,
)
_YEAR = re.compile(r"\b(?:fy\s?|fiscal\s+(?:year\s+)?)?(19[9]\d|20[0-4]\d)\b", re.I)
_FY_SHORT = re.compile(r"\bfy\s?'?(\d{2})\b", re.I)
_ITEM = re.compile(r"\bitem\s+(\d{1,2}[a-c]?)\b", re.I)
_10K = re.compile(r"\b(10-?k|annual report)\b", re.I)
_10Q = re.compile(r"\b(10-?q|quarterly report|quarter)\b", re.I)


@dataclass(frozen=True)
class QueryFilters:
    """Metadata filters applied before ranking. Empty fields mean "no filter"."""

    ciks: tuple[str, ...] = ()
    tickers: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    forms: tuple[str, ...] = ()
    items: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = field(default=(), compare=False)

    @property
    def is_empty(self) -> bool:
        """True when no filter was extracted."""
        return not (self.ciks or self.fiscal_years or self.forms or self.items)


@dataclass(frozen=True)
class CompanyDirectory:
    """Ticker / company-name lookup for the ingested universe."""

    by_ticker: dict[str, str]
    by_alias: dict[str, str]

    @classmethod
    def load(cls, universe_csv: Path, aliases_yaml: Path | None = None) -> CompanyDirectory:
        """Build from ingestion's ``<universe>_ciks.csv`` snapshot plus optional aliases.

        Aliases come from the SEC company name (suffixes stripped), its first word when that
        word is distinctive, and ``aliases_yaml`` (``TICKER: [alias, ...]``) for names
        people actually use ("Google", "Disney").
        """
        by_ticker: dict[str, str] = {}
        by_alias: dict[str, str] = {}
        if universe_csv.exists():
            with universe_csv.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    ticker, cik = row["ticker"].upper(), row["cik"]
                    by_ticker[ticker] = cik
                    for alias in _name_aliases(row.get("name", "")):
                        by_alias.setdefault(alias, cik)
        if aliases_yaml is not None and aliases_yaml.exists():
            extra = yaml.safe_load(aliases_yaml.read_text()) or {}
            for ticker, names in extra.items():
                cik = by_ticker.get(str(ticker).upper())
                if cik:
                    for name in names:
                        by_alias[str(name).lower()] = cik
        return cls(by_ticker=by_ticker, by_alias=by_alias)

    def tickers_for(self, ciks: tuple[str, ...]) -> tuple[str, ...]:
        """Tickers for the given CIKs, in input order."""
        reverse = {v: k for k, v in self.by_ticker.items()}
        return tuple(reverse[c] for c in ciks if c in reverse)


_GENERIC_WORDS = {"general", "american", "united", "international", "first", "national"}


def _name_aliases(name: str) -> list[str]:
    cleaned = re.sub(r"[^\w&\s-]", " ", _SUFFIXES.sub(" ", name)).lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" &-")
    if not cleaned:
        return []
    aliases = [cleaned]
    first = cleaned.split()[0]
    if len(first) >= 4 and first not in _GENERIC_WORDS and first != cleaned:
        aliases.append(first)
    return aliases


def parse_filters(question: str, directory: CompanyDirectory) -> QueryFilters:
    """Extract filters from ``question``.

    Tickers must appear in upper case (so "cat" the animal isn't Caterpillar); company names
    match case-insensitively on word boundaries. ``FY2023``/``fiscal 2023``/``2023`` set the
    fiscal year; ``10-K``/``annual report`` and ``10-Q``/``quarter`` set the form; an explicit
    ``Item 1A`` sets the item.
    """
    ciks: list[str] = []
    matched: list[str] = []
    for token in re.findall(r"\b[A-Z][A-Z.\-]{0,5}\b", question):
        cik = directory.by_ticker.get(token.replace(".", "-"))
        if cik and cik not in ciks:
            ciks.append(cik)
            matched.append(token)
    lowered = question.lower()
    for alias in sorted(directory.by_alias, key=len, reverse=True):
        if re.search(rf"(?<![\w-]){re.escape(alias)}(?:'s)?(?![\w-])", lowered):
            cik = directory.by_alias[alias]
            if cik not in ciks:
                ciks.append(cik)
            matched.append(alias)

    years = {int(m.group(1)) for m in _YEAR.finditer(question)}
    years |= {2000 + int(m.group(1)) for m in _FY_SHORT.finditer(question)}

    forms: list[str] = []
    if _10K.search(question):
        forms.append("10-K")
    if _10Q.search(question):
        forms.append("10-Q")

    items = tuple(dict.fromkeys(m.group(1).upper() for m in _ITEM.finditer(question)))
    return QueryFilters(
        ciks=tuple(ciks),
        tickers=directory.tickers_for(tuple(ciks)),
        fiscal_years=tuple(sorted(years)),
        forms=tuple(forms),
        items=items,
        matched_terms=tuple(matched),
    )


_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "main",
        "key",
        "major",
        "describe",
        "explain",
        "tell",
        "me",
        "about",
        "company",
        "company's",
        "companies",
        "fiscal",
        "year",
        "fy",
        "filing",
        "filings",
        "report",
        "annual",
        "quarterly",
    ]
)


def lexical_query(question: str, filters: QueryFilters) -> str:
    """Turn a question into an OR query for ``websearch_to_tsquery``.

    ``websearch_to_tsquery`` ANDs plain words, so a full question rarely matches anything.
    Quoted phrases are kept as phrases; other words (minus stopwords and the terms already
    used as filters) are joined with ``OR`` and ranked by ``ts_rank_cd``.
    """
    phrases = re.findall(r'"([^"]+)"', question)
    rest = re.sub(r'"[^"]+"', " ", question)
    skip = {t.lower() for t in filters.matched_terms} | {str(y) for y in filters.fiscal_years}
    skip_words = {w for t in skip for w in t.split()}
    words = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9&\-]*|\d+(?:\.\d+)?%?", rest):
        lw = word.lower().removesuffix("'s")
        if lw in _STOPWORDS or lw in skip_words or re.fullmatch(r"fy\d+", lw) or len(lw) < 2:
            continue
        words.append(lw)
    terms = [f'"{p}"' for p in phrases] + list(dict.fromkeys(words))
    return " OR ".join(terms)
