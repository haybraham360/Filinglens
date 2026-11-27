"""Word-list features: Loughran-McDonald sentiment shares and CEO/CFO change mentions.

The Loughran-McDonald Master Dictionary is free from the Notre Dame SRAF site
(https://sraf.nd.edu/loughranmcdonald-master-dictionary/). Download the CSV and point
``--lm-dictionary`` at it. In the CSV, a non-zero value in a category column (e.g.
``Negative``) is the year the word was added to that list, and 0 means not in the list.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = ("Negative", "Positive", "Uncertainty", "Litigious")

# Paper (p. 15): a word from {appoint, elect, hire, new, search} within 10 words of
# {CEO, CFO, Chief Executive Officer, Chief Financial Officer}.
_TRIGGERS = ("appoint", "elect", "hire", "new", "search")
_WINDOW = 10


@dataclass(frozen=True)
class LMDictionary:
    """Upper-cased word sets per Loughran-McDonald category."""

    words: dict[str, frozenset[str]]

    @classmethod
    def from_csv(cls, path: Path, categories: Sequence[str] = CATEGORIES) -> LMDictionary:
        """Load the Master Dictionary CSV (columns ``Word`` plus one per category)."""
        sets: dict[str, set[str]] = {c: set() for c in categories}
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                word = (row.get("Word") or "").strip().upper()
                if not word:
                    continue
                for c in categories:
                    value = (row.get(c) or "0").strip()
                    if value not in ("", "0", "0.0"):
                        sets[c].add(word)
        return cls({c: frozenset(s) for c, s in sets.items()})

    @classmethod
    def from_words(cls, **lists: Iterable[str]) -> LMDictionary:
        """Build from explicit lists, e.g. ``from_words(Negative=["loss"])`` (tests)."""
        return cls({c: frozenset(w.upper() for w in ws) for c, ws in lists.items()})

    def share(self, tokens: Sequence[str], category: str = "Negative") -> float:
        """Fraction of tokens in ``category`` (0 for an empty document)."""
        if not tokens:
            return 0.0
        words = self.words.get(category, frozenset())
        return sum(1 for t in tokens if t.upper() in words) / len(tokens)


def _is_trigger(tok: str) -> bool:
    return any(tok.startswith(t) for t in _TRIGGERS)


def mentions_officer_change(tokens: Sequence[str], role: str) -> bool:
    """True if a hiring word appears within 10 words of the role (``"ceo"`` or ``"cfo"``).

    Matches the paper's rule. "Appoint*", "elect*" and "hire*" are matched as prefixes, so
    "appointed" and "elected" count.
    """
    phrase = (
        ("chief", "executive", "officer") if role == "ceo" else ("chief", "financial", "officer")
    )
    role_at: list[int] = []
    for i, tok in enumerate(tokens):
        if tok == role or tuple(tokens[i : i + 3]) == phrase:
            role_at.append(i)
    if not role_at:
        return False
    trig_at = [i for i, t in enumerate(tokens) if _is_trigger(t)]
    j = 0
    for r in role_at:
        while j < len(trig_at) and trig_at[j] < r - _WINDOW:
            j += 1
        if j < len(trig_at) and trig_at[j] <= r + _WINDOW:
            return True
    return False


def officer_change_flags(tokens: Sequence[str]) -> dict[str, bool]:
    """``{"ceo_change": ..., "cfo_change": ...}`` for one document."""
    return {
        "ceo_change": mentions_officer_change(tokens, "ceo"),
        "cfo_change": mentions_officer_change(tokens, "cfo"),
    }
