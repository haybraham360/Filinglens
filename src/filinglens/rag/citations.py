"""Programmatic citation checks: is the quoted span really in the cited chunk?"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from filinglens.parsing.text import normalize_text

QUOTE_THRESHOLD = 0.9
_MAX_ANCHORS = 5


def _norm(text: str) -> str:
    text = normalize_text(text).lower()
    return re.sub(r"\s*\.{3}\s*|\s*…\s*", " ... ", text).strip()


def quote_match_score(quote: str, chunk_text: str) -> float:
    """Best fuzzy match of ``quote`` against any window of ``chunk_text`` (0..1).

    Exact (normalized) substrings score 1.0. Otherwise this is a partial ratio: anchor on
    the longest common blocks, compare the quote-sized window around each, keep the best.
    Ellipses in the quote are allowed: each "..."-separated part is matched separately and
    the scores are length-weighted.
    """
    q, text = _norm(quote), _norm(chunk_text)
    if not q:
        return 0.0
    parts = [p.strip() for p in q.split(" ... ") if p.strip()]
    if len(parts) > 1:
        total = sum(len(p) for p in parts)
        return sum(quote_match_score(p, chunk_text) * len(p) for p in parts) / total
    if q in text:
        return 1.0
    matcher = SequenceMatcher(None, text, q, autojunk=False)
    blocks = sorted(matcher.get_matching_blocks(), key=lambda b: -b.size)[:_MAX_ANCHORS]
    best = 0.0
    for block in blocks:
        if block.size == 0:
            continue
        start = max(0, block.a - block.b)
        window = text[start : start + len(q)]
        best = max(best, SequenceMatcher(None, window, q, autojunk=False).ratio())
    return best


def is_verified(quote: str, chunk_text: str, threshold: float = QUOTE_THRESHOLD) -> bool:
    """True if ``quote`` fuzzily appears in ``chunk_text`` at or above ``threshold``."""
    return quote_match_score(quote, chunk_text) >= threshold
