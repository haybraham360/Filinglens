"""Document similarity measures shared by `compare_sections` (specs/04) and Lazy Prices (specs/07).

All measures return a similarity in [0, 1] (1 = identical) and work on word tokens.
Following Cohen, Malloy & Nguyen (2020), lower similarity between consecutive filings
means more change. Exact definitions here are FilingLens's own and are documented per
function, so research results can state precisely what was computed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from difflib import SequenceMatcher

import numpy as np
from rapidfuzz.distance import Levenshtein

_WORD = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens (letters/digits, keeping internal apostrophes and hyphens)."""
    return _WORD.findall(text.lower())


def cosine_tf(a: Sequence[str], b: Sequence[str]) -> float:
    """Cosine similarity of raw term-frequency vectors."""
    ca, cb = Counter(a), Counter(b)
    if not ca or not cb:
        return 1.0 if not ca and not cb else 0.0
    dot = sum(ca[w] * cb[w] for w in ca.keys() & cb.keys())
    norm = math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values()))
    return dot / norm


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Jaccard similarity of the two word sets: size of intersection / size of union."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def min_edit(a: Sequence[str], b: Sequence[str]) -> float:
    """``1 - word-level Levenshtein distance / max(len(a), len(b))``.

    Exact minimum number of word insertions, deletions and substitutions, computed with
    rapidfuzz's bit-parallel implementation (fast enough for whole Item 1A sections).
    """
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - Levenshtein.distance(list(a), list(b)) / longest


def simple(a: Sequence[str], b: Sequence[str]) -> float:
    """Diff-based similarity: ``1 - (words added + words deleted) / (len(a) + len(b))``.

    Uses a ``diff``-style alignment (difflib); a replaced word counts as one deletion
    plus one addition, so the measure is 0 for disjoint documents and 1 for identical ones.
    """
    total = len(a) + len(b)
    if total == 0:
        return 1.0
    changed = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            changed += (i2 - i1) + (j2 - j1)
    return 1.0 - changed / total


def embedding_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two (mean-pooled) embedding vectors, clipped to [0, 1]."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / (na * nb))))


def all_measures(text_a: str, text_b: str) -> dict[str, float]:
    """The four classical Lazy Prices measures on two texts (embeddings handled separately)."""
    ta, tb = tokenize(text_a), tokenize(text_b)
    return {
        "cosine_tf": round(cosine_tf(ta, tb), 4),
        "jaccard": round(jaccard(ta, tb), 4),
        "min_edit": round(min_edit(ta, tb), 4),
        "simple": round(simple(ta, tb), 4),
    }
