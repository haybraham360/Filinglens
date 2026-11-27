"""Document similarity measures shared by `compare_sections` (specs/04) and Lazy Prices (specs/07).

All measures return a similarity in [0, 1] (1 = identical). Lower similarity between
consecutive filings means more change. Definitions follow Cohen, Malloy & Nguyen (2020),
Section II (NBER w25084, pp. 12-14), with the choices the paper leaves open stated per
function:

* ``cosine_tf``: cosine of raw term-frequency vectors (paper: "term frequency vectors").
* ``jaccard``: intersection over union of the two word *sets* ("the Jaccard measure is
  binary").
* ``min_edit``: smallest number of word insertions and deletions turning one document into
  the other. The paper's example counts replacing three words as "deleting 3 words ... and
  adding 3 words", so substitutions cost 2 (an Indel/LCS distance, not Levenshtein). The
  paper does not state its normalisation. We use ``1 - distance / (len(a) + len(b))``.
* ``simple``: the "side-by-side" ``diff`` / Track Changes comparison. Lines are diffed, the
  words in added, deleted and changed lines are counted, and the count ``c`` is divided by
  the combined size of both documents. The paper then feature-scales to ``(c_max - c) /
  c_max``. That is affine in ``c`` for a fixed ``c_max``, so quintile ranks and Fama-MacBeth
  t-stats are unchanged. We report ``1 - c`` and keep ``c`` as ``change_ratio``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from rapidfuzz.distance import Indel

_WORD = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens (letters/digits, keeping internal apostrophes and hyphens)."""
    return _WORD.findall(text.lower())


def tokenize_lines(text: str) -> list[list[str]]:
    """Word tokens per non-empty line: the unit ``diff`` works on for ``simple``."""
    return [toks for line in text.splitlines() if (toks := tokenize(line))]


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


def _ids(seqs: Sequence[Sequence[str]] | Sequence[str], vocab: dict[object, int]) -> list[int]:
    """Map hashable items (words or token tuples) to ints so rapidfuzz compares them fast."""
    out = []
    for s in seqs:
        key = s if isinstance(s, str) else tuple(s)
        out.append(vocab.setdefault(key, len(vocab)))
    return out


def min_edit(a: Sequence[str], b: Sequence[str]) -> float:
    """``1 - (word insertions + deletions) / (len(a) + len(b))``.

    Exact minimum (LCS-based Indel distance, bit-parallel in rapidfuzz), fast enough for
    whole 10-Ks. A replaced word costs one deletion plus one insertion, as in the paper.
    """
    total = len(a) + len(b)
    if total == 0:
        return 1.0
    vocab: dict[object, int] = {}
    return 1.0 - Indel.distance(_ids(a, vocab), _ids(b, vocab)) / total


def change_ratio(a_lines: Sequence[Sequence[str]], b_lines: Sequence[Sequence[str]]) -> float:
    """The paper's ``c``: words in changed, added and deleted lines / (size(D1) + size(D2)).

    Lines are compared whole (as ``diff`` does), with a minimal line-level alignment.
    """
    total = sum(map(len, a_lines)) + sum(map(len, b_lines))
    if total == 0:
        return 0.0
    vocab: dict[object, int] = {}
    ia, ib = _ids(a_lines, vocab), _ids(b_lines, vocab)
    changed = 0
    for op in Indel.opcodes(ia, ib):
        if op.tag != "equal":
            changed += sum(len(x) for x in a_lines[op.src_start : op.src_end])
            changed += sum(len(x) for x in b_lines[op.dest_start : op.dest_end])
    return changed / total


def simple(a_lines: Sequence[Sequence[str]], b_lines: Sequence[Sequence[str]]) -> float:
    """Simple (``diff``) similarity, ``1 - change_ratio``, so 1 = identical and 0 = disjoint."""
    return 1.0 - change_ratio(a_lines, b_lines)


def feature_scale_simple(c: np.ndarray, c_max: float) -> np.ndarray:
    """The paper's ``Sim_Simple = (c_max - c) / c_max``.

    Take ``c_max`` from data available at the time (e.g. the formation window), never from
    the full sample.
    """
    if c_max <= 0:
        return np.ones_like(c, dtype=float)
    return np.clip((c_max - c) / c_max, 0.0, 1.0)


def embedding_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two (mean-pooled) embedding vectors, clipped to [0, 1]."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / (na * nb))))


def all_measures(text_a: str, text_b: str, *, with_min_edit: bool = True) -> dict[str, float]:
    """The four classical Lazy Prices measures on two texts (embeddings handled separately).

    ``with_min_edit=False`` skips the most expensive measure (returned as NaN).
    """
    ta, tb = tokenize(text_a), tokenize(text_b)
    la, lb = tokenize_lines(text_a), tokenize_lines(text_b)
    return {
        "cosine_tf": round(cosine_tf(ta, tb), 4),
        "jaccard": round(jaccard(ta, tb), 4),
        "min_edit": round(min_edit(ta, tb), 4) if with_min_edit else math.nan,
        "simple": round(simple(la, lb), 4),
    }
