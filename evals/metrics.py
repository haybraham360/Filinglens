"""Programmatic metrics and aggregation. LLM-judged metrics live in ``evals/judge.py``."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from evals.dataset import Case, GoldSpan
from filinglens.agent.guards import matches_value, numbers_in

MIN_SPAN_OVERLAP = 0.3


@dataclass(frozen=True)
class RetrievedRef:
    """The minimum needed to score one retrieved chunk."""

    chunk_id: str
    accession_no: str
    item: str
    char_start: int | None
    char_end: int | None


def is_hit(ref: RetrievedRef, case: Case) -> bool:
    """Does a retrieved chunk cover gold evidence?

    With gold spans: same filing and item, and the overlap covers at least 30% of the gold
    span or of the chunk (so a big chunk containing a short gold span counts, as does a
    small chunk inside a long one). Without spans: exact chunk-id match.
    """
    if case.gold_spans:
        return any(_overlaps(ref, span) for span in case.gold_spans)
    return ref.chunk_id in set(case.gold_chunk_ids)


def _overlaps(ref: RetrievedRef, span: GoldSpan) -> bool:
    if ref.accession_no != span.accession_no or ref.item != span.item:
        return False
    if ref.char_start is None or ref.char_end is None:
        return False
    overlap = min(ref.char_end, span.char_end) - max(ref.char_start, span.char_start)
    if overlap <= 0:
        return False
    gold_len = max(1, span.char_end - span.char_start)
    chunk_len = max(1, ref.char_end - ref.char_start)
    return overlap / gold_len >= MIN_SPAN_OVERLAP or overlap / chunk_len >= MIN_SPAN_OVERLAP


def has_retrieval_gold(case: Case) -> bool:
    """Retrieval metrics apply only to cases with gold evidence."""
    return bool(case.gold_spans or case.gold_chunk_ids)


def retrieval_metrics(retrieved: Sequence[RetrievedRef], case: Case) -> dict[str, float]:
    """Recall@5, Recall@10 and MRR (first hit within the top 10)."""
    hits = [is_hit(r, case) for r in retrieved[:10]]
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    return {
        "recall_at_5": float(any(hits[:5])),
        "recall_at_10": float(any(hits[:10])),
        "mrr": 1.0 / first if first else 0.0,
    }


def numeric_correct(answer_text: str, case: Case) -> float | None:
    """1.0 if the answer states ``gold_value`` within tolerance (rounding, $bn, % allowed)."""
    if case.gold_value is None:
        return None
    gold = Decimal(str(case.gold_value))
    tol = Decimal(str(case.gold_tolerance))
    values = [v for _, v in numbers_in(answer_text, skip_years_and_counts=False)]
    return float(any(matches_value(v, gold, tol) for v in values))


def tool_selection(tools_called: Iterable[str], case: Case) -> float | None:
    """1.0 if every expected tool was called at least once (extra calls are allowed)."""
    if not case.expected_tools:
        return None
    return float(set(case.expected_tools) <= set(tools_called))


@dataclass(frozen=True)
class Estimate:
    """A mean over cases with a normal-approximation 95% CI half-width."""

    mean: float
    ci95: float
    n: int

    def fmt(self, pct: bool = True) -> str:
        """``72.0 ± 6.1 (n=40)``-style text."""
        if self.n == 0:
            return "n/a"
        scale = 100 if pct else 1
        return f"{self.mean * scale:.1f} ± {self.ci95 * scale:.1f} (n={self.n})"


def estimate(values: Sequence[float]) -> Estimate:
    """Mean and 95% CI of per-case values (reps averaged before calling this)."""
    n = len(values)
    if n == 0:
        return Estimate(float("nan"), float("nan"), 0)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return Estimate(mean, 1.96 * sd / math.sqrt(n), n)


def per_case_means(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, float]:
    """Average a metric over reps for each case, using only status-ok rows that have it."""
    acc: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        value = row.get("grade", {}).get(metric)
        if value is not None:
            acc[row["case_id"]].append(float(value))
    return {cid: statistics.fmean(vs) for cid, vs in acc.items()}


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[k]
