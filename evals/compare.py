"""Compare two eval summaries (paired by case) and print a markdown table.

    python -m evals.compare evals/reports/<baseline>.json evals/reports/<candidate>.json

Differences use the cases both runs scored: mean of per-case (B - A) with a 95% CI. A
change counts as a real difference only when the CI excludes zero. Runs on different
dataset versions or harness versions are flagged, because their scores aren't comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.metrics import estimate
from evals.report import COUNT_METRICS, METRICS


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Paired per-metric differences ``b - a``."""
    out: dict[str, Any] = {"warnings": [], "rows": []}
    for key in ("dataset_hash", "harness_sha"):
        if a["meta"].get(key) != b["meta"].get(key):
            out["warnings"].append(
                f"{key} differs ({a['meta'].get(key)} vs {b['meta'].get(key)}): "
                "scores may not be comparable."
            )
    if a["meta"].get("split") != b["meta"].get("split"):
        out["warnings"].append("runs are on different splits")
    labels = {k: (lbl, hib) for k, lbl, hib in METRICS} | {
        k: (lbl, True) for k, lbl in COUNT_METRICS
    }
    for key, (label, higher) in labels.items():
        pa, pb = a["per_case"].get(key, {}), b["per_case"].get(key, {})
        common = sorted(set(pa) & set(pb))
        if not common:
            continue
        diff = estimate([pb[c] - pa[c] for c in common])
        mean_a = sum(pa[c] for c in common) / len(common)
        mean_b = sum(pb[c] for c in common) / len(common)
        significant = abs(diff.mean) > diff.ci95
        better = (diff.mean > 0) == higher if significant else None
        out["rows"].append(
            {
                "metric": key,
                "label": label,
                "a": mean_a,
                "b": mean_b,
                "delta": diff.mean,
                "ci95": diff.ci95,
                "n": diff.n,
                "significant": significant,
                "better": better,
            }
        )
    return out


def to_markdown(a: dict[str, Any], b: dict[str, Any], result: dict[str, Any]) -> str:
    """README-style improvements table."""
    pct = {k for k, _, _ in METRICS}
    lines = [
        f"| Metric | {a['meta']['config']['name']} | {b['meta']['config']['name']} | Δ (95% CI) |",
        "|---|---|---|---|",
    ]
    for r in result["rows"]:
        scale = 100 if r["metric"] in pct else 1
        mark = "" if not r["significant"] else (" ✅" if r["better"] else " ⚠️")
        lines.append(
            f"| {r['label']} | {r['a'] * scale:.1f} | {r['b'] * scale:.1f} | "
            f"{r['delta'] * scale:+.1f} ± {r['ci95'] * scale:.1f} (n={r['n']}){mark} |"
        )
    for w in result["warnings"]:
        lines.append(f"\n> ⚠ {w}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)
    a = json.loads(args.baseline.read_text())
    b = json.loads(args.candidate.read_text())
    print(to_markdown(a, b, compare(a, b)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
