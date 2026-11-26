"""Aggregate a run directory into a summary (JSON + markdown)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.metrics import estimate, per_case_means, percentile
from filinglens.llm.pricing import cost_usd

# Ordered: what the markdown table shows, and whether higher is better.
METRICS: list[tuple[str, str, bool]] = [
    ("correctness", "Correctness (judge)", True),
    ("faithfulness", "Faithfulness (judge)", True),
    ("numeric_accuracy", "Numeric accuracy", True),
    ("citation_validity", "Citation validity", True),
    ("unanswerable_refused", "Unanswerable refused", True),
    ("false_refusal", "False refusals", False),
    ("recall_at_5", "Recall@5", True),
    ("recall_at_10", "Recall@10", True),
    ("mrr", "MRR", True),
    ("tool_selection", "Tool selection", True),
    ("numbers_without_tools", "Numbers w/o tools", False),
]
COUNT_METRICS = [("steps", "Avg LLM steps")]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def summarize(run_dir: Path) -> dict[str, Any]:
    """Compute every metric from raw rows (nothing is trusted from aggregates)."""
    meta = json.loads((run_dir / "meta.json").read_text())
    rows = _rows(run_dir / "results.jsonl")
    errors = _rows(run_dir / "errors.jsonl")
    scored = {(r["case_id"], r["rep"]) for r in rows}
    open_errors = [e for e in errors if (e["case_id"], e["rep"]) not in scored]
    ok = [r for r in rows if r["status"] == "ok"]

    metrics: dict[str, Any] = {}
    per_case: dict[str, dict[str, float]] = {}
    for key, _, higher in METRICS + [(k, lbl, True) for k, lbl in COUNT_METRICS]:
        means = per_case_means(rows, key)
        if not means:
            continue
        est = estimate(list(means.values()))
        metrics[key] = {"mean": est.mean, "ci95": est.ci95, "n": est.n, "higher_is_better": higher}
        per_case[key] = means

    by_type: dict[str, dict[str, Any]] = defaultdict(dict)
    types = {r["case_id"]: r["tags"][0] for r in rows}
    for key, means in per_case.items():
        groups: dict[str, list[float]] = defaultdict(list)
        for cid, v in means.items():
            groups[types[cid]].append(v)
        for t, vals in groups.items():
            est = estimate(vals)
            by_type[t][key] = {"mean": est.mean, "ci95": est.ci95, "n": est.n}

    latencies = [r["latency_s"] for r in ok]
    sys_costs = [
        cost_usd(r["model"], r["usage"]["input_tokens"], r["usage"]["output_tokens"])
        for r in ok
        if r.get("model")
    ]
    judge_costs = [
        cost_usd(
            r["judge_model"], r["judge_usage"]["input_tokens"], r["judge_usage"]["output_tokens"]
        )
        for r in ok
        if r.get("judge_model")
    ]
    known_sys = [c for c in sys_costs if c is not None]
    known_judge = [c for c in judge_costs if c is not None]
    ops = {
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "input_tokens_mean": _mean([r["usage"]["input_tokens"] for r in ok]),
        "output_tokens_mean": _mean([r["usage"]["output_tokens"] for r in ok]),
        "cost_per_question_usd": _mean(known_sys),
        "system_cost_total_usd": sum(known_sys),
        "judge_cost_total_usd": sum(known_judge),
        "fallback_rows": sum(1 for r in ok if r["meta"].get("fallback_used")),
        "retried_rows": sum(1 for r in rows if r["meta"].get("attempts", 1) > 1),
    }
    return {
        "run_id": meta["run_id"],
        "meta": meta,
        "counts": {
            "cases": meta["n_cases"],
            "rows": len(rows),
            "ok": len(ok),
            "truncated": sum(1 for r in rows if r["status"] == "truncated"),
            "open_errors": len(open_errors),
            "error_classes": _count(e["failure_class"] for e in open_errors),
        },
        "metrics": metrics,
        "by_type": dict(by_type),
        "ops": ops,
        "per_case": per_case,
    }


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def _fmt(m: dict[str, Any], pct: bool) -> str:
    scale = 100 if pct else 1
    return f"{m['mean'] * scale:.1f} ± {m['ci95'] * scale:.1f} (n={m['n']})"


def to_markdown(s: dict[str, Any]) -> str:
    """Human-readable summary table."""
    meta, c, ops = s["meta"], s["counts"], s["ops"]
    lines = [
        f"## Eval `{s['run_id']}`",
        "",
        f"config `{meta['config']['name']}` · mode `{meta['config']['mode']}` · system "
        f"`{meta['system']}` · dataset `{meta['dataset']}` ({meta['split']}, hash "
        f"`{meta['dataset_hash']}`) · harness `{meta['harness_sha']}` · commit "
        f"`{meta.get('git_commit')}`",
        "",
        f"Cases {c['cases']} · scored rows {c['ok']} · truncated {c['truncated']} · "
        f"unresolved errors {c['open_errors']} {c['error_classes'] or ''}",
        "",
        "| Metric | Mean ± 95% CI (%) |",
        "|---|---|",
    ]
    for key, label, higher in METRICS:
        if key in s["metrics"]:
            arrow = "↑" if higher else "↓"
            lines.append(f"| {label} {arrow} | {_fmt(s['metrics'][key], True)} |")
    for key, label in COUNT_METRICS:
        if key in s["metrics"]:
            lines.append(f"| {label} | {_fmt(s['metrics'][key], False)} |")
    lines += ["", "| Ops | Value |", "|---|---|"]
    for key, val in ops.items():
        shown = "n/a" if val is None else (f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append(f"| {key} | {shown} |")
    if s["by_type"]:
        keys = [k for k, _, _ in METRICS if any(k in v for v in s["by_type"].values())]
        lines += ["", "| Type | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for t in sorted(s["by_type"]):
            cells = [
                f"{s['by_type'][t][k]['mean'] * 100:.0f} (n={s['by_type'][t][k]['n']})"
                if k in s["by_type"][t]
                else "-"
                for k in keys
            ]
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_summary(summary: dict[str, Any], reports_dir: Path, run_id: str) -> tuple[Path, Path]:
    """Write ``<run_id>.json`` and ``<run_id>.md`` next to the runs directory."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{run_id}.json"
    md_path = reports_dir / f"{run_id}.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    md_path.write_text(to_markdown(summary))
    return json_path, md_path
