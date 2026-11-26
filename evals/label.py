"""Validate the LLM judge against human labels (specs/05).

    python -m evals.label sample --run evals/reports/runs/<run_id> --n 30   # label blind
    python -m evals.label agreement --run evals/reports/runs/<run_id>

``sample`` shows question, reference answer and the system's answer, never the judge's
verdict, and records your correct / partially correct / incorrect label in
``evals/datasets/judge_labels_v1.jsonl``. ``agreement`` compares those labels with the
judge's grades in the run: exact 3-way agreement, binary agreement (correct vs not), and
Cohen's kappa. The checklist bar is ~90% agreement on clear-cut cases.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from evals.dataset import DATASETS_DIR

LABELS_PATH = DATASETS_DIR / "judge_labels_v1.jsonl"
CHOICES = {"c": 1.0, "p": 0.5, "i": 0.0}


def _rows(run: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in (run / "results.jsonl").read_text().splitlines() if ln.strip()]


def _labels() -> dict[tuple[str, int, str], float]:
    if not LABELS_PATH.exists():
        return {}
    out = {}
    for ln in LABELS_PATH.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[(r["run_id"], r["rep"], r["case_id"])] = r["label"]
    return out


def cohen_kappa(a: list[float], b: list[float]) -> float:
    """Cohen's kappa for two raters over the same items."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(run: Path) -> dict[str, Any]:
    """Judge-vs-human agreement for the labelled rows of one run."""
    run_id = run.name
    labels = _labels()
    human, judge = [], []
    for row in _rows(run):
        key = (run_id, row["rep"], row["case_id"])
        j = row.get("grade", {}).get("correctness")
        if key in labels and j is not None and row["status"] == "ok":
            human.append(labels[key])
            judge.append(float(j))
    n = len(human)
    if n == 0:
        return {"n": 0}
    exact = sum(h == j for h, j in zip(human, judge, strict=True)) / n
    binary = sum((h == 1.0) == (j == 1.0) for h, j in zip(human, judge, strict=True)) / n
    return {
        "n": n,
        "exact_agreement": exact,
        "binary_agreement": binary,
        "kappa": cohen_kappa(human, judge),
        "confusion": Counter(f"human={h} judge={j}" for h, j in zip(human, judge, strict=True)),
    }


def sample(run: Path, n: int, seed: int) -> None:
    """Interactively label a stratified random sample of a run's answers."""
    run_id = run.name
    labels = _labels()
    rows = [r for r in _rows(run) if r["status"] == "ok" and "correctness" in r.get("grade", {})]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in sorted(rows, key=lambda r: (r["case_id"], r["rep"])):
        by_type.setdefault(r["tags"][0], []).append(r)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    for group in by_type.values():
        rng.shuffle(group)
    while len(picked) < n and any(by_type.values()):
        for group in by_type.values():
            if group and len(picked) < n:
                picked.append(group.pop())
    todo = [r for r in picked if (run_id, r["rep"], r["case_id"]) not in labels]
    print(f"{len(todo)} answers to label (judge verdicts hidden).")
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(todo, 1):
        trace = json.loads((run / "traces" / f"{row['case_id']}_rep{row['rep']}.json").read_text())
        print("\n" + "=" * 80)
        print(f"[{i}/{len(todo)}] {row['case_id']} ({row['tags'][0]})\n")
        print(f"Q: {trace['case']['question']}\n")
        print(f"Reference: {trace['case']['gold_answer']}\n")
        print(f"System answer:\n{(trace['answer'] or {}).get('answer', '')}")
        while (
            choice := input("\n[c]orrect  [p]artially  [i]ncorrect  [q]uit > ").strip().lower()
        ) not in {*CHOICES, "q"}:
            pass
        if choice == "q":
            break
        with LABELS_PATH.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "case_id": row["case_id"],
                        "rep": row["rep"],
                        "label": CHOICES[choice],
                    }
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=11)
    a = sub.add_parser("agreement")
    a.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "sample":
        sample(args.run, args.n, args.seed)
        return 0
    result = agreement(args.run)
    if result["n"] == 0:
        print("No labelled rows for this run yet: run `python -m evals.label sample` first.")
        return 1
    md = (
        f"## Judge-human agreement ({args.run.name})\n\n"
        f"| n | exact (3-way) | binary (correct vs not) | Cohen's kappa |\n|---|---|---|---|\n"
        f"| {result['n']} | {result['exact_agreement']:.0%} | {result['binary_agreement']:.0%} "
        f"| {result['kappa']:.2f} |\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in sorted(result["confusion"].items()))
        + "\n"
    )
    out = args.run.parent.parent / f"judge_agreement_{args.run.name}.md"
    out.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
