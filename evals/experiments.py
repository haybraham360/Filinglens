"""Run the retrieval experiments E1-E6 and write the README results table.

    python -m evals.experiments --split test

Each config in ``evals/configs/experiments/`` runs in ``retrieval`` mode (no LLM calls, so
no API cost), then is compared with ``r_default`` on the same cases. Output:
``evals/reports/experiments_<split>.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from evals.compare import compare
from evals.config import load_config
from evals.run import EVALS_DIR, REPORTS_DIR
from evals.run import main as run_main

EXPERIMENTS = {
    "r_default": "reference",
    "e1_chunk300": "E1 chunk size 300",
    "e1_chunk600": "E1 chunk size 600",
    "e1_chunk1000": "E1 chunk size 1000",
    "e2_fixed": "E2 fixed-size chunks",
    "e3_no_header": "E3 no context header",
    "e4_dense": "E4 dense only",
    "e5_rerank": "E5 + reranker",
    "e6_no_filters": "E6 no metadata filter",
}
COLUMNS = (("recall_at_5", "Recall@5"), ("recall_at_10", "Recall@10"), ("mrr", "MRR"))


def main(argv: list[str] | None = None) -> int:
    """Run every experiment config, then tabulate against the reference."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="golden_v1")
    parser.add_argument("--split", default="test", choices=["dev", "test", "all"])
    parser.add_argument("--only", help="Comma-separated experiment names.")
    args = parser.parse_args(argv)
    names = args.only.split(",") if args.only else list(EXPERIMENTS)
    if "r_default" not in names:
        names.insert(0, "r_default")

    summaries = {}
    for name in names:
        path = EVALS_DIR / "configs" / "experiments" / f"{name}.yaml"
        load_config(path)  # fail fast on a bad config
        run_id = f"{date.today():%Y-%m-%d}_{name}_{args.split}"
        code = run_main(
            [
                "--dataset",
                args.dataset,
                "--split",
                args.split,
                "--config",
                str(path),
                "--no-judge",
                "--run-id",
                run_id,
            ]
        )
        if code != 0:
            return code
        summaries[name] = json.loads((REPORTS_DIR / f"{run_id}.json").read_text())

    ref = summaries["r_default"]
    header = (
        "| Experiment | "
        + " | ".join(lbl for _, lbl in COLUMNS)
        + " | Δ MRR vs reference (95% CI) |"
    )
    lines = [
        f"### Retrieval experiments ({args.dataset}, {args.split} split)",
        "",
        header,
        "|---" * (len(COLUMNS) + 2) + "|",
    ]
    for name in names:
        s = summaries[name]
        cells = [
            f"{s['metrics'][k]['mean'] * 100:.1f}" if k in s["metrics"] else "-" for k, _ in COLUMNS
        ]
        delta = "-"
        if name != "r_default":
            row = next((r for r in compare(ref, s)["rows"] if r["metric"] == "mrr"), None)
            if row:
                flag = "" if not row["significant"] else (" ✅" if row["better"] else " ⚠️")
                delta = f"{row['delta'] * 100:+.1f} ± {row['ci95'] * 100:.1f}{flag}"
        lines.append(f"| {EXPERIMENTS.get(name, name)} | " + " | ".join(cells) + f" | {delta} |")
    n = ref["metrics"].get("mrr", {}).get("n", 0)
    lines += [
        "",
        f"n = {n} cases with gold evidence. ✅/⚠️ = CI of the paired difference "
        "excludes zero (better/worse).",
    ]
    out = REPORTS_DIR / f"experiments_{args.split}.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
