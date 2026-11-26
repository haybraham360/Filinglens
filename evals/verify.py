"""Hand-verify drafted questions into the golden set (specs/05).

    python -m evals.verify --reviewer "Your Name"          # review candidates one by one
    python -m evals.verify --assign-splits                 # 70/30 dev/test, stratified by type
    python -m evals.verify --status                        # counts vs the target mix

Shows each candidate with its evidence text (the gold chunk or span from the parsed
filing). Accept, edit (opens $EDITOR on the JSON), reject, or skip. Accepted cases get
``verified_by`` and go into ``golden_v1.jsonl``; rejected ids go into
``golden_v1.rejected.txt`` so they aren't shown again.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from evals.dataset import TARGET_MIX, Case, assign_splits, dataset_path, read_jsonl, write_jsonl


def evidence_text(case: Case, processed_dir: Path, limit: int = 1500) -> list[str]:
    """Text of each gold span from the parsed sections parquet."""
    import pandas as pd

    out = []
    for span in case.gold_spans:
        matches = list((processed_dir / "sections").glob(f"*/{span.accession_no}.parquet"))
        if not matches:
            out.append(f"[{span.accession_no} Item {span.item}: sections file not found]")
            continue
        df = pd.read_parquet(matches[0])
        rows = df[df["item"] == span.item]
        if rows.empty:
            out.append(f"[{span.accession_no}: Item {span.item} not in sections]")
            continue
        text = str(rows.iloc[0]["text"])[span.char_start : span.char_end]
        out.append(
            f"[{span.accession_no} Item {span.item} {span.char_start}-{span.char_end}]\n"
            + (text if len(text) <= limit else text[:limit] + " ...")
        )
    return out


def _edit(case: Case) -> Case | None:
    editor = os.environ.get("EDITOR", "vi")
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fh:
        fh.write(json.dumps(case.model_dump(), indent=2))
        path = fh.name
    try:
        while True:
            subprocess.run([editor, path], check=False)
            try:
                return Case.model_validate_json(Path(path).read_text())
            except ValidationError as exc:
                print(f"Invalid case: {exc}")
                if input("Edit again? [Y/n] ").strip().lower() == "n":
                    return None
    finally:
        Path(path).unlink(missing_ok=True)


def status(dataset: str) -> str:
    """Verified counts by type and split, next to the target mix."""
    path = dataset_path(dataset)
    cases = read_jsonl(path) if path.exists() else []
    total = len(cases)
    by_type: Counter[str] = Counter(c.type for c in cases)
    lines = [f"{total} verified cases in {path.name} (spec target: 50-100)"]
    for t, share in TARGET_MIX.items():
        n = by_type.get(t, 0)
        splits = Counter(c.split for c in cases if c.type == t)
        lines.append(
            f"  {t:13} {n:3} ({n / total:5.0%} vs {share:.0%} target)  "
            f"dev={splits.get('dev', 0)} test={splits.get('test', 0)} "
            f"unassigned={splits.get(None, 0)}"
            if total
            else f"  {t:13} 0"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="golden_v1")
    parser.add_argument("--reviewer", help="Your name, stored as verified_by.")
    parser.add_argument("--assign-splits", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)

    golden_path = dataset_path(args.dataset)
    if args.status:
        print(status(args.dataset))
        return 0
    if args.assign_splits:
        cases = assign_splits(read_jsonl(golden_path))
        write_jsonl(golden_path, cases)
        print(status(args.dataset))
        return 0
    if not args.reviewer:
        parser.error("--reviewer is required to verify cases")

    from filinglens.config import get_settings

    processed = get_settings().processed_dir
    cand_path = dataset_path(f"{args.dataset}.candidates")
    rejected_path = golden_path.with_suffix(".rejected.txt")
    golden = {c.id: c for c in read_jsonl(golden_path)} if golden_path.exists() else {}
    rejected = set(rejected_path.read_text().split()) if rejected_path.exists() else set()
    pending = [c for c in read_jsonl(cand_path) if c.id not in golden and c.id not in rejected]
    print(f"{len(pending)} candidates to review ({len(golden)} already verified).")

    for i, case in enumerate(pending, 1):
        print("\n" + "=" * 80)
        print(
            f"[{i}/{len(pending)}] {case.id}  type={case.type}  ticker={case.ticker}  "
            f"FY{case.fiscal_year}  source={case.source}"
        )
        print(f"\nQ: {case.question}\nGold answer: {case.gold_answer}")
        if case.gold_value is not None:
            print(f"Gold value: {case.gold_value}  (tolerance {case.gold_tolerance:.1%})")
        if case.expected_tools:
            print(f"Expected tools: {', '.join(case.expected_tools)}")
        if case.notes:
            print(f"Notes: {case.notes}")
        for text in evidence_text(case, processed):
            print("\n--- evidence ---\n" + text)
        while True:
            choice = input("\n[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit > ").strip().lower()
            if choice in {"a", "e", "r", "s", "q"}:
                break
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "r":
            rejected.add(case.id)
            rejected_path.write_text("\n".join(sorted(rejected)) + "\n")
            continue
        if choice == "e":
            edited = _edit(case)
            if edited is None:
                continue
            case = edited
        golden[case.id] = case.model_copy(update={"verified_by": args.reviewer})
        write_jsonl(golden_path, list(golden.values()))
    print("\n" + status(args.dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
