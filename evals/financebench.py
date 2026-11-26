"""Convert the FinanceBench open sample to candidate cases (specs/05, dataset B).

    huggingface-cli download PatronusAI/financebench --repo-type dataset \
        --local-dir data/financebench
    python -m evals.financebench --input data/financebench/<file>.jsonl

FinanceBench (Patronus AI, CC-BY-NC-4.0) evidence comes from PDFs, some of them 8-Ks and
earnings releases. Only questions whose source is a 10-K or 10-Q are kept, and only for
companies FilingLens can map to an ingested ticker. The kept/dropped counts and reasons
go to ``evals/datasets/financebench_subset.md``. Kept questions are unverified
candidates: evidence locations differ between PDFs and EDGAR HTML, so check each one
with ``python -m evals.verify --dataset financebench_subset``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evals.dataset import DATASETS_DIR, Case, write_jsonl

TYPE_MAP = {
    "metrics-generated": "numeric",
    "domain-relevant": "factual",
    "novel-generated": "factual",
}
REQUIRED = ("question", "answer")


def _form(record: dict[str, Any]) -> str | None:
    raw = str(record.get("doc_type") or record.get("doc_name") or "").lower().replace("-", "")
    if "10k" in raw:
        return "10-K"
    if "10q" in raw:
        return "10-Q"
    return None


def _year(record: dict[str, Any]) -> int | None:
    for key in ("doc_period", "doc_name"):
        m = re.search(r"(20\d{2})", str(record.get(key) or ""))
        if m:
            return int(m.group(1))
    return None


def _number(answer: str) -> float | None:
    m = re.search(r"-?\$?\s?([\d,]+(?:\.\d+)?)\s*(%|billion|million)?", answer)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    return value / 100 if unit == "%" else value * {"billion": 1e9, "million": 1e6}.get(unit, 1)


def convert(
    records: list[dict[str, Any]], aliases: dict[str, str]
) -> tuple[list[Case], Counter[str]]:
    """Map FinanceBench records to candidate cases; returns (cases, drop reasons)."""
    dropped: Counter[str] = Counter()
    cases: list[Case] = []
    for rec in records:
        if any(k not in rec for k in REQUIRED):
            dropped["missing question/answer"] += 1
            continue
        form = _form(rec)
        if form is None:
            dropped["source not a 10-K/10-Q"] += 1
            continue
        company = str(rec.get("company", "")).strip()
        ticker = aliases.get(company.lower())
        if ticker is None:
            dropped["company not in ingested universe"] += 1
            continue
        qtype = TYPE_MAP.get(str(rec.get("question_type", "")), "factual")
        answer = str(rec["answer"])
        evidence = rec.get("evidence") or []
        ev_text = " | ".join(
            str(e.get("evidence_text", ""))[:300] for e in evidence if isinstance(e, dict)
        )
        fid = str(
            rec.get("financebench_id") or hashlib.sha256(rec["question"].encode()).hexdigest()[:8]
        )
        cases.append(
            Case(
                id=f"fb_{fid}",
                question=str(rec["question"]),
                ticker=ticker,
                fiscal_year=_year(rec),
                type=qtype,
                gold_answer=answer,
                gold_value=_number(answer) if qtype == "numeric" else None,
                source=f"financebench:{rec.get('doc_name', '')}",
                notes=f"{form}; FinanceBench evidence (PDF): {ev_text}",
            )
        )
    return cases, dropped


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="FinanceBench .jsonl file")
    args = parser.parse_args(argv)
    from filinglens.config import get_settings
    from filinglens.retrieval.filters import CompanyDirectory

    settings = get_settings()
    universe = settings.universe_file.stem.removesuffix("_universe")
    directory = CompanyDirectory.load(
        settings.raw_dir / "universe" / f"{universe}_ciks.csv",
        settings.universe_file.parent / "company_aliases.yaml",
    )
    by_cik = {cik: t for t, cik in directory.by_ticker.items()}
    aliases = {alias: by_cik[cik] for alias, cik in directory.by_alias.items()}
    aliases |= {t.lower(): t for t in directory.by_ticker}

    records = [json.loads(ln) for ln in args.input.read_text().splitlines() if ln.strip()]
    cases, dropped = convert(records, aliases)
    write_jsonl(DATASETS_DIR / "financebench_subset.candidates.jsonl", cases)
    report = [
        f"# FinanceBench subset\n\nSource: {args.input.name}, {len(records)} records "
        f"(PatronusAI/financebench, CC-BY-NC-4.0).\n",
        f"Kept {len(cases)} (10-K/10-Q sources, companies in the ingested universe).\n",
        "| Dropped because | n |",
        "|---|---|",
    ]
    report += [f"| {k} | {v} |" for k, v in dropped.most_common()]
    report += [
        "",
        "Kept by type: " + ", ".join(f"{k}={v}" for k, v in Counter(c.type for c in cases).items()),
    ]
    (DATASETS_DIR / "financebench_subset.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
