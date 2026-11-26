"""Golden dataset schema, loading, and stratified dev/test splits."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DATASETS_DIR = Path(__file__).parent / "datasets"
CaseType = Literal["factual", "numeric", "yoy_change", "multi_hop", "unanswerable"]
TARGET_MIX: dict[str, float] = {
    "factual": 0.30,
    "numeric": 0.25,
    "yoy_change": 0.20,
    "multi_hop": 0.15,
    "unanswerable": 0.10,
}


class GoldSpan(BaseModel):
    """Evidence location: a character range in one section of one filing.

    Spans (not chunk ids) make retrieval metrics comparable across chunking configs: a
    retrieved chunk is a hit if it overlaps a gold span.
    """

    model_config = ConfigDict(extra="forbid")
    accession_no: str
    item: str
    char_start: int
    char_end: int


class Case(BaseModel):
    """One golden question (specs/05 schema, plus evidence spans and provenance)."""

    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    ticker: str | None = None
    fiscal_year: int | None = None
    type: CaseType
    gold_answer: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_spans: list[GoldSpan] = Field(default_factory=list)
    gold_value: float | None = None
    gold_tolerance: float = 0.005
    expected_tools: list[str] = Field(default_factory=list)
    split: Literal["dev", "test"] | None = None
    verified_by: str | None = None
    source: str = "golden"
    notes: str = ""


def dataset_path(name: str) -> Path:
    """``evals/datasets/<name>.jsonl`` (or ``name`` itself if it is a path)."""
    p = Path(name)
    return p if p.suffix == ".jsonl" else DATASETS_DIR / f"{name}.jsonl"


def read_jsonl(path: Path) -> list[Case]:
    """Parse every line of a JSONL dataset file."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [Case.model_validate_json(ln) for ln in lines]


def write_jsonl(path: Path, cases: list[Case]) -> None:
    """Write cases sorted by id (stable diffs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [c.model_dump_json(exclude_none=False) for c in sorted(cases, key=lambda c: c.id)]
    path.write_text("\n".join(rows) + ("\n" if rows else ""))


def load_cases(
    name: str, split: str | None = None, *, allow_unverified: bool = False
) -> list[Case]:
    """Load a dataset split. Unverified cases are refused unless explicitly allowed."""
    path = dataset_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} doesn't exist yet. Draft candidates with `python -m evals.draft`, then "
            "verify them with `python -m evals.verify --reviewer <name>`."
        )
    cases = read_jsonl(path)
    if not allow_unverified:
        unverified = [c.id for c in cases if not c.verified_by]
        if unverified:
            raise ValueError(
                f"{len(unverified)} unverified cases in {name} (e.g. {unverified[:3]}); "
                "verify them with `python -m evals.verify` first."
            )
    if split in ("dev", "test"):
        cases = [c for c in cases if c.split == split]
    return cases


def assign_splits(cases: list[Case], test_frac: float = 0.3, seed: int = 20251125) -> list[Case]:
    """Random 70/30 split stratified by type. Cases that already have a split keep it."""
    rng = random.Random(seed)
    by_type: dict[str, list[Case]] = defaultdict(list)
    for case in sorted(cases, key=lambda c: c.id):
        by_type[case.type].append(case)
    out: list[Case] = []
    for group in by_type.values():
        unassigned = [c for c in group if c.split is None]
        rng.shuffle(unassigned)
        n_test_target = round(len(group) * test_frac)
        n_test_existing = sum(1 for c in group if c.split == "test")
        n_test = max(0, n_test_target - n_test_existing)
        for i, case in enumerate(unassigned):
            out.append(case.model_copy(update={"split": "test" if i < n_test else "dev"}))
        out.extend(c for c in group if c.split is not None)
    return sorted(out, key=lambda c: c.id)


def dataset_hash(cases: list[Case]) -> str:
    """Content hash, so reports from different dataset versions are never compared."""
    blob = json.dumps([c.model_dump() for c in sorted(cases, key=lambda c: c.id)], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
