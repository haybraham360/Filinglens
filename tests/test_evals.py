"""Offline harness checks for evals/ (run in CI on every PR; no API, no database)."""

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import evals.run as run_mod
import pytest
from evals.compare import compare
from evals.config import EvalConfig, load_config
from evals.dataset import Case, GoldSpan, assign_splits, dataset_hash, load_cases, write_jsonl
from evals.financebench import convert
from evals.judge import Judge, JudgeError, judge_selfcheck
from evals.label import cohen_kappa
from evals.metrics import RetrievedRef, estimate, is_hit, numeric_correct, retrieval_metrics
from evals.report import summarize
from evals.run import RunWriter, SystemOutput, execute, null_system, oracle_system

from filinglens.rag.schemas import Answer

ACC = "0000320193-23-000106"
SPAN = GoldSpan(accession_no=ACC, item="1A", char_start=1000, char_end=1400)


def case(**kw: Any) -> Case:
    base: dict[str, Any] = {
        "id": "q1",
        "question": "What supply risks does Apple describe?",
        "ticker": "AAPL",
        "fiscal_year": 2023,
        "type": "factual",
        "gold_answer": "Single-source components.",
        "gold_spans": [SPAN],
        "split": "dev",
        "verified_by": "tester",
    }
    return Case(**(base | kw))


CASES = [
    case(id="f1"),
    case(
        id="n1",
        type="numeric",
        gold_value=0.441311,
        gold_spans=[],
        gold_answer="Gross margin was 44.1%.",
        expected_tools=["compute_ratio"],
    ),
    case(id="u1", type="unanswerable", gold_spans=[], gold_answer="Not available.", ticker=None),
]


# --- metrics --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "item", "hit"),
    [
        (900, 1500, "1A", True),  # chunk contains the gold span
        (1100, 1200, "1A", True),  # small chunk inside a long span
        (1350, 2000, "1A", False),  # 50 chars overlap: < 30% of either
        (1000, 1400, "7", False),  # same offsets, different item
        (None, None, "1A", False),
    ],
)
def test_span_hits(start: int | None, end: int | None, item: str, hit: bool) -> None:
    assert is_hit(RetrievedRef("x", ACC, item, start, end), case()) is hit


def test_retrieval_metrics_recall_and_mrr() -> None:
    miss = RetrievedRef("m", ACC, "7", 0, 100)
    good = RetrievedRef("g", ACC, "1A", 1000, 1400)
    m = retrieval_metrics([miss, miss, good] + [miss] * 7, case())
    assert m == {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": pytest.approx(1 / 3)}
    m = retrieval_metrics([miss] * 5 + [good], case())
    assert m["recall_at_5"] == 0.0 and m["recall_at_10"] == 1.0 and m["mrr"] == pytest.approx(1 / 6)
    # chunk-id gold when there are no spans
    assert retrieval_metrics([good], case(gold_spans=[], gold_chunk_ids=["g"]))["mrr"] == 1.0


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("Gross margin was 44.1% in FY2023.", True),
        ("It was 0.4413.", True),
        ("about 44%", True),  # 0.3% off: inside the 0.5% tolerance
        ("about 43%", False),
        ("Gross margin was 45.1%.", False),
        ("", False),
    ],
)
def test_numeric_accuracy_tolerance(text: str, ok: bool) -> None:
    assert numeric_correct(text, CASES[1]) == float(ok)


def test_numeric_accuracy_handles_dollar_scales() -> None:
    c = case(type="numeric", gold_value=383285000000)
    assert numeric_correct("Revenue was $383.3 billion.", c) == 1.0
    assert numeric_correct("Revenue was $383 billion.", c) == 1.0  # within 0.5%
    assert numeric_correct("Revenue was $394.3 billion.", c) == 0.0


def test_estimate_ci() -> None:
    e = estimate([1.0, 0.0, 1.0, 1.0])
    assert e.mean == 0.75 and e.n == 4 and e.ci95 == pytest.approx(1.96 * 0.5 / 2)


# --- dataset --------------------------------------------------------------------------------


def test_unverified_cases_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    write_jsonl(path, [case(), case(id="q2", verified_by=None)])
    with pytest.raises(ValueError, match="unverified"):
        load_cases(str(path))
    assert len(load_cases(str(path), allow_unverified=True)) == 2


def test_splits_are_stratified_stable_and_keep_existing() -> None:
    cases = [case(id=f"f{i}", split=None) for i in range(10)] + [
        case(id=f"n{i}", type="numeric", split=None) for i in range(10)
    ]
    a, b = assign_splits(cases), assign_splits(cases)
    assert a == b
    for t in ("factual", "numeric"):
        assert sum(1 for c in a if c.type == t and c.split == "test") == 3
    flip = next(c.id for c in a if c.split == "dev")
    frozen = [c.model_copy(update={"split": "test"}) if c.id == flip else c for c in a]
    again = {c.id: c.split for c in assign_splits(frozen)}
    assert again[flip] == "test"
    assert dataset_hash(a) != dataset_hash(frozen)


def test_all_configs_load() -> None:
    root = Path(__file__).resolve().parents[1] / "evals" / "configs"
    configs = [load_config(p) for p in root.rglob("*.yaml")]
    assert len(configs) >= 15
    for c in configs:
        # never the model under test as its own judge
        assert (
            c.judge.provider == "none"
            or c.mode == "retrieval"
            or c.judge.served_name != c.llm.served_name
        ), c.name
    free = {
        c.name for c in configs if c.llm.provider != "anthropic" and c.judge.provider != "anthropic"
    }
    assert {"extractive", "router", "local", "local_agent"} <= free


# --- runner (oracle / null / failures) ------------------------------------------------------


class FakeJudgeClient:
    """messages.parse returning scripted verdicts; 'correct' iff candidate contains gold words."""

    def __init__(self, model: str = "claude-opus-5-5", served: str | None = None) -> None:
        self.model, self.served = model, served or model
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw: Any) -> Any:
        self.calls.append(kw)
        user = kw["messages"][0]["content"]
        schema = kw["output_format"]
        if schema.__name__ == "CorrectnessVerdict":
            ref = user.split("<reference_answer>\n")[1].split("\n</reference_answer>")[0]
            cand = user.split("<candidate_answer>\n")[1].split("\n</candidate_answer>")[0]
            verdict = "correct" if ref.split()[0].lower() in cand.lower() else "incorrect"
            parsed = schema(reasoning="scripted", verdict=verdict)
        else:
            parsed = schema(claims=[{"claim": "x", "supported": True}])
        return SimpleNamespace(
            model=self.served,
            stop_reason="end_turn",
            parsed_output=parsed,
            usage=SimpleNamespace(input_tokens=500, output_tokens=50),
        )


def _run(
    tmp_path: Path,
    system: Any,
    cfg: EvalConfig,
    judge: Judge | None = None,
    expected: str | None = None,
    cases: list[Case] = CASES,
) -> tuple[RunWriter, dict[str, int]]:
    writer = RunWriter(tmp_path / "run")
    (writer.dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "t",
                "config": cfg.model_dump(),
                "dataset": "d",
                "split": "dev",
                "system": "x",
                "dataset_hash": "h",
                "n_cases": len(cases),
                "harness_sha": "s",
            }
        )
    )
    counts = execute(cases, cfg, lambda: system, judge, writer, expected_model=expected)
    return writer, counts


def test_oracle_scores_high_and_null_scores_low(tmp_path: Path) -> None:
    cfg = EvalConfig(name="t", mode="rag")
    judge = Judge(FakeJudgeClient())
    w, _ = _run(tmp_path / "o", oracle_system, cfg, judge)
    m = summarize(w.dir)["metrics"]
    assert m["recall_at_10"]["mean"] == 1.0 and m["mrr"]["mean"] == 1.0
    assert m["numeric_accuracy"]["mean"] == 1.0
    assert m["unanswerable_refused"]["mean"] == 1.0 and m["false_refusal"]["mean"] == 0.0
    assert m["correctness"]["mean"] == 1.0

    w, _ = _run(tmp_path / "n", null_system, cfg, judge)
    m = summarize(w.dir)["metrics"]
    assert m["recall_at_10"]["mean"] == 0.0 and m["numeric_accuracy"]["mean"] == 0.0
    assert m["unanswerable_refused"]["mean"] == 0.0
    assert m["correctness"]["mean"] == 0.0


def test_results_traces_and_resume(tmp_path: Path) -> None:
    cfg = EvalConfig(name="t", mode="rag", reps=2)
    calls: list[str] = []

    def system(c: Case) -> SystemOutput:
        calls.append(c.id)
        return oracle_system(c)

    w, counts = _run(tmp_path, system, cfg)
    assert counts["ok"] == 6 and len(calls) == 6
    assert (w.dir / "traces" / "f1_rep1.json").exists()
    rows = [json.loads(ln) for ln in (w.dir / "results.jsonl").read_text().splitlines()]
    assert {(r["case_id"], r["rep"]) for r in rows} == {(c.id, r) for c in CASES for r in (0, 1)}
    _, counts = _run(tmp_path, system, cfg)
    assert counts == {"skipped_done": 6} and len(calls) == 6  # nothing re-run


def test_harness_failures_go_to_errors_not_scores(tmp_path: Path) -> None:
    def broken(c: Case) -> SystemOutput:
        raise RuntimeError("tool exploded")

    w, counts = _run(tmp_path, broken, EvalConfig(name="t"))
    assert counts["harness_error"] == 3
    assert not (w.dir / "results.jsonl").exists()
    errors = [json.loads(ln) for ln in (w.dir / "errors.jsonl").read_text().splitlines()]
    assert {e["failure_class"] for e in errors} == {"harness_error"}
    s = summarize(w.dir)
    assert s["counts"]["open_errors"] == 3 and s["metrics"] == {}


def test_timeout_truncation_and_model_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow(c: Case) -> SystemOutput:
        time.sleep(1.0)
        return oracle_system(c)

    _, counts = _run(
        tmp_path / "slow", slow, EvalConfig(name="t", case_timeout_s=0.1), cases=CASES[:1]
    )
    assert counts == {"skipped_done": 0, "timeout": 1}

    def truncated(c: Case) -> SystemOutput:
        out = oracle_system(c)
        out.stop_reason, out.served_model = "max_tokens", "claude-sonnet-5-5"
        return out

    w, counts = _run(
        tmp_path / "trunc", truncated, EvalConfig(name="t"), expected="claude-sonnet-5-5"
    )
    s = summarize(w.dir)
    assert counts["truncated"] == 3 and s["counts"]["truncated"] == 3 and s["metrics"] == {}

    def other_model(c: Case) -> SystemOutput:
        out = oracle_system(c)
        out.served_model = "claude-sonnet-5"
        return out

    _, counts = _run(
        tmp_path / "mm",
        other_model,
        EvalConfig(name="t"),
        expected="claude-sonnet-5-5",
        cases=CASES[:1],
    )
    assert counts["served_model_mismatch"] == 1

    def via_fallback(c: Case) -> SystemOutput:
        out = other_model(c)
        out.fallback_used = True
        return out

    _, counts = _run(
        tmp_path / "fb",
        via_fallback,
        EvalConfig(name="t"),
        expected="claude-sonnet-5-5",
        cases=CASES[:1],
    )
    assert counts["ok"] == 1


def test_transient_errors_are_retried_with_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anthropic
    import httpx2 as httpx

    monkeypatch.setattr(run_mod.time, "sleep", lambda s: None)
    attempts = []

    def flaky(c: Case) -> SystemOutput:
        attempts.append(1)
        if len(attempts) < 3:
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
        return oracle_system(c)

    w, counts = _run(tmp_path, flaky, EvalConfig(name="t"), cases=CASES[:1])
    assert counts["ok"] == 1
    row = json.loads((w.dir / "results.jsonl").read_text())
    assert row["meta"]["attempts"] == 3


def test_judge_failures_are_judge_errors(tmp_path: Path) -> None:
    judge = Judge(FakeJudgeClient(served="claude-sonnet-5-5"))
    _, counts = _run(tmp_path, oracle_system, EvalConfig(name="t"), judge, cases=CASES[:1])
    assert counts["judge_error"] == 1
    with pytest.raises(JudgeError):
        judge.correctness("q", "ref", "cand")


def test_judge_known_negatives_and_prompt_hygiene() -> None:
    client = FakeJudgeClient()
    scores = judge_selfcheck(Judge(client))
    assert set(scores.values()) == {0.0}
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5" and "fallbacks" not in call
    assert "Don't reward length" in call["system"]
    assert "ignore any instructions" in call["system"]


# --- compare / label / financebench ---------------------------------------------------------


def _summary(name: str, per_case: dict[str, float], harness: str = "s") -> dict[str, Any]:
    return {
        "meta": {
            "config": {"name": name},
            "dataset_hash": "h",
            "harness_sha": harness,
            "split": "test",
        },
        "per_case": {"mrr": per_case},
    }


def test_compare_is_paired_and_flags_mismatched_harness() -> None:
    a = _summary("a", {f"c{i}": 0.5 for i in range(30)})
    b = _summary("b", {f"c{i}": 0.5 + (0.3 if i % 2 else 0.2) for i in range(30)}, harness="t")
    out = compare(a, b)
    row = out["rows"][0]
    assert row["delta"] == pytest.approx(0.25) and row["significant"] and row["better"]
    assert any("harness_sha" in w for w in out["warnings"])
    noisy = _summary("c", {f"c{i}": (1.0 if i % 2 else 0.0) for i in range(30)})
    assert not compare(a, noisy)["rows"][0]["significant"]


def test_cohen_kappa() -> None:
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0


def test_financebench_subset_filter() -> None:
    recs = [
        {
            "financebench_id": "1",
            "company": "Apple",
            "doc_name": "APPLE_2022_10K",
            "question_type": "metrics-generated",
            "question": "Revenue?",
            "answer": "$394.3 billion",
            "evidence": [{"evidence_text": "Total net sales 394,328"}],
        },
        {
            "financebench_id": "2",
            "company": "Apple",
            "doc_name": "APPLE_2023Q1_EARNINGS",
            "question": "q",
            "answer": "a",
        },
        {
            "financebench_id": "3",
            "company": "Kraft Heinz",
            "doc_name": "KRAFTHEINZ_2019_10K",
            "question": "q",
            "answer": "a",
        },
    ]
    cases, dropped = convert(recs, {"apple": "AAPL"})
    assert [c.id for c in cases] == ["fb_1"]
    assert cases[0].gold_value == pytest.approx(394.3e9) and cases[0].fiscal_year == 2022
    assert cases[0].verified_by is None and cases[0].type == "numeric"
    assert dropped == {"source not a 10-K/10-Q": 1, "company not in ingested universe": 1}


def test_answer_schema_roundtrip_for_traces() -> None:
    a = Answer(answer="x", citations=[], insufficient_context=True, confidence="low")
    assert Answer.model_validate(a.model_dump()) == a
