"""Eval runner (specs/05).

    python -m evals.run --dataset golden_v1 --split dev --config evals/configs/hybrid_rerank.yaml

Writes ``evals/reports/runs/<run_id>/`` (``results.jsonl``, ``errors.jsonl``,
``traces/<case>_rep<k>.json``, ``meta.json``) and a summary
``evals/reports/<run_id>.json`` + ``.md``. Resumable: re-running the same command skips
(case, rep) pairs already in ``results.jsonl``. ``--system oracle`` / ``--system null``
push the gold answers / an empty answer through the same grading to check the harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import queue
import random
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from evals.config import EvalConfig, load_config
from evals.dataset import Case, dataset_hash, load_cases
from evals.judge import JUDGE_PROMPT_VERSION, Judge, JudgeCall, JudgeError
from evals.metrics import (
    RetrievedRef,
    has_retrieval_gold,
    numeric_correct,
    retrieval_metrics,
    tool_selection,
)
from filinglens.rag.schemas import Answer

logger = logging.getLogger("evals")
EVALS_DIR = Path(__file__).parent
REPORTS_DIR = EVALS_DIR / "reports"
HARNESS_FILES = ("config.py", "dataset.py", "metrics.py", "judge.py", "run.py", "index.py")
MAX_ATTEMPTS = 3


class CaseTimeoutError(RuntimeError):
    """The per-case wall-clock ceiling was hit."""


class ServedModelMismatchError(RuntimeError):
    """The response came from a different model than configured (and not via fallback)."""


@dataclass
class SystemOutput:
    """What one system run produced for one case."""

    answer: Answer | None = None
    retrieved: list[RetrievedRef] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    llm_steps: int = 0
    served_model: str | None = None
    fallback_used: bool = False
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)
    cited_sources: list[tuple[int, str]] = field(default_factory=list)


System = Callable[[Case], SystemOutput]


# --- systems ----------------------------------------------------------------------------


def oracle_system(case: Case) -> SystemOutput:
    """Answers with the gold answer and retrieves the gold evidence (should score ~100%)."""
    refs = [
        RetrievedRef(f"gold-{i}", s.accession_no, s.item, s.char_start, s.char_end)
        for i, s in enumerate(case.gold_spans)
    ] or [RetrievedRef(cid, "", "", None, None) for cid in case.gold_chunk_ids]
    text = case.gold_answer
    if case.gold_value is not None:
        text = f"{text} (value: {case.gold_value})"
    answer = Answer(
        answer=text,
        citations=[],
        insufficient_context=case.type == "unanswerable",
        confidence="high",
    )
    return SystemOutput(
        answer=answer,
        retrieved=refs,
        tools_called=list(case.expected_tools),
        served_model="oracle",
        llm_steps=1,
    )


def null_system(case: Case) -> SystemOutput:
    """Empty answer, nothing retrieved, no tools (should score ~0% except refusals)."""
    answer = Answer(answer="", citations=[], insufficient_context=False, confidence="low")
    return SystemOutput(answer=answer, served_model="null")


def _refs(chunks: list[Any]) -> list[RetrievedRef]:
    return [
        RetrievedRef(c.chunk_id, c.accession_no, c.item, c.char_start, c.char_end) for c in chunks
    ]


def make_app_system(cfg: EvalConfig, schema: str) -> Callable[[], System]:
    """Factory of per-thread app systems wired to the eval index ``schema``."""
    import anthropic

    from evals.index import connect_index
    from filinglens.agent.cli import build_agent
    from filinglens.config import get_settings
    from filinglens.llm.calllog import PostgresCallLog
    from filinglens.rag.answer import answer_question
    from filinglens.retrieval.embedder import load_embedder
    from filinglens.retrieval.filters import CompanyDirectory
    from filinglens.retrieval.rerank import CrossEncoderReranker
    from filinglens.retrieval.retriever import RetrievalConfig as AppRetrievalConfig
    from filinglens.retrieval.retriever import Retriever

    settings = get_settings().model_copy(
        update={
            "llm_model": cfg.llm.model,
            "llm_provider": cfg.llm.provider,
            "ollama_model": cfg.llm.model,
            "llm_effort": cfg.llm.effort,
            "embedding_model": cfg.index.embedding_model,
            "retrieval_top_k": cfg.retrieval.top_k,
            "retrieval_candidate_k": cfg.retrieval.candidate_k,
            "use_hybrid": cfg.retrieval.use_hybrid,
            "rrf_k": cfg.retrieval.rrf_k,
            "use_rerank": cfg.retrieval.use_rerank,
            "rerank_top_n": cfg.retrieval.rerank_top_n,
            "rerank_model": cfg.retrieval.rerank_model,
        }
    )
    embedder = load_embedder(cfg.index.embedding_model)  # shared across threads
    reranker = (
        CrossEncoderReranker(cfg.retrieval.rerank_model) if cfg.retrieval.use_rerank else None
    )
    universe = settings.universe_file.stem.removesuffix("_universe")
    directory = CompanyDirectory.load(
        settings.raw_dir / "universe" / f"{universe}_ciks.csv",
        settings.universe_file.parent / "company_aliases.yaml",
    )
    k_eval = max(cfg.retrieval_eval_k, cfg.retrieval.top_k)
    retr_cfg = AppRetrievalConfig(
        top_k=k_eval,
        candidate_k=max(cfg.retrieval.candidate_k, k_eval),
        use_hybrid=cfg.retrieval.use_hybrid,
        rrf_k=cfg.retrieval.rrf_k,
        use_rerank=cfg.retrieval.use_rerank,
        rerank_top_n=max(cfg.retrieval.rerank_top_n, k_eval),
        use_filters=cfg.retrieval.use_filters,
    )
    answer_k = cfg.retrieval.rerank_top_n if cfg.retrieval.use_rerank else cfg.retrieval.top_k

    def factory() -> System:
        from filinglens.agent.cli import build_router_agent
        from filinglens.llm.factory import make_llm

        conn = connect_index(settings.database_url, schema)
        retriever = Retriever(conn, embedder, directory, retr_cfg, reranker)
        llm = make_llm(settings, cfg.llm.provider, PostgresCallLog(conn))
        agent: Any = None
        if cfg.mode == "agent":
            if cfg.llm.provider == "anthropic":
                agent = build_agent(settings, conn, anthropic.Anthropic())
            else:
                agent = build_router_agent(settings, conn, cfg.llm.provider)

        def run(case: Case) -> SystemOutput:
            t0 = time.perf_counter()
            if cfg.mode == "agent":
                assert agent is not None
                res = agent.run(case.question)
                return SystemOutput(
                    answer=res.answer,
                    tools_called=res.tools_called,
                    unsupported_numbers=res.unsupported_numbers,
                    llm_steps=sum(1 for s in res.steps if s.kind == "llm"),
                    served_model=res.model,
                    fallback_used=res.fallback_used,
                    stop_reason=res.final_stop_reason,
                    input_tokens=res.input_tokens,
                    output_tokens=res.output_tokens,
                    latency_s=time.perf_counter() - t0,
                    trace={"steps": [vars(s) for s in res.steps], "transcript": res.transcript},
                    cited_sources=sorted(res.cited_texts.items()),
                )
            retrieval = retriever.retrieve(case.question)
            refs = _refs(retrieval.chunks)
            out = SystemOutput(
                retrieved=refs,
                trace={
                    "filters": vars(retrieval.filters) | {"matched_terms": None},
                    "relaxed": retrieval.relaxed,
                    "retrieved": [
                        {
                            "chunk_id": c.chunk_id,
                            "ranks": c.ranks,
                            "score": c.score,
                            "text": c.text[:500],
                        }
                        for c in retrieval.chunks
                    ],
                },
            )
            if cfg.mode == "retrieval":
                out.latency_s = time.perf_counter() - t0
                return out
            retrieval.chunks = retrieval.chunks[:answer_k]
            ans = answer_question(case.question, retriever, llm, retrieval=retrieval)
            out.answer = ans.answer
            out.served_model = ans.llm_model
            out.fallback_used = ans.fallback_used
            out.stop_reason = ans.stop_reason
            out.input_tokens, out.output_tokens = ans.input_tokens, ans.output_tokens
            out.llm_steps = 1
            out.latency_s = time.perf_counter() - t0
            text_by_id = {c.chunk_id: c.text for c in retrieval.chunks}
            out.cited_sources = [
                (c.n, text_by_id.get(c.chunk_id or "", "")) for c in ans.answer.citations
            ]
            return out

        return run

    return factory


# --- grading ------------------------------------------------------------------------------


def grade(
    case: Case, out: SystemOutput, cfg: EvalConfig, judge: Judge | None
) -> tuple[dict[str, float | None], list[JudgeCall]]:
    """All metrics for one case. Raises JudgeError if the judge fails."""
    g: dict[str, float | None] = {}
    calls: list[JudgeCall] = []
    if cfg.mode != "agent" and has_retrieval_gold(case):
        g.update(retrieval_metrics(out.retrieved, case))
    if cfg.mode == "retrieval" or out.answer is None:
        return g, calls
    a = out.answer
    refused = a.insufficient_context
    if case.type == "unanswerable":
        g["unanswerable_refused"] = float(refused)
    else:
        g["false_refusal"] = float(refused)
    g["citation_validity"] = a.citation_validity
    g["numeric_accuracy"] = numeric_correct(a.answer, case)
    if cfg.mode == "agent" or out.tools_called:
        g["tool_selection"] = tool_selection(out.tools_called, case)
        g["numbers_without_tools"] = float(bool(out.unsupported_numbers))
        g["steps"] = float(out.llm_steps)
    if judge is not None:
        score, call = judge.correctness(case.question, case.gold_answer, a.answer)
        g["correctness"] = score
        calls.append(call)
        if a.citations and not refused:
            sources = out.cited_sources or [(c.n, c.quote) for c in a.citations]
            faith, call = judge.faithfulness(a.answer, sources)
            g["faithfulness"] = faith
            calls.append(call)
    return g, calls


# --- execution ----------------------------------------------------------------------------


def run_with_ceiling(fn: Callable[[], Any], timeout_s: float) -> Any:
    """Run ``fn`` with a hard wall-clock ceiling (the worker thread is abandoned on timeout)."""
    box: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            box.put((True, fn()))
        except BaseException as exc:
            box.put((False, exc))

    threading.Thread(target=target, daemon=True).start()
    try:
        ok, value = box.get(timeout=timeout_s)
    except queue.Empty as exc:
        raise CaseTimeoutError(f"case exceeded {timeout_s:.0f}s") from exc
    if not ok:
        raise value
    return value


def _transient(exc: BaseException) -> bool:
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return False
    return isinstance(
        exc,
        (
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        ),
    )


def harness_sha() -> str:
    """Digest of the runner, metrics, judge and dataset code (scores across changes differ)."""
    h = hashlib.sha256(JUDGE_PROMPT_VERSION.encode())
    for name in HARNESS_FILES:
        h.update((EVALS_DIR / name).read_bytes())
    return h.hexdigest()[:12]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=EVALS_DIR,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class RunWriter:
    """Append-only, thread-safe writers for one run directory."""

    def __init__(self, run_dir: Path) -> None:
        """Create the run directory layout."""
        self.dir = run_dir
        (run_dir / "traces").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def done_keys(self) -> set[tuple[str, int]]:
        """(case, rep) pairs already scored (errors don't count: they get retried)."""
        path = self.dir / "results.jsonl"
        if not path.exists():
            return set()
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        return {(r["case_id"], r["rep"]) for r in rows}

    def result(self, row: dict[str, Any], trace: dict[str, Any]) -> None:
        """Write a scored row and its trace."""
        with self._lock:
            trace_path = self.dir / "traces" / f"{row['case_id']}_rep{row['rep']}.json"
            trace_path.write_text(json.dumps(trace, indent=1, default=str))
            with (self.dir / "results.jsonl").open("a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")

    def error(self, row: dict[str, Any]) -> None:
        """Record a failed attempt (never occupies the (case, rep) slot)."""
        with self._lock, (self.dir / "errors.jsonl").open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")


def run_case(
    case: Case,
    rep: int,
    system: System,
    cfg: EvalConfig,
    judge: Judge | None,
    writer: RunWriter,
    expected_model: str | None,
) -> str:
    """Run, grade and record one (case, rep). Returns the outcome class."""
    attempts = 0
    while True:
        attempts += 1
        try:
            out: SystemOutput = run_with_ceiling(lambda: system(case), cfg.case_timeout_s)
            if (
                expected_model
                and out.served_model
                and out.served_model != expected_model
                and not out.fallback_used
            ):
                raise ServedModelMismatchError(
                    f"served {out.served_model}, expected {expected_model}"
                )
            truncated = out.stop_reason == "max_tokens"
            grades, calls = ({}, []) if truncated else grade(case, out, cfg, judge)
            break
        except Exception as exc:
            if _transient(exc) and attempts < MAX_ATTEMPTS:
                time.sleep(min(60.0, 5.0 * 2 ** (attempts - 1)) * (0.5 + random.random()))
                continue
            cls = (
                "timeout"
                if isinstance(exc, CaseTimeoutError)
                else "judge_error"
                if isinstance(exc, JudgeError)
                else "served_model_mismatch"
                if isinstance(exc, ServedModelMismatchError)
                else "api_error"
                if _transient(exc)
                else "harness_error"
            )
            writer.error(
                {
                    "case_id": case.id,
                    "rep": rep,
                    "failure_class": cls,
                    "error": f"{type(exc).__name__}: {exc}",
                    "attempts": attempts,
                    "ts": datetime.now(UTC).isoformat(),
                }
            )
            logger.warning("%s rep%d: %s (%s)", case.id, rep, cls, exc)
            return cls

    judge_usage = {
        "input_tokens": sum(c.input_tokens for c in calls),
        "output_tokens": sum(c.output_tokens for c in calls),
    }
    row = {
        "case_id": case.id,
        "rep": rep,
        "prompt": case.question,
        "tags": [case.type, case.ticker or "-", case.split or "-"],
        "status": "truncated" if truncated else "ok",
        "stop_reason": out.stop_reason,
        "grade": grades,
        "model": out.served_model,
        "usage": {"input_tokens": out.input_tokens, "output_tokens": out.output_tokens},
        "judge_model": calls[0].model if calls else None,
        "judge_usage": judge_usage,
        "latency_s": round(out.latency_s, 3),
        "tool_calls": len(out.tools_called),
        "meta": {
            "attempts": attempts,
            "fallback_used": out.fallback_used,
            "refused": bool(out.answer and out.answer.insufficient_context),
        },
    }
    trace = {
        "case": case.model_dump(),
        "config": cfg.name,
        "answer": out.answer.model_dump() if out.answer else None,
        "tools_called": out.tools_called,
        "unsupported_numbers": out.unsupported_numbers,
        "system": out.trace,
        "judge": [vars(c) for c in calls],
        "grade": grades,
    }
    writer.result(row, trace)
    return "truncated" if truncated else "ok"


def execute(
    cases: list[Case],
    cfg: EvalConfig,
    system_factory: Callable[[], System],
    judge: Judge | None,
    writer: RunWriter,
    *,
    expected_model: str | None,
) -> dict[str, int]:
    """Run all pending (case, rep) pairs; returns outcome counts."""
    done = writer.done_keys()
    todo = [(c, r) for c in cases for r in range(cfg.reps) if (c.id, r) not in done]
    counts: dict[str, int] = {"skipped_done": len(done)}
    local = threading.local()

    def work(case: Case, rep: int) -> str:
        if not hasattr(local, "system"):
            local.system = system_factory()
        return run_case(case, rep, local.system, cfg, judge, writer, expected_model)

    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
        futures = [pool.submit(work, c, r) for c, r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            outcome = fut.result()
            counts[outcome] = counts.get(outcome, 0) + 1
            if i % 10 == 0 or i == len(futures):
                logger.info("%d/%d cases done", i, len(futures))
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="golden_v1")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--config", type=Path, default=EVALS_DIR / "configs" / "hybrid.yaml")
    parser.add_argument("--system", default="app", choices=["app", "oracle", "null"])
    parser.add_argument("--reps", type=int)
    parser.add_argument("--limit", type=int, help="Only the first N cases (pilot runs).")
    parser.add_argument("--cases", help="Comma-separated case ids.")
    parser.add_argument("--no-judge", action="store_true", help="Programmatic metrics only.")
    parser.add_argument("--judge-model", help="Override the judge model.")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--run-id", help="Defaults to <date>_<config>_<split>[_<system>].")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow cases without verified_by (drafts; never for reported numbers).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_config(args.config)
    if args.reps:
        cfg = cfg.model_copy(update={"reps": args.reps})
    if args.judge_model:
        cfg = cfg.model_copy(
            update={"judge": cfg.judge.model_copy(update={"model": args.judge_model})}
        )
    try:
        cases = load_cases(
            args.dataset,
            None if args.split == "all" else args.split,
            allow_unverified=args.allow_unverified,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.id in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        logger.error("no cases selected")
        return 1
    if args.no_judge or cfg.judge.provider == "none":
        cfg = cfg.model_copy(update={"judge": cfg.judge.model_copy(update={"provider": "none"})})
    if (
        cfg.judge.provider != "none"
        and cfg.mode != "retrieval"
        and cfg.judge.served_name == cfg.llm.served_name
    ):
        logger.error("judge model %s is the model under test; pick another judge", cfg.judge.model)
        return 1

    suffix = "" if args.system == "app" else f"_{args.system}"
    run_id = args.run_id or f"{date.today():%Y-%m-%d}_{cfg.name}_{args.split}{suffix}"
    writer = RunWriter(REPORTS_DIR / "runs" / run_id)

    schema = None
    expected_model: str | None = None
    if args.system == "app":
        from evals.index import ensure_index
        from filinglens.config import get_settings
        from filinglens.retrieval.embedder import load_embedder

        settings = get_settings()
        schema = ensure_index(
            settings.database_url,
            settings.processed_dir,
            cfg.index,
            load_embedder,
            rebuild=args.rebuild_index,
        )
        factory = make_app_system(cfg, schema)
        expected_model = cfg.llm.served_name if cfg.mode != "retrieval" else None
        if cfg.mode == "agent" and cfg.llm.provider != "anthropic":
            expected_model = None  # router agent: no LLM, or the local model only writes
    else:
        fixed = oracle_system if args.system == "oracle" else null_system

        def factory() -> System:
            return fixed

    judge = None
    if cfg.judge.provider != "none" and cfg.mode != "retrieval":
        if cfg.judge.provider == "ollama":
            from filinglens.config import get_settings
            from filinglens.llm.calllog import MemoryCallLog
            from filinglens.llm.ollama import OllamaLLM

            local = OllamaLLM(cfg.judge.model, MemoryCallLog(), url=get_settings().ollama_url)
            judge = Judge(None, cfg.judge.model, llm=local)
        else:
            import anthropic

            judge = Judge(anthropic.Anthropic(), cfg.judge.model, cfg.judge.effort)

    meta = {
        "run_id": run_id,
        "config": cfg.model_dump(),
        "dataset": args.dataset,
        "split": args.split,
        "system": args.system,
        "dataset_hash": dataset_hash(cases),
        "n_cases": len(cases),
        "harness_sha": harness_sha(),
        "git_commit": _git_commit(),
        "index_schema": schema,
        "judge": None
        if judge is None
        else {"model": cfg.judge.model, "prompt": JUDGE_PROMPT_VERSION},
        "started_at": datetime.now(UTC).isoformat(),
    }
    (writer.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    counts = execute(cases, cfg, factory, judge, writer, expected_model=expected_model)
    logger.info("outcomes: %s", counts)

    from evals.report import summarize, write_summary

    summary = summarize(writer.dir)
    json_path, md_path = write_summary(summary, REPORTS_DIR, run_id)
    print(md_path.read_text())
    print(f"\nSummary: {json_path}\nRun dir: {writer.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
