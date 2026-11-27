# FilingLens evals

How FilingLens is measured (specs/05). Every retrieval, prompt or model change should come with a before/after comparison from here.

## Running for free (no API key)

Everything except the LLM judge and the Claude drafter runs at zero cost:

| What | Command |
|---|---|
| Retrieval experiments E1–E6 (Recall@k, MRR) | `make experiments` |
| Extractive answers: citation validity, refusals, retrieval | `python -m evals.run --config evals/configs/free/extractive.yaml` |
| Rule-based agent: tool selection, numbers without tools | `python -m evals.run --config evals/configs/free/router.yaml` |
| Local model (Ollama `qwen2.5:3b`) writing answers | `python -m evals.run --config evals/configs/free/local.yaml` |

The free configs skip the judge, so correctness and faithfulness aren't scored. Every programmatic metric still is. You can judge with a local model (`judge: {provider: ollama, model: ...}`), but small judges are unreliable: check them against your own labels with `evals.label agreement` before trusting them. A local judge must also differ from the model under test. The golden set can be written and verified by hand, with no drafter needed: `evals/dataset.py` documents the schema.

## Workflow

1. **Draft candidates** from the ingested corpus. Factual and multi-hop drafts make paid Opus calls; the other types are built from the tools.
   ```
   python -m evals.draft --n 90
   ```
2. **Verify every candidate by hand.** Accept, edit or reject; only verified cases are ever scored.
   ```
   python -m evals.verify --reviewer "Your Name"
   python -m evals.verify --assign-splits     # stratified 70/30 dev/test
   python -m evals.verify --status            # counts vs the target mix
   ```
3. **Check the harness** before any paid run. The oracle should score about 100% and null about 0%.
   ```
   make eval-offline
   ```
4. **Run** a config on dev. This is paid: the model under test answers and Opus 5.5 judges.
   ```
   python -m evals.run --dataset golden_v1 --split dev --config evals/configs/hybrid.yaml --limit 5   # pilot first
   python -m evals.run --dataset golden_v1 --split dev --config evals/configs/hybrid.yaml --reps 2
   ```
5. **Compare** two runs. The comparison is paired by case and reports a 95% CI on each difference.
   ```
   python -m evals.compare evals/reports/<a>.json evals/reports/<b>.json
   ```
6. **Validate the judge.** Label 30 answers blind, then report agreement with the judge.
   ```
   python -m evals.label sample --run evals/reports/runs/<run_id> --n 30
   python -m evals.label agreement --run evals/reports/runs/<run_id>
   ```
7. **Run the retrieval experiments E1–E6** on the test split. This makes no LLM calls.
   ```
   make experiments          # writes evals/reports/experiments_test.md for the README
   ```

## Layout

| Path | What |
|---|---|
| `datasets/golden_v1.jsonl` | Verified golden set (committed) |
| `datasets/golden_v1.candidates.jsonl` | Unverified drafts |
| `datasets/judge_labels_v1.jsonl` | Human labels for judge validation |
| `configs/*.yaml` | Full-pipeline configs: `hybrid`, `dense`, `hybrid_rerank`, `agent`, `llm_haiku` (E7), `smoke` |
| `configs/experiments/*.yaml` | Retrieval-only E1–E6 configs plus the `r_default` reference |
| `reports/<run_id>.json` / `.md` | Run summaries (committed) |
| `reports/runs/<run_id>/` | `results.jsonl`, `errors.jsonl`, `traces/`, `meta.json` (gitignored) |

## Metrics

| Metric | How | Applies to |
|---|---|---|
| Recall@5, Recall@10, MRR | A retrieved chunk is a hit if it overlaps a gold evidence span by ≥30% (of the span or of the chunk). Spans make the numbers comparable across chunking configs | cases with gold evidence |
| Correctness | Opus 5.5 judge vs the reference: correct 1, partially correct 0.5, incorrect 0 | rag, agent |
| Faithfulness | Opus 5.5 judge: share of claims supported by the cited sources' full text | answers with citations |
| Numeric accuracy | A number in the answer matches `gold_value` within 0.5%; rounding, $bn and % are accepted | numeric cases |
| Citation validity | Share of quotes found in their cited source (programmatic, ≥0.9 fuzzy) | answers with citations |
| Unanswerable refused / false refusals | `insufficient_context` on unanswerable / answerable cases | all |
| Tool selection, numbers w/o tools, steps | Expected tools ⊆ tools called; any number not traceable to a tool result; LLM steps | agent |
| p50/p95 latency, tokens, cost | From each response's `usage` and the model that actually served it; judge cost reported separately | all |

## Harness guarantees

- **Failures aren't scores.** API errors after retries, timeouts, judge failures and served-model mismatches go to `errors.jsonl` with a failure class and are re-run on resume. Truncated answers are marked `status: truncated` and left out of the means. Refusals are their own metric.
- **Resumable at (case, rep).** Rows are written as each case finishes, and re-running the same command skips finished pairs.
- **Served-model check.** A response from a different model fails the attempt, unless it came through the documented refusal fallback, which is flagged per row. The judge runs without a fallback.
- **Isolated indexes.** Each index config (chunk size, strategy, header, embedding model) gets its own `eval_<fingerprint>` Postgres schema built from cached sections, so production `public.chunks` is never modified.
- **Comparability.** Each run records the dataset hash, harness sha and git commit, and `compare.py` warns when they differ.
- **Variance.** `--reps N` runs each case N times; every metric reports a 95% CI over cases. The noise floor for a pass rate is about `1/sqrt(n·reps)`, so 50 cases × 2 reps is about ±10 points.

## CI

Every PR runs the offline harness tests (`tests/test_evals.py`): metric math, oracle/null runs, failure classes, resume and the judge rubric. The paid 10-question smoke eval is a manual workflow (`.github/workflows/eval-smoke.yml`). It needs the `ANTHROPIC_API_KEY` secret and the `EVAL_CORPUS_URL` variable, which points to a tarball made with `make seed-export`.

## FinanceBench

```
python -m evals.financebench --input data/financebench/<file>.jsonl
python -m evals.verify --dataset financebench_subset --reviewer "Your Name"
```
This keeps only questions sourced from a 10-K or 10-Q, for companies in the ingested universe. It writes the kept and dropped counts to `datasets/financebench_subset.md`.
