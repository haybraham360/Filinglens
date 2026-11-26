# 05 — Evaluations

## Goal
Measure quality objectively so every change (chunking, hybrid search, reranking, prompts, models) is justified by numbers. **This is the part of the project that most impresses AI engineering interviewers.**

## Datasets

### A. Own golden set (primary): `evals/datasets/golden_v1.jsonl`
50–100 hand-verified questions over the pilot companies. Schema:
```json
{"id": "q001", "question": "...", "ticker": "AAPL", "fiscal_year": 2023,
 "type": "factual|numeric|yoy_change|multi_hop|unanswerable",
 "gold_answer": "...", "gold_chunk_ids": ["..."], "gold_value": 383285000000,
 "expected_tools": ["compute_ratio"], "notes": "..."}
```
Target mix:
- 30% factual lookup (risk/business/MD&A)
- 25% numeric (must use tools; exact value check with tolerance)
- 20% year-over-year change
- 15% multi-hop (numbers + explanation)
- **10% unanswerable**: the system must say it doesn't know. This tests hallucination.

How to build it fast: let an LLM *draft* questions from sampled chunks, then **you verify every one by hand**. Unverified questions don't go in the set. Split it 70/30 into **dev** (tune on this) and **test** (report on this; touch it rarely).

### B. External benchmark: FinanceBench
- FinanceBench (Patronus AI, 2023) is a benchmark for open-book financial QA over public filings. The full set has 10,231 questions. The **open-source sample has 150 annotated questions** with answers and evidence, on Hugging Face as `PatronusAI/financebench` (license CC-BY-NC-4.0, fine for a non-commercial portfolio).
- The paper found that all tested model setups showed weaknesses such as hallucination, which makes it a good stress test.
- Its source documents are PDFs, and some are 8-Ks and earnings releases. Use only the questions whose source is a 10-K/10-Q you can ingest from EDGAR, and document that subset.

## Metrics

**Retrieval** (needs `gold_chunk_ids`)
- Recall@k (k = 5, 10): was a gold chunk retrieved?
- MRR: how high was the first gold chunk?

**Answer quality**
- Correctness: LLM-as-judge against `gold_answer` with a rubric (correct / partially / incorrect). **Validate the judge**: hand-label 30 answers and report agreement with the judge.
- Numeric accuracy: exact match within 0.5% tolerance on `gold_value`.
- Faithfulness: is every claim supported by its cited chunks (LLM judge)?
- Citation validity: share of quotes verifiably present in cited chunks (programmatic).
- Unanswerable handling: share of unanswerable questions correctly refused.

**Agent**
- Tool-selection accuracy vs `expected_tools`
- "No numbers without tools" violations (should be 0)
- Average steps per question

**Ops**
- p50/p95 latency, tokens and cost per question

## Runner
- `python -m evals.run --dataset golden_v1 --split dev --config configs/hybrid_rerank.yaml`
- Saves `evals/reports/{date}_{config}.json` and a markdown summary table.
- `evals/compare.py` diffs two reports, which is how you produce the README "improvements" table.
- CI runs a 10-question smoke subset on every PR (cheap model, fail if correctness drops >10 pts).
- Optional: `ragas` for extra faithfulness/context metrics. Keep the custom metrics as the source of truth.

## Experiments to run and report
| Exp | Variable | Expectation |
|---|---|---|
| E1 | chunk size 300 / 600 / 1000 tokens | find the sweet spot |
| E2 | per-risk-factor chunks vs fixed-size | likely better for Item 1A |
| E3 | context header on/off | likely better retrieval with header |
| E4 | dense vs hybrid (RRF) | hybrid better on exact terms |
| E5 | + reranker | better precision, higher latency |
| E6 | metadata filter on/off | big gain on company/year questions |
| E7 | 2 LLMs | quality vs cost trade-off |

## As built
See `evals/README.md` for the workflow.
- **Datasets:** `evals/dataset.py`. Cases also carry `gold_spans` (accession, item, character range), so retrieval hits are measured by overlap and comparable across chunking configs. `verified_by` is required to score. Splits are a random stratified 70/30 with a fixed seed.
- **Drafting** (`evals/draft.py`):
  - Factual: Opus 5.5 writes questions from sampled chunks.
  - Numeric: from `get_financials`/`compute_ratio`. The reviewer must check each value against the filing, because the tool under test produced it.
  - YoY: added risk headings from `compare_sections`.
  - Multi-hop: a ratio change plus an Opus summary of MD&A drivers.
  - Unanswerable: templated.
- **Verification:** `evals/verify.py` is interactive and shows each candidate's evidence text from the parsed filing.
- **Runner** (`evals/run.py`): `retrieval` / `rag` / `agent` modes, running the app's real entry points.
  - Each run writes `results.jsonl`, `errors.jsonl` and per-case traces, and resumes at (case, rep).
  - Failure classes: API error after jittered retries, timeout (per-case wall-clock ceiling), judge error, served-model mismatch.
  - Truncated answers are excluded from means.
  - `--system oracle|null` checks the harness itself.
- **Judge** (`evals/judge.py`): Opus 5.5 by default, chosen to differ from the Sonnet 5.5 under test. One property per call (correctness, faithfulness), structured outputs, a rubric that says not to reward length, candidate text treated as data, no server fallback, and a known-negatives self-check. Prompts are in `src/filinglens/llm/prompts/judge_*_v1.md`. Judge-human agreement uses `evals/label.py` (exact, binary and Cohen's kappa).
- **Experiments:** E1–E6 run in `retrieval` mode (no LLM cost) through `evals/experiments.py`. Index-time variants (E1–E3) each build an isolated `eval_<fingerprint>` Postgres schema. E7 is `configs/llm_haiku.yaml` against `configs/hybrid.yaml`.
- **CI:** offline harness tests on every PR. The paid 10-question smoke is a manual workflow (`eval-smoke.yml`) that needs `ANTHROPIC_API_KEY` and a corpus snapshot (`make eval-snapshot`).
- **FinanceBench:** `evals/financebench.py` converts the 10-K/10-Q subset to unverified candidates and documents what was dropped.

## Acceptance criteria
- [ ] Golden set v1 (≥50 verified questions) committed.
- [ ] Judge-human agreement measured and reported.
- [ ] Results table for E1–E6 on the test split in the README.
