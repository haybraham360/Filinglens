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

## Acceptance criteria
- [ ] Golden set v1 (≥50 verified questions) committed.
- [ ] Judge-human agreement measured and reported.
- [ ] Results table for E1–E6 on the test split in the README.
