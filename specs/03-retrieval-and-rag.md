# 03 — Retrieval & RAG

## Goal
Answer natural-language questions about filings with accurate, **cited** answers. Build it in measured steps: baseline → hybrid → rerank, with eval scores recorded at each step (see `specs/05`).

## Step 1: Baseline (dense only)
- Embed chunks (with context headers) and store them in pgvector. Use an HNSW index with cosine distance.
- Query: embed question → top-k (k=8) → LLM answers using only the retrieved chunks.
- **Metadata filtering first:** parse ticker/year/form from the question (LLM extraction or simple rules) and filter with `WHERE cik = … AND fiscal_year = …` before vector search. This matters a lot for filings, because every company's Risk Factors look alike.

## Step 2: Hybrid search
- Add Postgres full-text search: a `tsvector` column with a GIN index, queried with `websearch_to_tsquery`.
- Fuse dense and lexical rankings with **Reciprocal Rank Fusion (RRF)**: `score = Σ 1/(k + rank_i)`, k=60 by default. Standard Postgres guides (Supabase, DigitalOcean, Railway) document this exact pattern in plain SQL, with no extra search engine needed.
- Why: exact terms like "goodwill impairment", "Item 7A", product names and tickers are often missed by pure vector search. One practitioner write-up reported retrieval precision going from about 62% to 84% after adding full-text + RRF. Treat that as anecdotal and **measure on your own eval set**.

## Step 3: Reranking
- Retrieve the top 30–50 candidates from hybrid search, rerank with a cross-encoder (e.g. a `bge-reranker` model locally, or a hosted rerank API), and keep the top 6–8.
- Track the latency and cost added. Rerank only when the eval gain justifies it.

## Answer generation
**System prompt rules** (version them in `llm/prompts/answer_v1.md`):
- Answer only from the provided context. If the context is insufficient, say so.
- Every factual sentence must cite a chunk ID.
- Never compute numbers yourself. Request the `compute_ratio` / `get_financials` tools (see `specs/04`).
- Distinguish what the company *says* from fact ("The company states that…").

**Structured output (Pydantic):**
```python
class Citation(BaseModel):
    chunk_id: str
    accession_no: str
    item: str
    quote: str  # short verbatim span, verified to exist in the chunk


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool
    confidence: Literal["low", "medium", "high"]
```
- **Citation verification:** after generation, check programmatically that each `quote` appears in the cited chunk text (fuzzy match ≥0.9). Drop or flag failures. Log the citation-validity rate as a metric.

## Year-over-year questions
Questions like "what new risks did X add in 2023?" are routed to the `compare_sections` tool (`specs/04`), not to plain RAG.

## Config knobs (all eval-tunable)
`chunk_size, overlap, top_k, use_hybrid, rrf_k, use_rerank, rerank_top_n, embedding_model, llm_model`

## Acceptance criteria
- [ ] CLI: `filinglens ask "What are Apple's main supply chain risks in FY2023?"` returns an answer with valid citations.
- [ ] Eval report showing baseline vs hybrid vs hybrid+rerank on the same dataset.
- [ ] Citation validity ≥95%.
- [ ] p95 latency logged (target under 8s without rerank).
