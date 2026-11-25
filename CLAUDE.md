# FilingLens — Project Guide

> An AI analyst for SEC filings (10-K / 10-Q) that answers questions with citations, detects what changed between filings, computes financial metrics with real tools, and tests whether filing changes predict stock returns ("Lazy Prices" replication with LLMs).

This is the master project file. Claude Code reads it automatically at the start of every session. Detailed specs live in `specs/`. Read the relevant spec before starting a component.

---

## 1. Goals

1. **Portfolio-grade AI engineering project:** RAG, agents and tool use, structured outputs, evals, and deployment, all done properly.
2. **Quant research extension:** replicate and extend Cohen, Malloy & Nguyen (2020) *Lazy Prices* using classical and embedding-based similarity, with rigorous backtesting.
3. **Candidate for the MScFE capstone** (WorldQuant University).

### Non-goals (for v1)
- Real-time trading, or any live money.
- Non-US filings (20-F, IFRS). Possibly in v2.
- PDF filings. EDGAR HTML/XBRL only.
- User accounts and auth beyond a simple API key.

---

## 2. Core features (v1)

| # | Feature | Spec |
|---|---|---|
| F1 | Ingest 10-K/10-Q filings + XBRL facts for a chosen universe of companies | `specs/01-data-ingestion.md` |
| F2 | Parse filings into Items (1, 1A, 7, 7A…) and chunk with metadata | `specs/02-parsing-and-chunking.md` |
| F3 | Q&A with hybrid retrieval + reranking + **citations** | `specs/03-retrieval-and-rag.md` |
| F4 | Agent with tools: search, year-over-year diff, financial ratios | `specs/04-agent-and-tools.md` |
| F5 | Eval suite (retrieval + answer quality + tool correctness) | `specs/05-evals.md` |
| F6 | FastAPI backend, simple frontend, Docker, deployment | `specs/06-api-frontend-deployment.md` |
| F7 | Lazy Prices research: similarity signals → portfolios → backtest | `specs/07-research-lazy-prices.md` |

Reference sources and links: `specs/REFERENCES.md`

---

## 3. Architecture

```
                 ┌───────────────────────────────┐
SEC EDGAR ──────►│ ingestion/  (rate-limited,     │──► data/raw/  (filings, companyfacts JSON)
(data.sec.gov,   │  cached, idempotent)          │
 Archives)       └──────────────┬────────────────┘
                                ▼
                 ┌───────────────────────────────┐
                 │ parsing/  Item splitter,       │──► data/processed/ (sections, chunks .parquet)
                 │  cleaner, chunker              │
                 └──────────────┬────────────────┘
                                ▼
                 ┌───────────────────────────────┐
                 │ Postgres + pgvector            │  chunks(embedding, tsvector, metadata)
                 │  + financial_facts table       │  facts(cik, concept, period, value)
                 └──────────────┬────────────────┘
                                ▼
User ─► frontend ─► FastAPI ─► agent/ ──► tools: search_filings, compare_sections,
                                  │              get_financials, compute_ratio
                                  ▼
                               LLM API ─► answer + citations (structured JSON)

evals/      ─► golden datasets + runners + reports (run in CI)
research/   ─► similarity signals, portfolio construction, backtests (notebooks + src)
```

---

## 4. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | type hints everywhere |
| Env / deps | `uv` (or poetry) | lockfile committed |
| EDGAR access | `edgartools` + raw `httpx` for data.sec.gov | must set identity / User-Agent |
| DB | PostgreSQL 16 + `pgvector` | one DB for vectors, full-text and facts |
| Embeddings | provider API or local `sentence-transformers` | model name stored per chunk |
| LLM | Claude API (or any provider behind an interface) | provider abstraction in `llm/` |
| Backend | FastAPI + Pydantic v2 | |
| Frontend | Streamlit (v1) | React optional later |
| Evals | custom runner + pytest; optional `ragas` | results saved as JSON + markdown |
| Research | pandas, numpy, statsmodels, scikit-learn | |
| Infra | Docker + docker-compose; deploy to Render/Railway/Fly | |
| CI | GitHub Actions: lint, tests, small eval subset | |

---

## 5. Repository layout

```
filinglens/
├── CLAUDE.md
├── README.md                 # public-facing, written at the end
├── pyproject.toml
├── docker-compose.yml
├── .env.example              # never commit .env
├── specs/                    # specs (this folder)
├── src/filinglens/
│   ├── config.py             # pydantic-settings
│   ├── ingestion/            # edgar client, downloaders, cache
│   ├── parsing/              # item splitter, cleaner, chunker
│   ├── db/                   # schema, migrations, repositories
│   ├── retrieval/            # embed, hybrid search, rerank
│   ├── llm/                  # provider interface, prompts
│   ├── agent/                # tools + orchestration loop
│   ├── api/                  # FastAPI app
│   └── research/             # similarity measures, portfolio, backtest
├── frontend/                 # streamlit app
├── evals/
│   ├── datasets/             # golden Q&A jsonl
│   ├── runners/
│   └── reports/
├── notebooks/                # exploration only, no core logic
├── data/                     # gitignored (raw/, processed/, cache/)
└── tests/
```

---

## 6. Conventions and rules for Claude Code

**General**
- Small, focused modules. Core logic goes in `src/`, never only in notebooks.
- Every public function gets type hints and a docstring. Use `ruff` + `mypy`.
- Config comes from env vars via `pydantic-settings`. Never hardcode keys or emails.
- Write tests alongside code. Mock network calls in unit tests.

**SEC EDGAR (hard rules)**
- Always send a declared User-Agent (`"FilingLens <your-email>"`) read from env.
- Stay **under 10 requests/second** globally. Use a shared rate limiter.
- Cache every response on disk. Never download the same file twice.
- Ingestion must be idempotent and resumable.

**LLM usage**
- All prompts live in `src/filinglens/llm/prompts/`, versioned.
- The LLM **never does arithmetic** for reported answers. Numbers come from tools.
- Every answer returns structured output: `answer`, `citations[]` (filing accession, item, chunk id, quote), `confidence`.
- Log every LLM call (model, tokens, latency, cost) to a table or file.

**Research (hard rules)**
- **No lookahead:** signals use only information available at the filing's acceptance date. Portfolio entry is the month *after* the filing.
- Always report survivorship-bias caveats when price data excludes delisted stocks.
- Be aware of LLM **pretraining leakage**: LLMs may "know" what happened to a company later. See `specs/07`.
- Walk-forward / out-of-sample splits only. Never tune on the test period.

**Git**
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`).
- `data/`, `.env` and model caches are gitignored.

---

## 7. Milestones

| Week | Milestone | Done when |
|---|---|---|
| 1 | Ingestion | 10-Ks + 10-Qs (5 years) + companyfacts for 20 pilot tickers cached locally |
| 2 | Parsing + chunking | Items 1, 1A, 7, 7A extracted for >95% of pilot filings; chunks in parquet |
| 3 | Baseline RAG | Q&A with citations via CLI; pgvector populated |
| 4 | Evals v1 | 50+ golden questions; baseline scores recorded |
| 5 | Retrieval upgrades | hybrid + rerank; eval scores improve and are documented |
| 6 | Agent + tools | ratios and YoY diff tools working, tool-correctness evals pass |
| 7 | API + deploy | Dockerized, deployed, public URL |
| 8–9 | Research | similarity signals for full universe; quintile backtest + factor regressions |
| 10 | Polish | README, architecture diagram, demo video, blog post |

**Pilot universe:** 20 large US companies across sectors (e.g. AAPL, MSFT, JPM, XOM, JNJ, WMT, TSLA, NVDA, KO, BA, PFE, GS, AMZN, META, CVX, PG, DIS, INTC, NFLX, CAT). The research phase expands to a broad universe; see `specs/07`.

---

## 8. Definition of done (whole project)

- [ ] Live URL where anyone can ask questions about the pilot companies
- [ ] Answers include clickable citations to the exact filing section
- [ ] Eval report in the README with before/after retrieval improvements
- [ ] Research report: methodology, results table, factor-adjusted alphas, limitations
- [ ] README with architecture diagram, setup instructions and a demo GIF/video
- [ ] CI green: lint, unit tests, eval smoke test
