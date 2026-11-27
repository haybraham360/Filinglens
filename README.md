# FilingLens

An AI analyst for SEC filings (10-K / 10-Q). It answers questions with citations, shows what changed between filings, computes financial metrics with real tools, and tests whether filing changes predict stock returns (a "Lazy Prices" replication with LLMs).

> Work in progress. See `CLAUDE.md` for the project guide and `specs/` for component specs.

**It runs for free.** With no API key, answers are extractive (verbatim, cited filing sentences) and a rule-based router picks the tools (XBRL financials, exact ratios, year-over-year risk-factor diffs). A local model through [Ollama](https://ollama.com) can write the answers, and Claude takes over when `ANTHROPIC_API_KEY` is set. Numbers always come from code, never from a language model.

## Quick start (Docker)

Requirements: Docker, plus [uv](https://docs.astral.sh/uv/) for the ingest commands.

```bash
cp .env.example .env                 # set EDGAR_IDENTITY="Your Name you@example.com"
make db                              # Postgres 16 + pgvector on localhost:5433
uv sync                              # Python deps for the CLI
uv run filinglens ingest --tickers AAPL,MSFT,JPM --since 2022   # SEC EDGAR (free)
uv run filinglens parse
uv run filinglens embed
make up                              # db + api + frontend
```

Then open http://localhost:8501 for the app and http://localhost:8000/docs for the API.

Optional:
- **Local model:** `brew install ollama && ollama pull qwen2.5:3b` (about 2 GB), and keep Ollama running.
- **Claude:** set `ANTHROPIC_API_KEY` in `.env`. `LLM_PROVIDER` picks the provider explicitly (`auto` by default).

## Without Docker

```bash
make setup                    # Python deps (all extras) + landing page deps
uv run filinglens doctor      # checks your configuration
make db
make api                      # http://localhost:8000
make frontend                 # http://localhost:8501
```

## Common commands

| Command | What it does |
|---|---|
| `filinglens ask "…"` | Cited answer from the command line (`--provider none|ollama|anthropic`) |
| `filinglens agent "…" --trace` | Answer with tools, showing each step |
| `make up` | db, api and frontend in Docker |
| `make experiments` / `make eval-free` | Free evals (see `evals/README.md`) |
| `make loadtest` | 20 concurrent `/ask` requests against a running API |
| `make seed-export` | Bundle the ingested corpus for a deployment (see `deploy/README.md`) |
| `make check` | Lint, type-check and test |
| `npm run dev` | Landing page on http://localhost:5173 |

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/ready` | Liveness; readiness (DB, provider, chunk count) |
| GET | `/companies` | Ingested companies and fiscal years |
| POST | `/ask` | `{question, ticker?, fiscal_year?, mode?, quality?}` → cited answer + trace id |
| POST | `/ask/stream` | Same, as Server-Sent Events (`step` per tool call, then `answer`) |
| POST | `/compare` | `{ticker, item, year_a?, year_b?}` → added/removed/modified passages + similarity |
| GET | `/financials/{ticker}` | Revenue, income, margins, growth, ROE, current ratio from XBRL |
| GET | `/filings/{accession}/chunk/{chunk_id}` | Source text behind a citation |
| GET | `/filings/{accession}/section/{item}?start&end` | Passage behind a diff citation |
| GET | `/traces/{trace_id}` | Steps behind an answer |

Rate-limited per IP, with the stricter limit on `quality: high`. Answers are cached, logs are JSON with request and trace ids, and the daily LLM spend cap falls back to the free mode instead of failing.

## Repository layout

```
specs/             component specs (ingestion, parsing, RAG, agent, evals, deployment, research)
src/filinglens/    core package: ingestion, parsing, db, retrieval, llm, agent, api, research
config/            pilot universe, XBRL concept map, company aliases
frontend/          Streamlit app; frontend/landing is the static landing page
evals/             golden datasets, runners, configs, reports
deploy/            production compose + Caddy (VPS); render.yaml at the root (Render)
loadtest/          concurrent /ask load test
tests/
```

Data from SEC EDGAR. Not investment advice.
