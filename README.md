# FilingLens

An AI analyst for SEC filings (10-K / 10-Q). It answers questions with citations, shows what changed between filings, computes financial metrics with real tools, and tests whether filing changes predict stock returns (a "Lazy Prices" replication with LLMs).

> Work in progress. See `CLAUDE.md` for the project guide and `specs/` for component specs.

## Setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 22+, and Docker for the database.

```bash
cp .env.example .env          # then set EDGAR_IDENTITY and ANTHROPIC_API_KEY
make setup                    # Python deps (all extras) + landing page deps
uv run filinglens doctor      # checks your configuration
make db                       # Postgres 16 + pgvector, schema applied on first start
```

## Common commands

| Command | What it does |
|---|---|
| `make api` | Run the FastAPI backend on http://localhost:8000 |
| `make frontend` | Run the Streamlit app on http://localhost:8501 |
| `npm run dev` | Run the landing page on http://localhost:5173 |
| `make up` | Run db, api and frontend together in Docker |
| `make check` | Lint, type-check and test |

## Repository layout

```
specs/             component specs (ingestion, parsing, RAG, agent, evals, deployment, research)
src/filinglens/    core package: ingestion, parsing, db, retrieval, llm, agent, api, research
config/            pilot universe and XBRL concept map
frontend/          Streamlit app; frontend/landing is the static landing page
evals/             golden datasets, runners, reports
notebooks/         exploration only
tests/
```

Data from SEC EDGAR. Not investment advice.
