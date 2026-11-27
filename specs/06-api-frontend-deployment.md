# 06 — API, Frontend & Deployment

## Goal
Make FilingLens a real, usable product with a public URL, while keeping costs low and protecting API keys.

## Backend (FastAPI)
Endpoints:
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/companies` | list ingested companies + available years |
| POST | `/ask` | `{question, ticker?, fiscal_year?}` → `Answer` (with citations + trace id) |
| POST | `/ask/stream` | Server-Sent Events streaming version |
| POST | `/compare` | `{ticker, item, year_a, year_b}` → diff result |
| GET | `/financials/{ticker}` | key metrics table |
| GET | `/filings/{accession}/chunk/{chunk_id}` | source text for citation pop-ups |
| GET | `/traces/{trace_id}` | agent steps (debug/"show reasoning") |

Production basics:
- Pydantic request/response models, so OpenAPI docs are generated automatically.
- **Rate limiting** per IP (e.g. `slowapi`) and a **daily LLM spend cap** checked before each call. A public demo needs both.
- Cache answers to identical questions (keyed on question + config version).
- Structured JSON logging, plus request ID and trace ID on every log line.
- Secrets only from env vars. `.env.example` lists them without values.
- CORS restricted to the frontend origin.

## Frontend (Streamlit, v1)
- Company + year selectors and a chat box.
- Answers with numbered citations. Clicking one shows the source passage with the quote highlighted, plus a link to the filing on sec.gov.
- A "What changed?" tab: pick company + two years, see added/removed risk factors side by side.
- A "Financials" tab: key ratios chart.
- A collapsible "How I got this answer" panel showing agent tool calls.
- Example questions on the landing page, so a recruiter can try it in 10 seconds.

## Docker
- `Dockerfile` (multi-stage, slim Python image, non-root user).
- `docker-compose.yml`: `db` (image `pgvector/pgvector:pg16`), `api`, `frontend`. Optionally `adminer` for local DB browsing.
- `make up`, `make ingest`, `make eval`, `make test` shortcuts.

## Deployment
- Option A: **Render** or **Railway**. Managed Postgres (enable the `vector` extension; on Railway use the pgvector template) plus two web services.
- Option B: **Fly.io**.
- Option C: a small VPS with docker-compose (cheapest, most control).
- Pre-ingest the pilot universe and load it with a seed/migration script. Never ingest from EDGAR on user request in production.
- Use a cheaper model for the public demo and allow a "high quality" toggle with stricter rate limits.

## CI/CD (GitHub Actions)
1. `ruff` + `mypy`
2. `pytest` (unit tests, network mocked)
3. Eval smoke test (10 questions) when relevant secrets are present
4. Build Docker image; deploy on merge to `main`

## Security checklist
- [x] No keys in the repo (`gitleaks` runs in CI)
- [x] Prompt-injection hygiene: filing text is treated as data, and every prompt says so
- [x] Input length limits on `/ask` (`API_MAX_QUESTION_CHARS`, plus schema limits on every input)
- [x] Spend cap + rate limit active in production (on by default)

## As built
- **API** (`src/filinglens/api/`): all endpoints above, plus `/ready` and `/filings/{acc}/section/{item}` for diff citations.
  - `services.py` holds a psycopg connection pool, a shared embedder and the provider choice.
  - Traces are stored in a `traces` table.
  - Answers are cached in process (TTL + LRU), keyed on the normalized question, the selectors and a config version (app, parser and prompt versions, provider, models, retrieval knobs).
  - Rate limiting is an in-process per-IP sliding window: standard, stricter for `quality: high`, and an "other" bucket. `X-Forwarded-For` is honoured only with `API_TRUST_PROXY`.
  - When the spend cap is reached, answers fall back to the free mode and are flagged `degraded`.
  - Logs are JSON, with request and trace ids on every line; both ids are also returned as response headers.
  - 404, 422 and 500 responses are clean; 500s don't leak exception text.
- **Free by default:** `LLM_PROVIDER=auto` picks Claude, else Ollama, else none. With Claude, `quality: standard` uses `DEMO_LLM_MODEL` (Haiku 5.5) and `high` uses `LLM_MODEL` (Sonnet 5.5).
- **Embeddings:** fastembed (ONNX) is the default backend. It gives vectors identical to sentence-transformers (cosine 1.0000 in a side-by-side check) with no PyTorch, so the API image is light and needs about 400 MB of RAM.
- **Frontend** (`frontend/app.py`):
  - Company and year selectors, a mode choice (agent or search + answer) and a quality toggle.
  - Example questions generated from the ingested companies.
  - Answers stream tool calls over SSE. Citations open the source passage with the quote highlighted and a sec.gov link. A "How I got this answer" panel shows the steps.
  - "What changed?" tab with similarity metrics and added/removed headings side by side; Financials tab with a table, a margins chart, formulas and sources.
- **Docker:** a multi-stage, slim, non-root API image with the model baked in, a healthcheck, and an entrypoint that seeds from `SEED_URL`. The frontend image has a healthcheck. Compose mounts `./data` and can reach Ollama on the host.
- **Seeding:** `filinglens seed-export` / `seed-import` (with `--url` and `--if-empty`) move the tables and parsed sections as COPY text, so the host doesn't need `pg_dump`. The CI eval smoke workflow uses the same bundle.
- **Deploy:** `render.yaml` (Option A), or `deploy/docker-compose.prod.yml` + Caddy for a VPS (Option C); see `deploy/README.md`. CI builds both images on every PR and triggers the deploy hook on merge to `main` once the secret exists.

## Acceptance criteria
- [ ] `docker compose up` gives a working local app from a clean clone (following the README). Images not yet built locally: the dev machine was out of disk.
- [ ] Public URL live with the pilot companies. Needs a hosting account.
- [x] Load test: 20 concurrent `/ask` requests without errors. `python -m loadtest`, run locally in free mode on the fixture corpus: 60/60 OK, p95 0.20 s. Re-run it against the deployment.
