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
- [ ] No keys in the repo (run a secret scan, e.g. `gitleaks`, in CI)
- [ ] Prompt-injection hygiene: filing text is treated as data, and the system prompt says so
- [ ] Input length limits on `/ask`
- [ ] Spend cap + rate limit active in production

## Acceptance criteria
- [ ] `docker compose up` gives a working local app from a clean clone (following the README).
- [ ] Public URL live with the pilot companies.
- [ ] Load test: 20 concurrent `/ask` requests without errors (`locust` or `hey`).
