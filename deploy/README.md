# Deploying FilingLens

Production never calls SEC EDGAR on a user's request. Ingest, parse and embed on your machine, export a seed bundle, and load that on the server.

```bash
uv run filinglens ingest --tickers AAPL,MSFT,JPM --since 2022
uv run filinglens parse && uv run filinglens embed
make seed-export                      # -> seed.tar.gz (tables + parsed sections)
```

Upload `seed.tar.gz` somewhere the server can download it, such as a GitHub release asset. Set its URL as `SEED_URL` on the API. On boot the API re-extracts the section files every time, and loads the tables only when the database is empty.

## Cost and size

- **The API needs about 400 MB of RAM.** Embeddings run on ONNX (fastembed), with no PyTorch, and the model is baked into the image.
- **No LLM spend in the default mode.** The default is `LLM_PROVIDER=none`: extractive answers and rule-based tools. To use Claude, set `ANTHROPIC_API_KEY` and keep `LLM_DAILY_SPEND_CAP_USD` low. When the cap is hit, the API falls back to the free mode and flags the response `degraded`.
- **Check your host's current plans and limits before you start.** Free tiers change, and some sleep idle services or expire databases.

## Option V: Vercel + Neon (what the live site uses)

Two Vercel projects:

| Project | Root | What it is |
|---|---|---|
| `filinglens` | `frontend/landing` | Static landing page plus the app at `/app` (Vite). Build-time `VITE_API_URL` points at the API |
| `filinglens-api` | repo root | The FastAPI API as a container function (`Dockerfile.vercel`), with a Neon Postgres database from the Vercel Marketplace |

1. **Database:** `vercel install neon` in the API project. This sets `DATABASE_URL` (pooled). The API turns off server-side prepared statements, so it works through Neon's pooler.
2. **Data:** ingest, parse and embed locally, run `make seed-export`, then load the tables into Neon with `DATABASE_URL=<Neon URL> filinglens seed-import seed.tar.gz --replace`. Upload `seed.tar.gz` as the `seed-v1` GitHub release asset. `Dockerfile.vercel` bakes its section files into the image at build time.
3. **API settings** (production): `LLM_PROVIDER=none`, `API_CORS_ORIGINS=<landing URLs>`, `API_TRUST_PROXY=true` (Vercel sets `X-Forwarded-For`), `PORT=8000`, `DB_POOL_SIZE=3`.
4. **Deploy:** `vercel deploy --prod` at the repo root (API), then in `frontend/landing` with `VITE_API_URL` set to the API's URL.

Notes:
- **Limits are per instance.** The rate limiter and answer cache live in memory, so under heavy traffic Vercel's extra instances each keep their own.
- **New data:** after changing the corpus, re-export, re-import into Neon, upload a new release asset, and redeploy the API.

## Option A: Render (`render.yaml`)

1. New → Blueprint → this repository. This creates the API, the frontend and a Postgres database (with the `vector` extension, which the API enables itself).
2. In the dashboard, set `SEED_URL` on `filinglens-api`, and `FILINGLENS_API_URL` on `filinglens-frontend` to the API's public URL.
3. Optional: add each service's deploy hook as the `RENDER_DEPLOY_HOOK_URL` / `RENDER_DEPLOY_HOOK_URL_FRONTEND` repository secrets. CI then deploys on every merge to `main`, after lint, tests and the Docker builds pass.

## Option C: a small VPS with Docker Compose

```bash
git clone <repo> && cd filinglens
cp .env.example .env                  # LLM_PROVIDER=none (or a key), SEED_URL=...
export DOMAIN=filinglens.example.com POSTGRES_PASSWORD=$(openssl rand -hex 24)
docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build
```

Caddy gets HTTPS certificates automatically for `$DOMAIN` (the app) and `api.$DOMAIN` (the API). Point both DNS records at the server first. Only ports 80 and 443 are public, because the production override removes the database and service port mappings. `API_TRUST_PROXY=true` lets rate limiting see real client IPs through Caddy.

## Checks after deploying

```bash
curl https://api.$DOMAIN/ready          # {"status":"ok","database":true,...}
python -m loadtest --url https://api.$DOMAIN --concurrency 20 --requests 60
```

The load test sends distinct questions. Raise `API_RATE_LIMIT_PER_MINUTE` on the server while it runs, otherwise you're measuring the rate limiter.
