.PHONY: setup up down db api frontend landing loadtest prod-up ingest parse embed eval eval-free eval-offline experiments research-synthetic research seed-export test-db test lint typecheck check

setup:        ## Install Python deps (all extras) and landing page deps
	uv sync --all-extras
	npm --prefix frontend/landing install

up:           ## Start db, api and frontend in Docker
	docker compose up --build

down:
	docker compose down

db:           ## Start only Postgres + pgvector
	docker compose up -d db

api:          ## Run the API locally with reload
	uv run uvicorn filinglens.api.main:app --reload

frontend:     ## Run the Streamlit app locally
	uv run streamlit run frontend/app.py

API_URL ?= http://localhost:8000

loadtest:     ## 20 concurrent /ask requests against a running API (raise its rate limit first)
	uv run python -m loadtest --url $(API_URL) --concurrency 20 --requests 60

prod-up:      ## VPS: compose + Caddy HTTPS (needs DOMAIN and POSTGRES_PASSWORD)
	docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build

landing:      ## Run the landing page dev server
	npm --prefix frontend/landing run dev

ingest:
	uv run filinglens ingest --universe pilot --forms 10-K,10-Q --since 2015

parse:
	uv run filinglens parse

embed:        ## Embed chunks into pgvector
	uv run filinglens embed

eval:         ## RAG eval on the dev split (paid: Sonnet answers + Opus judge)
	uv run python -m evals.run --dataset golden_v1 --split dev --config evals/configs/hybrid.yaml

eval-free:    ## Free evals: extractive answers + rule-based agent (no LLM, no judge)
	uv run python -m evals.run --dataset golden_v1 --split dev --config evals/configs/free/extractive.yaml
	uv run python -m evals.run --dataset golden_v1 --split dev --config evals/configs/free/router.yaml

eval-offline: ## Harness checks with oracle and null systems (free)
	uv run python -m evals.run --dataset golden_v1 --split dev --system oracle --no-judge
	uv run python -m evals.run --dataset golden_v1 --split dev --system null --no-judge

experiments:  ## Retrieval experiments E1-E6 on the test split (free: no LLM calls)
	uv run python -m evals.experiments --split test

research-synthetic: ## Lazy Prices pipeline check on simulated data (free, ~30 s)
	uv run filinglens research backtest --synthetic

research:     ## Lazy Prices on real data: signals -> prices -> factors -> backtest (see research_report.md)
	uv run filinglens research signals
	uv run filinglens research prices
	uv run filinglens research factors
	uv run filinglens research backtest

seed-export:  ## Bundle the ingested corpus for a deployment or the CI smoke eval
	uv run filinglens seed-export --out seed.tar.gz

test:
	uv run pytest

test-db:      ## Include Postgres tests (uses the filinglens_test database)
	docker compose exec -T db psql -U filinglens -tc "SELECT 1 FROM pg_database WHERE datname='filinglens_test'" | grep -q 1 || docker compose exec -T db psql -U filinglens -c 'CREATE DATABASE filinglens_test'
	FILINGLENS_TEST_DATABASE_URL=postgresql://filinglens:filinglens@localhost:5433/filinglens_test uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy src evals

check: lint typecheck test
