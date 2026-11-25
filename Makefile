.PHONY: setup up down db api frontend landing ingest parse eval test lint typecheck check

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

landing:      ## Run the landing page dev server
	npm --prefix frontend/landing run dev

ingest:
	uv run filinglens ingest --universe pilot --forms 10-K,10-Q --since 2015

parse:
	uv run filinglens parse

eval:
	uv run python -m evals.run --dataset golden_v1 --split dev

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy src

check: lint typecheck test
