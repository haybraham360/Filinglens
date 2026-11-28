# FilingLens

**An AI assistant that reads company annual reports (SEC 10-K filings) and answers your questions, with a source for every answer.**

👉 **Try it live:** [filinglens-two.vercel.app](https://filinglens-two.vercel.app)

---

## Why this exists

Annual reports are long, often more than 100 pages. Finding one fact, or working out what changed since last year, can take hours.

FilingLens does that reading for you. You ask a question in plain English and get a short answer that links back to the exact part of the filing it came from, so you can check it yourself.

## What it can do

| Feature | Example question |
|---|---|
| **Answer questions with sources** | "What are Tesla's main supply chain risks?" |
| **Show what changed between years** | "How did Apple's risk factors change from 2023 to 2024?" |
| **Give real financial numbers** | "What was Microsoft's revenue growth in 2024?" |

**Numbers are never made up.** Figures such as revenue, margins and growth come straight from official SEC data and are calculated by code, not by the AI.

## Companies covered

Apple, Microsoft, NVIDIA, Amazon, JPMorgan Chase, Coca-Cola and Tesla: 49 annual reports from 2020 to 2026.

---

## How it works

```
 1. Download          2. Split up              3. Search              4. Answer
 ┌────────────┐      ┌──────────────┐       ┌───────────────┐      ┌──────────────────┐
 │ SEC EDGAR  │ ───► │ Break filings│ ───►  │ Find the most │ ───► │ Write a short    │
 │ filings +  │      │ into sections│       │ relevant parts│      │ answer + sources │
 │ financials │      │ and passages │       │ of the filings│      │ (numbers by code)│
 └────────────┘      └──────────────┘       └───────────────┘      └──────────────────┘
```

1. **Download.** Filings and financial data come free from the SEC's EDGAR website.
2. **Split up.** Each filing is cut into its sections (such as Risk Factors and Management's Discussion) and then into small passages.
3. **Search.** When you ask a question, it finds the most relevant passages using both meaning-based search and keyword search.
4. **Answer.** An assistant picks the right tool (search, financials, ratio calculator or year-over-year comparison) and writes an answer that cites its sources.

**It works without an AI key.** With no key, answers are made of exact sentences quoted from the filings. If you add a Claude API key, or run a local model with [Ollama](https://ollama.com), the answers are written in natural language instead.

## Built with

Python · FastAPI · PostgreSQL + pgvector · Claude API (optional) · Docker · Vercel + Neon (hosting)

---

## Run it on your own computer

**You need:** [Docker](https://www.docker.com/) and [uv](https://docs.astral.sh/uv/) (a Python package manager).

```bash
# 1. Set up your settings file
cp .env.example .env
#    Open .env and set EDGAR_IDENTITY="Your Name you@example.com"
#    (the SEC asks every user to identify themselves)

# 2. Start the database and install dependencies
make db
uv sync

# 3. Download, split up and index some filings
uv run filinglens ingest --tickers AAPL,MSFT,JPM --since 2022
uv run filinglens parse
uv run filinglens embed

# 4. Start everything
make up
```

Then open:
- **The app:** http://localhost:8501
- **The API docs:** http://localhost:8000/docs

**Optional extras**
- **Better answers with Claude:** add `ANTHROPIC_API_KEY` to your `.env` file.
- **Free local AI model:** `brew install ollama && ollama pull qwen2.5:3b` (about 2 GB).

**Something not working?** Run `uv run filinglens doctor` to check your setup.

## Useful commands

| Command | What it does |
|---|---|
| `uv run filinglens ask "your question"` | Ask a question from the terminal |
| `uv run filinglens agent "your question" --trace` | Ask, and see each step the assistant takes |
| `make up` / `make down` | Start or stop everything |
| `make check` | Run code checks and tests |
| `make eval-free` | Measure answer quality (free, no API key) |

---

## Research: do changes in reports predict stock prices?

A 2020 study called *Lazy Prices* (Cohen, Malloy & Nguyen) found that when companies quietly change their annual reports, their stock tends to do worse afterwards.

This project rebuilds that study and tests whether modern AI text comparison spots these changes better than older methods. It is careful to use only information that was public at the time (no "peeking into the future").

**Status:** the full pipeline is built and tested on simulated data. Results on real market data are not in yet. Details are in [`research_report.md`](research_report.md).

---

## Project layout

```
src/filinglens/   main code (download, parsing, search, AI, API, research)
frontend/         the website
evals/            tests that measure answer quality
specs/            design documents for each part
tests/            code tests
```

## For developers

- **API endpoints:** see the live docs at [filinglens-api-one.vercel.app/docs](https://filinglens-api-one.vercel.app/docs).
- **Measuring quality:** [`evals/README.md`](evals/README.md)
- **Deploying:** [`deploy/README.md`](deploy/README.md)
- **Full design:** [`CLAUDE.md`](CLAUDE.md) and the [`specs/`](specs/) folder

---

## Get in touch

I'm open to **collaboration, contract work and full-time roles** in AI engineering, fintech and quantitative research. Feel free to reach out on GitHub ([@haybraham360](https://github.com/haybraham360)) or LinkedIn.

---

*Data comes from SEC EDGAR. This is not investment advice.*
