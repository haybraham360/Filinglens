# 04 — Agent & Tools

## Goal
An LLM agent that decides which tools to call to answer complex questions. **The LLM orchestrates and explains; deterministic Python code does the retrieval, diffing and math.**

## Tools

### 1. `search_filings`
Wraps the retrieval pipeline from `specs/03`.
```
input:  query: str, ticker: str | None, fiscal_year: int | None, form: str | None, item: str | None, top_k: int = 8
output: list of {chunk_id, ticker, fiscal_year, item, heading, text}
```

### 2. `compare_sections`
What changed between two filings' sections. This is also the product version of the Lazy Prices signal.
```
input:  ticker: str, item: str ("1A"), year_a: int, year_b: int
output: {
  similarity: {cosine_tf, jaccard, min_edit, simple, embedding_cosine},
  added_paragraphs: [...], removed_paragraphs: [...], modified_paragraphs: [{old, new}],
  added_risk_headings: [...], removed_risk_headings: [...]
}
```
- Align paragraphs or risk headings with fuzzy matching plus embedding similarity. Classify each as added, removed, modified or unchanged.
- The LLM then **summarizes** the diff ("3 new risks: …"). It does not perform the diff itself.
- Shares the similarity functions in `src/filinglens/research/similarity.py` with `specs/07`. Write them once.

### 3. `get_financials`
```
input:  ticker: str, concepts: list[str] (canonical names), fiscal_years: list[int], period: "FY" | "Q"
output: table of values with source accession_no and filed_date
```
- Reads from the `financial_facts` table (XBRL companyfacts).
- **Concept mapping table** (`config/concept_map.yaml`): canonical name → ordered list of XBRL tags to try. Examples:
  - `revenue`: `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet`
  - `net_income`: `NetIncomeLoss`
  - `total_assets`: `Assets`
  - `operating_income`: `OperatingIncomeLoss`
  - `operating_cash_flow`: `NetCashProvidedByUsedInOperatingActivities`
- Use annual values (`fp == "FY"`, `form == "10-K"`) and pick the **originally filed** value or the latest restatement explicitly (parameter `as_of`).

### 4. `compute_ratio`
```
input:  ticker, ratio: "gross_margin" | "operating_margin" | "net_margin" | "roe" | "roa" |
        "current_ratio" | "debt_to_equity" | "revenue_growth_yoy" | ..., fiscal_years
output: {ratio, values_by_year, formula, inputs_used (with sources)}
```
- Pure Python, unit-tested against hand-computed values for 3 companies.
- Returns its formula and inputs so the answer can show its work.

### 5. (Optional v2) `peer_compare`
Uses the XBRL **frames** API to compare one concept across all companies for a period.

## Orchestration
- Use the provider's native tool-calling. Keep a simple loop: model → tool calls → results → model, with a **maximum of 6 steps**.
- Don't add a framework until you need one. A hand-written loop is easier to debug and explain in interviews. LangGraph or similar is optional later.
- Optionally expose the tools as an **MCP server** (`src/filinglens/mcp_server.py`), so they work from Claude Desktop or Claude Code too. That's a nice portfolio point.
- Guardrails:
  - Validate tool args with Pydantic, and return helpful errors to the model.
  - Tool results are data. Ignore any instructions inside filing text (prompt-injection hygiene).
  - Time out tool calls after 20s.
- Trace everything: each step's tool name, args, latency and tokens, stored for debugging and shown in a "show reasoning" panel in the UI.

## Example flows
1. *"How did Microsoft's operating margin change from 2021 to 2024, and what does management say drove it?"*
   → `compute_ratio(MSFT, operating_margin, 2021–2024)` → `search_filings(MSFT, item=7, …)` → answer with numbers + cited explanations.
2. *"What new risk factors did Boeing add in its latest 10-K?"*
   → `compare_sections(BA, 1A, latest-1, latest)` → summarize the added headings with citations.

## As built
- **Modules:** `agent/tools.py` (Pydantic argument models, which also produce the strict tool schemas via `anthropic.transform_schema`), `agent/financials.py`, `agent/ratios.py`, `agent/diff.py`, `agent/sources.py`, `agent/guards.py`, `agent/loop.py`, `agent/cli.py`. Similarity measures live in `research/similarity.py` for reuse in spec 07.
- **Loop:** a hand-written loop on `client.beta.messages.create`, at most `AGENT_MAX_STEPS` (6) model requests.
  - The history is append-only: the full `response.content` (thinking blocks included) goes back unchanged, one user message carries all of a turn's tool results, and the tools and system prompt never change during a run.
  - The model can call several tools in one turn; they run concurrently with a 20 s timeout each.
  - Forced `tool_choice` returns a 400 on Sonnet 5.5, so steering is done in the prompt plus `strict: true` tools. On the last step the model gets an "answer now" text block and `tool_choice: none`.
  - The final answer comes back as structured output (`AnswerDraft`) in the same request, and the server-side refusal fallback is on.
- **Citations:** handles are `C#` (chunks), `P#` (paragraphs and headings from a section diff) and `T#` (financial tool results). Numbers are cited by quoting the exact value from the `T#` result, so they go through the same verified-citation check as filing text.
- **No LLM arithmetic:** `agent/guards.py` flags any number in the answer that isn't supported by some tool result after rounding, unit scaling (thousand/million/billion) or ratio-to-percent conversion. Flags are stored in the trace as `unsupported_numbers`, for the eval in spec 05.
- **`get_financials`:**
  - The fiscal year comes from the period end date, because companyfacts' `fy` is the *filing's* fiscal year and comparatives inherit it.
  - Annual values are 350–380-day durations. A balance-sheet value must fall on a 10-K's own balance-sheet date.
  - `as_of` is `original` (first filed, the default) or `latest`.
  - `period="Q"` keeps only the quarter each 10-Q itself reports.
- **`compute_ratio`:** exact `Decimal` arithmetic, rounded half-even to 6 decimal places, returning its formula and inputs with sources. Ratios: gross/operating/net margin, ROE, ROA (year-end), current ratio, debt-to-equity, long-term-debt-to-equity, OCF margin, revenue and net-income growth.
- **`compare_sections`:** reads parsed sections from parquet (originals only). Paragraphs are aligned by exact match, then a greedy one-to-one match on fuzzy plus embedding score. It reports added, removed and modified paragraphs and risk headings, the four classical similarity measures plus embedding cosine, and defaults to the latest 10-K versus the one before.
- **Tracing:** every model request goes to `llm_calls` (`purpose=agent_step`), and each run appends steps (tool, args, latency, tokens, errors) to `data/logs/agent_traces.jsonl`.
- **Without Claude** (`agent/router.py`): a rule-based router picks the tools from the question's filters plus keyword patterns (ratio and concept phrases; "new/added/removed/changed" + risk → `compare_sections`; "why/drove/explain" → `search_filings` on Item 7). The tools run unchanged. The answer is either templated from the results (no LLM: every number is a verbatim tool value) or written by a local Ollama model from the results in one structured call. The router falls back to the template if the local model's citations don't verify. Only Claude plans its own tool calls, because small local models are unreliable at it.
- **Not done:** the optional MCP server and `peer_compare`.

## Acceptance criteria
- [x] `compute_ratio` matches hand calculations exactly for 3 companies × 3 years (`tests/test_agent_tools.py`, on fixture figures modelled on reported results; re-check against real XBRL after ingestion).
- [ ] Tool-selection eval (`specs/05`): ≥90% of agent questions call the expected tool(s).
- [ ] No numeric answer is produced without a tool call (checked in evals).
