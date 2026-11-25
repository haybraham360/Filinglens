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

## Acceptance criteria
- [ ] `compute_ratio` matches hand calculations exactly for 3 companies × 3 years.
- [ ] Tool-selection eval (`specs/05`): ≥90% of agent questions call the expected tool(s).
- [ ] No numeric answer is produced without a tool call (checked in evals).
