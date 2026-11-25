# 01 — Data Ingestion (SEC EDGAR)

## Goal
Download and cache 10-K and 10-Q filings plus structured XBRL financial facts for a configurable universe of companies, while complying with SEC fair-access rules.

## What we know (research findings)

**APIs, free with no key**
- `data.sec.gov` hosts RESTful JSON APIs. **No authentication or API key** is needed.
- **Submissions (filing history per company):** `https://data.sec.gov/submissions/CIK##########.json`. The CIK must be **10 digits with leading zeros**. A missing zero-pad is the most common silent failure.
- **XBRL company facts (all financial concepts for a company):** `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`
- **Company concept (one concept over time):** `https://data.sec.gov/api/xbrl/companyconcept/CIK##########/{taxonomy}/{tag}.json` (e.g. `us-gaap/Revenues`)
- **Frames (one concept across all companies for a period):** `https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/CY{year}{Q#}{I}.json`. Useful for peer comparisons.
- **Bulk downloads, refreshed nightly:**
  - `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip`
  - `https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip`
  - **For the research phase (thousands of companies), use the bulk zips instead of per-company calls.**
- Filing documents live under `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/`.
- Ticker → CIK mapping: `https://www.sec.gov/files/company_tickers.json`
- EDGAR electronic filings start around **1994–1995**.

**Fair access (hard requirements)**
- **Max 10 requests/second in total**, regardless of how many machines you use.
- **Declare a User-Agent** with your name/organization and email, e.g. `User-Agent: FilingLens yourname@email.com`. Without it you'll get HTTP 403.
- Violations lead to 429/403 errors and temporary IP blocks (reported as 10 minutes to 24 hours).
- Browser JavaScript can't set User-Agent, so all EDGAR calls must go through the backend, never the frontend.

**Libraries**
- **`edgartools`** (`pip install edgartools`) is the recommended main library. It turns filings into typed Python objects (`TenK`, `TenQ`, `EightK`), extracts sections (business, risk factors, MD&A) and parses XBRL financials. Set identity with `set_identity("name email")` or the `EDGAR_IDENTITY` env var. It has built-in rate limiting (≤10 req/s) and retries on 429.
- `sec-edgar-downloader` is an alternative for bulk raw downloads.
- Use raw `httpx` for data.sec.gov JSON endpoints when you want full control.

## Spec

### Config (`.env`)
```
EDGAR_IDENTITY="Your Name your@email.com"
EDGAR_MAX_RPS=8            # stay safely under 10
DATA_DIR=./data
UNIVERSE_FILE=./config/pilot_universe.csv
START_YEAR=2015
FORMS=10-K,10-Q
```

### Modules
- `ingestion/client.py`: shared `httpx` client with User-Agent, gzip, **global token-bucket rate limiter**, retries with exponential backoff on 429/5xx, and an on-disk cache keyed by URL.
- `ingestion/universe.py`: load tickers, map them to zero-padded CIKs via `company_tickers.json`, and cache the mapping.
- `ingestion/filings.py`: for each CIK, list 10-K/10-Q filings since `START_YEAR` from the submissions API and download the primary document. Store metadata.
- `ingestion/facts.py`: download companyfacts JSON and flatten it to a long table.
- `ingestion/cli.py`: `filinglens ingest --universe pilot --forms 10-K,10-Q --since 2015`

### Filing metadata table (`filings`)
| column | type | note |
|---|---|---|
| accession_no | text PK | e.g. 0000320193-23-000106 |
| cik | text | zero-padded |
| ticker | text | as of ingestion |
| form | text | 10-K, 10-K/A, 10-Q |
| filing_date | date | |
| acceptance_datetime | timestamptz | **critical for no-lookahead research** |
| period_of_report | date | fiscal period end |
| primary_doc_url | text | |
| local_path | text | |
| is_amendment | bool | |

### Financial facts table (`financial_facts`)
`cik, taxonomy, concept, unit, value, period_start, period_end, fy, fp, form, accession_no, filed_date, frame`

Keep `filed_date`. The same period can be restated in later filings, so point-in-time queries need it.

## Pitfalls
- Zero-pad CIKs to 10 digits for data.sec.gov but **not** in Archive paths.
- Submissions JSON includes only recent filings inline. Older ones are in the extra `files` listed in the JSON, so follow them.
- Amendments (10-K/A) can be partial documents. Flag them and exclude them from the Lazy Prices comparison by default.
- `acceptanceDateTime` in the submissions JSON ends in `Z` but is **US Eastern clock time**. Parse it as `America/New_York` and store UTC, or every timestamp lands 4–5 hours early.
- Companies change tickers and names. **CIK is the stable ID.** Never key anything on ticker.
- XBRL concepts vary by company (`Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax`). Handle this with a concept-mapping table in `specs/04`.

## Acceptance criteria
- [ ] `ingest` for 20 pilot tickers × 5+ years completes, and re-running downloads nothing new.
- [ ] Rate never exceeds the configured RPS (unit test with a mock clock).
- [ ] `filings` and `financial_facts` populated; row counts logged.
- [ ] Every request carries the User-Agent (test asserts the header).
