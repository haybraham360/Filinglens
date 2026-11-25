# 02 — Parsing & Chunking

## Goal
Turn raw filing HTML into clean, sectioned text (by Item) and retrieval-ready chunks with rich metadata. The same sectioned text also feeds the Lazy Prices research.

## Key 10-K / 10-Q sections
| Form | Item | Name | Why we care |
|---|---|---|---|
| 10-K | 1 | Business | company overview |
| 10-K | 1A | Risk Factors | **most informative for Lazy Prices**; Q&A on risks |
| 10-K | 3 | Legal Proceedings | litigation changes are informative (per the paper) |
| 10-K | 7 | MD&A | performance drivers |
| 10-K | 7A | Market Risk | quant-relevant |
| 10-K | 8 | Financial Statements | numbers come from XBRL instead |
| 10-Q | Part I Item 2 | MD&A | |
| 10-Q | Part II Item 1A | Risk Factors (updates) | |

## Approach
> **As built:** the order of steps 1 and 2 is reversed. Testing edgartools 6.4 showed that it finds item boundaries well, but it drops `<b>` inside `<font>` tags in legacy filings, which loses the risk-factor headings. It also keeps running footers and escapes its markdown output (`U\.S\.`), which breaks verbatim citation quotes. So the native splitter is primary, and edgartools cross-checks every filing and fills in any *required* item (10-K 1A/7, 10-Q Part I Item 2) that the native splitter misses.

1. **Native splitter (`parsing/blocks.py`, `parsing/sections.py`):** lxml turns HTML into ordered heading/paragraph/table blocks with emphasis detection (bold/italic/underline, `font-weight` overrides, all-caps). ToC tables (≥3 rows starting with "Item N") are dropped. An item heading is a short heading-like block matching `Item 1A.`. Each item takes its *last* occurrence, and the longest run of items in canonical order is kept.
2. **edgartools cross-check:** `HTMLParser(ParserConfig(form=...)).parse(html)` runs offline. Its per-item lengths go in the quality report (the `agrees` column is true when the ratio is within 0.8–1.25).
3. **Clean (`parsing/clean.py`, `parsing/text.py`):** drops hidden XBRL (`ix:header`, `display:none`), page numbers (`12`, `- 5 -`, `ii`, `F-3`), "Table of Contents" back-links, and running headers/footers. A footer is detected document-wide as a short line that repeats with changing digits, or one that names the form. Text is normalized with NFKC, ASCII quotes, Windows-1252 control characters mapped, and whitespace collapsed. Tables become markdown, with split `$ | 3,105` and `(3 | )%` cells joined.
4. **Quality log:** `data/processed/reports/parse_quality.csv` and `parse_coverage.md` record found, chars, source, short and edgartools agreement for 10-K items 1, 1A, 3, 7, 7A and 10-Q items I-2, II-1A. Originals only; amendments are excluded from coverage.

**Item keys:** 10-K items are bare codes (`1A`, `7`). 10-Q item numbers repeat across parts, so 10-Q keys carry the part (`I-2` = MD&A, `II-1A` = Risk Factors).

**Fiscal year/period:** taken from inline XBRL `dei:DocumentFiscalYearFocus`/`FiscalPeriodFocus`. If that's missing, the companyfacts `fy`/`fp` for the accession is used, then the calendar year of the period of report.

## Chunking strategy
- **Structure-aware:** never cross Item boundaries. Split by sub-headings first, then paragraphs.
- Target **~400–800 tokens per chunk with ~10–15% overlap**. Make these values config parameters, because evals in `specs/05` will choose the final numbers.
- Risk Factors are usually a list of bolded risk headings. Try **one chunk per risk factor** (heading + body) and compare this against fixed-size chunking in evals.
- Prepend a short context header to each chunk before embedding:
  `"[AAPL | 10-K | FY2023 | Item 1A Risk Factors | <risk heading>]"`. This usually improves retrieval for company/year-specific questions.

## Chunk schema (`chunks`)
| column | type |
|---|---|
| chunk_id | text PK (accession + item + index) |
| accession_no | text FK |
| cik, ticker, form | text |
| fiscal_year, fiscal_period | int, text |
| item | text (e.g. "1A") |
| heading | text (nearest sub-heading) |
| text | text |
| token_count | int |
| char_start, char_end | int (offset in section, for citation highlighting) |
| embedding | vector(n) |
| embedding_model | text |
| tsv | tsvector (generated column for full-text search) |

## Outputs
- `data/processed/sections/{cik}/{accession}.parquet`: one row per Item (full text plus block spans). **Used by research.** It's cached per `PARSER_VERSION`, with a `.quality.json` sidecar.
- `data/processed/chunks/{cik}/{accession}.parquet`, then loaded into Postgres. A filing's chunks are rewritten only if they changed, so embeddings survive re-runs. Chunks are always regenerated from cached sections, so a chunk-size sweep never re-parses HTML.
- Chunked items by default: 10-K `1, 1A, 1C, 3, 7, 7A`; 10-Q `I-2, I-3, II-1, II-1A`. Item 8 is left out because numbers come from XBRL.
- Chunk defaults: `CHUNK_MAX_TOKENS=448`, `CHUNK_OVERLAP_TOKENS=48`, `CHUNK_MIN_TOKENS=64`, `CHUNK_STRATEGY=structure|fixed`. 448 sits within the spec's 400–800 range and leaves headroom under bge-small's 512-token limit for the context header. Tokens are counted approximately (words plus punctuation).
- `chunks.context_header` stores the `[AAPL | 10-K | FY2023 | Item 1A Risk Factors | <heading>]` prefix used at embedding time.

## Commands
```
filinglens parse [--tickers AAPL] [--force] [--strategy fixed] [--no-db] [--no-edgartools]
filinglens show-section <accession> 1A [--chunks]      # manual spot-checks
```

## Acceptance criteria
- [ ] Item 1A and Item 7 extracted for ≥95% of pilot 10-Ks (report coverage table).
- [ ] Manual spot-check of 10 random filings: no ToC fragments, no page-number junk.
- [x] Unit tests on 3 saved fixture filings with known item boundaries (`tests/fixtures/filings/`: modern iXBRL 10-K, legacy 2015 10-K, 10-Q). These are synthetic, built to mimic real filing quirks; add real pilot filings once ingested.
- [x] Re-running the parser is deterministic (tested).
