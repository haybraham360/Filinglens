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
1. **First choice:** use `edgartools` section access. Its docs show `filing.obj()` returning a `TenK`, with properties like `business_description` and section names such as `business`, `risk_factors` and `management_discussion`, plus item-style access. *Verify the exact current API in the edgartools docs before coding.*
2. **Fallback:** a regex/HTML item splitter for filings where the library fails. Match `Item 1A.` style headings, skip table-of-contents matches (take the *last* occurrence of each heading before the next item, or use heading-tag heuristics).
3. **Clean:** strip HTML, remove page headers/footers and page numbers, normalize whitespace and unicode, drop XBRL inline tags, and keep tables as markdown where feasible.
4. **Log extraction quality:** section found (y/n), character length, and a flag for suspiciously short sections (<500 chars for 1A in a 10-K usually means a parse failure, or a smaller reporting company that omits it).

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
- `data/processed/sections/{cik}/{accession}.parquet`: one row per Item (full text). **Used by research.**
- `data/processed/chunks/*.parquet`, then loaded into Postgres.

## Acceptance criteria
- [ ] Item 1A and Item 7 extracted for ≥95% of pilot 10-Ks (report coverage table).
- [ ] Manual spot-check of 10 random filings: no ToC fragments, no page-number junk.
- [ ] Re-running the parser is deterministic (same chunk IDs).
- [ ] Unit tests on 3 saved fixture filings with known item boundaries.
