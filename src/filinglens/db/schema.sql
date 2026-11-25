-- FilingLens schema (specs/01, specs/02). Applied automatically by docker-compose on first start.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS filings (
    accession_no         text PRIMARY KEY,
    cik                  text NOT NULL,          -- zero-padded to 10 digits
    ticker               text,                   -- as of ingestion; never a key
    form                 text NOT NULL,          -- 10-K, 10-K/A, 10-Q
    filing_date          date NOT NULL,
    acceptance_datetime  timestamptz,            -- critical for no-lookahead research
    period_of_report     date,
    primary_doc_url      text,
    local_path           text,
    is_amendment         boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS filings_cik_form_idx ON filings (cik, form, period_of_report);

CREATE TABLE IF NOT EXISTS financial_facts (
    id            bigserial PRIMARY KEY,
    cik           text NOT NULL,
    taxonomy      text NOT NULL,
    concept       text NOT NULL,
    unit          text NOT NULL,
    value         numeric NOT NULL,
    period_start  date,
    period_end    date NOT NULL,
    fy            int,
    fp            text,
    form          text,
    accession_no  text,
    filed_date    date NOT NULL,                 -- keep: restatements need point-in-time queries
    frame         text
);
CREATE INDEX IF NOT EXISTS facts_lookup_idx ON financial_facts (cik, concept, period_end, filed_date);

-- Embedding dimension must match EMBEDDING_DIM (384 for bge-small-en-v1.5).
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         text PRIMARY KEY,           -- accession + item + index
    accession_no     text NOT NULL REFERENCES filings (accession_no),
    cik              text NOT NULL,
    ticker           text,
    form             text NOT NULL,
    fiscal_year      int,
    fiscal_period    text,
    item             text NOT NULL,
    heading          text,
    text             text NOT NULL,
    token_count      int,
    char_start       int,
    char_end         int,
    embedding        vector(384),
    embedding_model  text,
    tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(heading, '') || ' ' || text)) STORED
);
-- Added in specs/02: the "[AAPL | 10-K | FY2023 | Item 1A ...]" header prepended before embedding.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context_header text;
CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (cik, fiscal_year, form, item);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS llm_calls (
    id             bigserial PRIMARY KEY,
    created_at     timestamptz NOT NULL DEFAULT now(),
    trace_id       text,
    model          text NOT NULL,
    input_tokens   int,
    output_tokens  int,
    latency_ms     int,
    cost_usd       numeric(12, 6)
);
-- Added in specs/03.
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS purpose text;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS prompt_version text;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS stop_reason text;
CREATE INDEX IF NOT EXISTS llm_calls_created_idx ON llm_calls (created_at);
