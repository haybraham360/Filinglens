# References & Resources

## SEC EDGAR
- EDGAR APIs documentation: https://www.sec.gov/edgar/sec-api-documentation
- Accessing EDGAR data (fair access, User-Agent, rate limits): https://www.sec.gov/os/accessing-edgar-data
- Developer resources: https://www.sec.gov/developer
- Ticker → CIK mapping file: https://www.sec.gov/files/company_tickers.json
- Bulk company facts: https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
- Bulk submissions: https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip

## Libraries
- edgartools docs: https://edgartools.readthedocs.io/
- edgartools SEC compliance guide: https://edgartools.readthedocs.io/en/latest/resources/sec-compliance/
- edgartools on PyPI: https://pypi.org/project/edgartools/
- pgvector: https://github.com/pgvector/pgvector
- FastAPI: https://fastapi.tiangolo.com/

## Retrieval
- Supabase hybrid search guide (RRF in Postgres): https://supabase.com/docs/guides/ai/hybrid-search
- Railway hybrid search with pgvector: https://docs.railway.com/guides/hybrid-search-postgres-pgvector
- DigitalOcean hybrid query example: https://docs.digitalocean.com/products/vector-databases/postgresql/how-to/use-hybrid-search/

## Evals
- FinanceBench paper: https://arxiv.org/abs/2311.11944
- FinanceBench open-source sample (150 Qs): https://huggingface.co/datasets/PatronusAI/financebench

## Research
- Lazy Prices, NBER WP 25084: https://www.nber.org/papers/w25084
- Published version (J. Finance 2020): https://doi.org/10.1111/jofi.12885
- Internet appendix: https://admin.nber.org/data-appendix/w25084/w25084.appendix.pdf
- NBER Digest summary: https://www.nber.org/digest/nov18/are-investors-inattentive-sec-mandated-corporate-reports
- Kenneth French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- Loughran–McDonald Master Dictionary (Notre Dame SRAF): https://sraf.nd.edu/
- Related: Stanford CS224N student project applying vector representations to Lazy Prices: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2761951.pdf

## Notes
- Verify library APIs (especially edgartools section access) against current docs before coding. They change between versions.
- The SEC rate limit (10 req/s) and User-Agent requirement are hard rules. Violating them gets your IP blocked.
