# 07 — Research: "Lazy Prices" with LLMs

## Research question
Do year-over-year changes in 10-K text predict future stock returns? And do **embedding/LLM-based** change measures add information beyond the classical similarity measures used in the original paper?

## The original paper (what to replicate)
Cohen, L., Malloy, C. & Nguyen, Q. (2020). *Lazy Prices*. **Journal of Finance 75(3), 1371–1415.** NBER Working Paper 25084. SSRN 1658471.

Key findings (paraphrased):
- **Sample:** all quarterly and annual filings (10-K, 10-Q) of US firms, **1995–2014**.
- When firms actively change their report language, that signals future operations. Changes **predict future earnings, profitability, news announcements and even bankruptcies**.
- A portfolio **long "non-changers", short "changers"** earned up to **188 bps/month alpha (over 22%/yr)** in the strongest specification. Value-weighted abnormal returns were about **34–58 bps/month (≈4–7%/yr)**.
- **No announcement effect**: returns accrue gradually, for **up to 18 months**, and are **not reversed**. They weren't explained by size, time, industry or one-time firm events.
- Changes in **Risk Factors**, **litigation**, and language about the **CEO/CFO team** are especially informative.
- **Four similarity measures:** cosine similarity (term frequencies), Jaccard, minimum edit distance, and simple similarity (word-level diff). All give similar results.
- **Portfolio construction:** quintiles based on the **prior year's distribution** of similarity scores across all stocks. A stock **enters its quintile portfolio in the month after the filing's public release**.
- 10-Ks were more than 6× longer in 2014 than in 1995, which supports the investor-inattention story.

Read the paper and its internet appendix (NBER) before coding. Match their definitions exactly for the replication part.

## Our extension
| Signal | Type | Notes |
|---|---|---|
| `sim_cosine_tf` | classical | replicate |
| `sim_jaccard` | classical | replicate |
| `sim_minedit` | classical | replicate (expensive; compute on Item 1A + sample) |
| `sim_simple` | classical | replicate |
| `sim_embed` | **new** | mean / paragraph-aligned cosine of embeddings |
| `n_new_risks`, `n_removed_risks` | **new** | from `compare_sections` risk-heading alignment |
| `llm_change_severity` | **new, careful** | LLM rates the materiality of changes (see leakage below) |
| `lm_neg_delta` | **new** | change in Loughran–McDonald negative-word share |

Sections: full document, Item 1A, Item 3, Item 7, each separately.

## Data
- **Text:** EDGAR 10-Ks (bulk via edgartools; see `specs/01`). Compare each 10-K with the **same firm's previous 10-K**.
- **Universe:** all firms with consecutive 10-Ks. Start with a manageable subset (e.g. ~1,000 firms × 2010–2024), then scale.
- **Prices (major constraint):** free sources like `yfinance` mostly don't include delisted stocks, so a free-data universe carries **survivorship bias**. Delisted firms are often the "changers" that went bankrupt, so this bias likely **understates** the short-leg returns. Survivorship-free data (CRSP, Norgate) is paid.
  - **Action:** ask WorldQuant University whether students can get access to WRDS/CRSP or similar data through the program. If not, use free data, state the bias clearly and discuss its likely direction.
  - Use adjusted close for returns, and pull monthly returns.
- **Factors:** Kenneth French Data Library (free, no key). Fama/French 3 and 5 factors plus momentum, monthly. Updated regularly. `pandas-datareader` can download them, or use the CSVs directly.
- **Loughran–McDonald dictionary:** free from the Notre Dame SRAF site (see REFERENCES).

## Methodology
1. For each firm-year 10-K: compute the signals vs the prior 10-K. Timestamp each one with the **acceptance datetime**.
2. Each month *t*: take firms whose 10-K was released in month *t−1*. Assign quintiles by the **prior calendar year's** cross-sectional distribution of the signal.
3. Hold each firm in its quintile for 3, 6 and 12 months (report all), rebalancing monthly with overlapping cohorts.
4. Compute equal- and value-weighted returns per quintile, plus the long–short (Q5 non-changers − Q1 changers).
5. Regress L/S excess returns on FF3, FF5 and FF5+momentum. Report alpha, t-stats with **Newey–West** standard errors, and Sharpe ratio.
6. **Robustness:** by section (1A vs full), by size tercile, pre/post 2014 (**out-of-sample relative to the paper**: the key test of whether the anomaly decayed after publication), excluding micro-caps, and with transaction-cost assumptions.
7. **Incremental value:** Fama–MacBeth regressions of next-month returns on classical + new signals together, with controls (size, B/M, momentum).

## ⚠️ LLM pretraining leakage
An LLM may *know* what happened to a company after the filing (e.g. bankruptcy), which would contaminate `llm_change_severity`.
- Use **embedding similarity** signals (purely textual) as the main new signal. Embeddings are much less able to "leak" outcomes.
- For LLM-judgment signals: **anonymize** company names, tickers and executive names, and **restrict the evaluation period to after the model's training cutoff**, or treat pre-cutoff results as in-sample only.
- Discuss this in the write-up. It's a sophisticated point reviewers will appreciate.

## Other bias checklist
- [ ] Lookahead: signal uses only data up to the acceptance timestamp; entry in the following month
- [ ] Quintile breakpoints from the prior year only
- [ ] Survivorship bias disclosed, with its direction discussed
- [ ] No parameter tuning on the post-2014 test period
- [ ] Amendments (10-K/A) excluded
- [ ] Multiple-testing awareness: report all variants tried, not only the best

## Deliverables
- `src/filinglens/research/similarity.py` (shared with the product)
- `notebooks/research/01_signals.ipynb`, `02_portfolios.ipynb`, `03_regressions.ipynb`
- `research_report.md`: question, data, method, results tables, limitations, conclusion. This can become the capstone write-up.
- A chart of cumulative L/S returns, with a pre/post-2014 split marker.
