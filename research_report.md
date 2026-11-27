# Lazy Prices with LLMs: do year-over-year 10-K changes predict returns?

**Status: the method and code are complete and tested. No real-data results yet.** The tables in §6 are produced by `filinglens research backtest` once the real inputs exist (see §8). So far the pipeline has only been run on synthetic data with a planted effect (§5), which checks the code, not the hypothesis.

## 1. Question

Cohen, Malloy and Nguyen (2020, *Journal of Finance* 75(3), "Lazy Prices") show that firms which substantially change the language of their 10-K/10-Q filings earn lower future returns. A portfolio long "non-changers" and short "changers" earns up to 188 bps per month in their strongest specification. The returns accrue slowly over months, with no announcement effect.

We ask three things:

1. **Replication:** does the effect hold with our own EDGAR pipeline and the paper's four similarity measures?
2. **Decay:** does it survive *after* publication? The paper's sample ends in 2014, so 2015 onward is out of sample.
3. **Extension:** do embedding-based and structured change measures (new and removed risk factors, change in negative tone) add information beyond the classical measures?

## 2. Data

| Input | Source | Notes |
|---|---|---|
| Filings | SEC EDGAR, via `filinglens ingest` | Original 10-K (and optionally 10-Q). **Amendments excluded.** Each carries its acceptance timestamp (US Eastern clock) |
| Sections | `filinglens parse` | Items 1A (Risk Factors), 3 (Legal Proceedings), 7 (MD&A), and "full" (all extracted Items in order) |
| Returns | CRSP via WRDS if available, else Yahoo Finance | See survivorship (§7) |
| Market equity | CRSP `|PRC| x SHROUT`, or close x XBRL shares outstanding (point in time) | Used for value weights and size controls |
| Book equity | XBRL `StockholdersEquity`, 10-K values, usable the month after filing | Free stand-in for Compustat |
| Factors | Kenneth French Data Library (FF3, FF5, momentum); Pastor-Stambaugh traded liquidity | Monthly |
| Sentiment | Loughran-McDonald Master Dictionary | For `lm_neg_delta` |

## 3. Signals

Each filing is compared with the same firm's previous 10-K (for 10-Qs, the same fiscal quarter a year earlier). The "previous" filing must have a period end 300 to 430 days earlier, so missing years never create a fake pair. Lower similarity means more change. Definitions are in `src/filinglens/research/similarity.py`.

| Signal | Definition | Paper reference |
|---|---|---|
| `sim_cosine_tf` | Cosine of raw term-frequency vectors | p. 12, exact |
| `sim_jaccard` | Intersection over union of word *sets* | p. 13, exact |
| `sim_minedit` | `1 - (word insertions + deletions) / (len_a + len_b)` | p. 13. Substitutions cost 2, matching the paper's worked example. **Normalization is our choice**, because the paper doesn't state one |
| `sim_simple` | `1 - c`, with `c` = words in changed, added and deleted *lines* (a `diff`) / (size_a + size_b) | p. 14. The paper feature-scales to `(c_max - c)/c_max`, which is affine in `c`, so quintiles and Fama-MacBeth t-stats are identical |
| `sim_embed` | Cosine of mean-pooled paragraph embeddings (bge-small-en-v1.5) | New |
| `sim_embed_aligned` | Mean, over new paragraphs, of the best cosine match among old paragraphs | New |
| `n_new_risks`, `n_removed_risks` | Risk-factor headings added or removed (deterministic alignment, `compare_sections`) | New |
| `lm_neg_delta` | Change in the Loughran-McDonald negative-word share | New |
| `llm_change_severity` | LLM materiality rating (1-5) of anonymized Item 1A changes | New, guarded (§7) |

## 4. Method

**Formation** (`portfolios.py`):

- A filing released in calendar month *m* (US Eastern) gets its quintile from **breakpoints computed only from filings released in the previous calendar year**. That is the paper's Table II wording. The paper's text says "prior month's distribution", so a trailing-12-month variant is also run and reported.
- The stock **enters in month m+1** and is held for 3 months (the paper's main specification), with 6 and 12 months as robustness checks. Portfolios are rebalanced monthly. A newer filing replaces an older assignment, so each stock is in at most one quintile at a time.
- Q1 = changers (lowest similarity), Q5 = non-changers, L/S = Q5 - Q1.
- Equal and value weights. Value weights use market equity at the end of the previous month.

**Regressions** (`regressions.py`):

- Quintile excess returns and L/S returns are regressed on CAPM, FF3, the paper's "5-factor" model (FF3 + momentum + liquidity), and FF5 + momentum.
- **t-statistics use Newey-West** standard errors, with lags `max(floor(4(T/100)^(2/9)), hold - 1)`, because overlapping holding windows make monthly returns autocorrelated. Annualized Sharpe ratios are reported too.

**Robustness**, all pre-registered in `config/research.yaml`:

- Sections (full, 1A, 3, 7) and holding periods (3, 6, 12 months).
- Breakpoint rule.
- Size: terciles, and excluding micro-caps (below the 20th percentile of lagged market equity).
- Transaction costs of 10 and 25 bps per unit of weight traded, applied to measured turnover.
- **Pre-2015 vs post-2014.**

**Fama-MacBeth:**

- Monthly cross-sections of returns on each signal, with controls: log lagged market equity, log B/M, the prior month's return, and the t-12 to t-2 return. The paper also uses SUE, which needs Compustat and is omitted.
- Signals are winsorized at 1/99 within each month.
- An "incremental" specification enters the classical and new signals together.

## 5. Pipeline validation (synthetic data, not results)

`filinglens research backtest --synthetic` simulates 400 firms over 2006-2021, with a planted drift of 0.4% per month per standard deviation of similarity, lasting 6 months after each filing. The tests check:

- **The effect is recovered:** EW L/S FF3 alpha of about 105 bps/month (t ≈ 6). Expected: about 0.4% x 2.7 s.d. between the Q5 and Q1 means.
- **Release-month returns aren't captured:** a large return planted in the *release month itself* produces no L/S alpha, because entry is the month after.
- **Breakpoints ignore current-year data:** extreme values in the current year leave other firms' quintiles unchanged.
- **Value weights use the previous month's market equity, not the current month's.**
- **Point-in-time fundamentals:** XBRL values are usable only from the month after filing.
- **Delisting returns:** CRSP delisting returns are compounded into the last month.

## 6. Results

*Not run on real data yet.* `filinglens research backtest` writes `reports/research/latest/results.md`, with these tables:

1. The main quintile table per signal (EW and VW): excess return and alphas, Q1 to Q5 and L/S, with NW t-stats.
2. L/S alpha for the full sample, pre-2015 and post-2014.
3. Robustness by family.
4. Fama-MacBeth.
5. A multiple-testing summary, plus `variants.csv` with every test run.
6. `cumulative_ls.png`: cumulative L/S with the 2015 split marked.

## 7. Biases and safeguards

| Risk | How it is handled |
|---|---|
| Lookahead | Signals use only the two filings themselves. Release month comes from the acceptance timestamp on EDGAR's Eastern clock. Entry is the month after. Breakpoints use only earlier filings. Fundamentals are lagged to the month after filing. Tests cover each rule |
| Breakpoints | From the prior calendar year (or the trailing 12 months), never the formation month |
| Survivorship | With Yahoo prices, delisted firms are missing. The paper finds changers are more likely to go bankrupt, so the missing firms would probably have had very negative Q1 returns. The short leg, and the L/S, is probably **understated**. The report prints this caveat whenever the source isn't CRSP. CRSP runs include delisting returns |
| Tuning on the test period | The variant grid is fixed in `config/research.yaml` before any results. Post-2014 is only reported, never optimized |
| Amendments | 10-K/A excluded, and re-filed originals de-duplicated by period |
| Multiple testing | Every variant x period x model is logged and published, including failures. The report gives the number of tests and the number expected to pass by chance |
| **LLM pretraining leakage** | An LLM may know a company's later fate. The embedding signals are the main new measures, because they're purely textual. `llm_change_severity` masks company names, tickers and people, sends only changed Item 1A passages, and **skips filings before the model's training cutoff**. If those filings are included anyway, they are flagged `in_sample` and kept out of the main results. Anonymization is best-effort: products, places and numbers can still identify a firm |

## 8. How to produce the results

```bash
# 1. Filings (a broad universe: firms with consecutive 10-Ks, e.g. 2008-2024)
uv run filinglens ingest --universe <file> --forms 10-K --since 2008
uv run filinglens parse
# 2. Signals (add --embed for embedding similarity, --lm-dictionary for tone)
uv run filinglens research signals --lm-dictionary LoughranMcDonald_MasterDictionary.csv
# 3. Returns: CRSP if WRDS access is available, otherwise yfinance (biased)
uv run filinglens research prices --source crsp --crsp-file msf.csv --link-file cik_permno.csv
# 4. Factors (Pastor-Stambaugh file URL from Lubos Pastor's website)
uv run filinglens research factors --liquidity <liq_data_url>
# 5. Everything else
uv run filinglens research backtest
```

## 9. Limitations

- **Sample size.** Quintile sorts need hundreds of firms per month. The 20-company pilot universe is far too small, and a run on it would be noise.
- **Section extraction** relies on our Item parser (specs/02). The paper's "full document" removes tables and exhibits from the whole filing. Ours joins the extracted Items, which is close but not identical.
- **Book equity** comes from XBRL, available from 2009 onward, without Compustat's adjustments.
- **Liquidity factor:** we use the Pastor-Stambaugh traded factor. The paper doesn't say which liquidity measure it used.
- **Not yet checked against live downloads:** the factor-file parsers follow the published layouts but couldn't be tested against fresh downloads from the development machine. Check the first `filinglens research factors` run.
- **The `sim_minedit` normalization** isn't specified in the paper (§3).

## Reference

Cohen, L., Malloy, C. & Nguyen, Q. (2020). Lazy Prices. *Journal of Finance* 75(3), 1371-1415. NBER Working Paper 25084.
