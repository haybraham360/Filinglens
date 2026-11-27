"""Synthetic filings, returns and factors with a *known* effect, for checking the pipeline.

The output is for tests and for checking the pipeline, never a result. Firms file one
10-K a year. Returns load on a market factor, and in the months after a filing they drift
by ``effect * z``, where ``z`` is the filing's standardized similarity (low similarity, so
lower returns). ``filing_month_effect`` puts a return in the *release month itself*. A
backtest that enters on time can't earn it, so the lookahead tests use it as a trap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticWorld:
    """Signals (signals-table format), a returns panel and factors."""

    signals: pd.DataFrame
    panel: pd.DataFrame
    factors: pd.DataFrame


def make_world(
    *,
    n_firms: int = 400,
    start: str = "2006-01",
    end: str = "2021-12",
    effect: float = 0.004,
    effect_months: int = 6,
    filing_month_effect: float = 0.0,
    noise: float = 0.06,
    seed: int = 7,
) -> SyntheticWorld:
    """Build a world. ``effect`` is the monthly return per s.d. of similarity."""
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    n_m = len(months)
    factors = pd.DataFrame(
        {
            "MKT": rng.normal(0.006, 0.045, n_m),
            "SMB": rng.normal(0.001, 0.03, n_m),
            "HML": rng.normal(0.001, 0.03, n_m),
            "RMW": rng.normal(0.002, 0.02, n_m),
            "CMA": rng.normal(0.001, 0.02, n_m),
            "MOM": rng.normal(0.005, 0.04, n_m),
            "LIQ": rng.normal(0.003, 0.03, n_m),
            "RF": np.full(n_m, 0.001),
        },
        index=months,
    )
    firm_ids = [f"{i:010d}" for i in range(1, n_firms + 1)]
    beta = rng.normal(1.0, 0.3, n_firms)
    filing_month = rng.integers(1, 13, n_firms)  # each firm's usual release month

    sig_rows = []
    drift = np.zeros((n_firms, n_m))
    for f, cik in enumerate(firm_ids):
        for year in range(months[0].year, months[-1].year + 1):
            m = pd.Period(year=year, month=int(filing_month[f]), freq="M")
            if m < months[0] or m > months[-1]:
                continue
            sim = float(np.clip(rng.beta(8, 2), 0, 1))
            z = (sim - 0.8) / 0.12
            accept = pd.Timestamp(year=year, month=m.month, day=15, hour=20, tz="UTC")
            sig_rows.append(
                {
                    "cik": cik,
                    "ticker": f"T{f}",
                    "form": "10-K",
                    "accession_no": f"{cik}-{year}",
                    "section": "full",
                    "filing_month": m,
                    "acceptance_datetime": accept,
                    "sim_cosine_tf": sim,
                    "sim_jaccard": float(np.clip(sim + rng.normal(0, 0.03), 0, 1)),
                    "sim_embed": float(np.clip(sim + rng.normal(0, 0.05), 0, 1)),
                    "n_new_risks": float(rng.poisson(max(0.1, 6 * (1 - sim)))),
                }
            )
            i = int(months.get_loc(m))  # type: ignore[arg-type]
            drift[f, i] += filing_month_effect * z
            drift[f, i + 1 : i + 1 + effect_months] += effect * z

    eps = rng.normal(0, noise, (n_firms, n_m))
    rets = (
        factors["RF"].to_numpy() + beta[:, None] * (factors["MKT"].to_numpy() - 0.0) + drift + eps
    )
    me0 = rng.lognormal(21, 1.5, n_firms)
    me = me0[:, None] * np.cumprod(1 + rets, axis=1)
    panel = pd.DataFrame(
        {
            "firm_id": np.repeat(firm_ids, n_m),
            "month": np.tile(months, n_firms),
            "ret": rets.ravel(),
            "me": me.ravel(),
        }
    )
    panel["month"] = pd.PeriodIndex(panel["month"], freq="M")
    panel.attrs["source"] = "synthetic"
    signals = pd.DataFrame(sig_rows)
    signals["filing_month"] = pd.PeriodIndex(signals["filing_month"], freq="M")
    return SyntheticWorld(signals=signals, panel=panel, factors=factors)
