"""Calendar-time quintile portfolios on a change signal (specs/07, steps 2-4).

Timing (Cohen, Malloy & Nguyen 2020, p. 16 and Table II):

* A filing released in month *m* (US Eastern) is sorted with **breakpoints from earlier
  filings only**. The default is the previous calendar year's cross-section ("prior year's
  distribution", Table II). The paper's text also says "prior month's distribution", so
  ``trailing_12m`` and ``prior_month`` are available as robustness variants.
* The stock **enters in month m+1** and is held for ``hold_months`` (paper: 3), with
  monthly rebalancing. If a newer filing of the same firm arrives while it is held, the
  newer assignment replaces the older one, so a stock is in at most one quintile at a time.
* Q1 = lowest similarity ("changers"), Qn = highest ("non-changers"); L/S = Qn - Q1.
* Equal or value weights. Value weights use market equity at the end of the previous
  month, so the weight is known before the return.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

Breakpoints = Literal["prior_year", "trailing_12m", "prior_month"]
Weighting = Literal["ew", "vw"]


@dataclass(frozen=True)
class PortfolioSpec:
    """One portfolio variant. Every field is reported, so variants stay auditable."""

    signal: str
    section: str = "full"
    n_quantiles: int = 5
    hold_months: int = 3
    breakpoints: Breakpoints = "prior_year"
    weighting: Weighting = "ew"
    min_breakpoint_obs: int = 25
    exclude_microcaps: bool = False  # drop stocks below the 20th percentile of lagged ME
    size_tercile: int | None = None  # 1 = small ... 3 = large (lagged ME, each month)
    cost_bps: float = 0.0  # one-way cost per unit of weight traded

    def label(self) -> str:
        """Short, unique description used in tables and the variants log."""
        parts = [
            self.signal,
            self.section,
            f"h{self.hold_months}",
            self.weighting,
            self.breakpoints,
        ]
        if self.exclude_microcaps:
            parts.append("no-micro")
        if self.size_tercile:
            parts.append(f"size{self.size_tercile}")
        if self.cost_bps:
            parts.append(f"tc{self.cost_bps:g}")
        return "/".join(parts)

    def with_(self, **kw: object) -> PortfolioSpec:
        """Copy with fields changed."""
        return replace(self, **kw)  # type: ignore[arg-type]


def _reference_mask(months: pd.Series, m: pd.Period, rule: Breakpoints) -> pd.Series:
    mask: pd.Series
    if rule == "prior_year":
        mask = months.dt.year == m.year - 1
    elif rule == "trailing_12m":
        mask = (months >= m - 12) & (months <= m - 1)
    else:
        mask = months == m - 1
    return mask


def assign_quantiles(signals: pd.DataFrame, spec: PortfolioSpec) -> pd.DataFrame:
    """Quantile per filing: ``firm_id, filing_month, acceptance_datetime, value, q``.

    Breakpoints for month *m* use only filings released before *m* (see module doc).
    Filings with too few reference observations are left unassigned.
    """
    df = signals[signals["section"] == spec.section][
        ["cik", "filing_month", "acceptance_datetime", spec.signal]
    ].dropna(subset=[spec.signal])
    df = df.rename(columns={"cik": "firm_id", spec.signal: "value"}).reset_index(drop=True)
    if df.empty:
        return df.assign(q=pd.Series(dtype="Int64"))
    probs = np.arange(1, spec.n_quantiles) / spec.n_quantiles
    q = pd.Series(pd.NA, index=df.index, dtype="Int64")
    months = df["filing_month"]
    for m in sorted(months.unique()):
        ref = df.loc[_reference_mask(months, m, spec.breakpoints), "value"].to_numpy()
        if len(ref) < spec.min_breakpoint_obs:
            continue
        bps = np.quantile(ref, probs)
        here = months == m
        q[here] = 1 + np.searchsorted(bps, df.loc[here, "value"].to_numpy(), side="right")
    return df.assign(q=q).dropna(subset=["q"])


def holdings(assigned: pd.DataFrame, hold_months: int) -> pd.DataFrame:
    """Firm-months held: ``firm_id, month, q, age`` (age 1 = the month after release)."""
    if assigned.empty:
        return pd.DataFrame(columns=["firm_id", "month", "q", "age"])
    parts = []
    for age in range(1, hold_months + 1):
        parts.append(
            assigned.assign(month=assigned["filing_month"] + age, age=age)[
                ["firm_id", "month", "q", "age", "filing_month", "acceptance_datetime"]
            ]
        )
    held = pd.concat(parts, ignore_index=True)
    # newest filing wins when two holding windows overlap
    held = held.sort_values(["firm_id", "month", "filing_month", "acceptance_datetime"])
    held = held.drop_duplicates(["firm_id", "month"], keep="last")
    return held[["firm_id", "month", "q", "age"]].reset_index(drop=True)


def _size_filters(df: pd.DataFrame, panel: pd.DataFrame, spec: PortfolioSpec) -> pd.DataFrame:
    if not (spec.exclude_microcaps or spec.size_tercile):
        return df
    # cross-sectional cut-offs from the whole return panel that month (lagged ME)
    me = panel[["month", "me_lag"]].dropna()
    cuts = me.groupby("month")["me_lag"].quantile(np.array([0.2, 1 / 3, 2 / 3])).unstack()
    cuts.columns = ["p20", "t1", "t2"]
    df = df.merge(cuts, left_on="month", right_index=True, how="left")
    if spec.exclude_microcaps:
        df = df[df["me_lag"] >= df["p20"]]
    if spec.size_tercile:
        tercile = (
            1 + (df["me_lag"] >= df["t1"]).astype(int) + (df["me_lag"] >= df["t2"]).astype(int)
        )
        df = df[tercile == spec.size_tercile]
    return df.drop(columns=["p20", "t1", "t2"])


def portfolio_returns(
    signals: pd.DataFrame, panel: pd.DataFrame, spec: PortfolioSpec
) -> pd.DataFrame:
    """Monthly quantile returns, L/S, holdings counts and turnover.

    Columns: ``Q1..Qn``, ``LS`` (gross), ``LS_net`` (after ``cost_bps``), ``n_Q1..n_Qn``,
    ``turnover`` (sum of absolute weight changes over both legs). ``panel`` needs
    ``firm_id, month, ret, me_lag``. Use :func:`filinglens.research.prices.add_characteristics`.
    """
    held = holdings(assign_quantiles(signals, spec), spec.hold_months)
    df = held.merge(panel[["firm_id", "month", "ret", "me_lag"]], on=["firm_id", "month"])
    df = df.dropna(subset=["ret"])
    df = _size_filters(df, panel, spec)
    if spec.weighting == "vw":
        df = df[df["me_lag"] > 0]
        df["w"] = df["me_lag"] / df.groupby(["month", "q"])["me_lag"].transform("sum")
    else:
        df["w"] = 1.0 / df.groupby(["month", "q"])["ret"].transform("size")
    n = spec.n_quantiles
    rets = (df["w"] * df["ret"]).groupby([df["month"], df["q"]]).sum().unstack("q")
    rets = rets.reindex(columns=range(1, n + 1))
    counts = df.groupby(["month", "q"]).size().unstack("q").reindex(columns=range(1, n + 1))
    out = pd.DataFrame(index=rets.index.sort_values())
    for k in range(1, n + 1):
        out[f"Q{k}"] = rets[k]
        out[f"n_Q{k}"] = counts[k].fillna(0).astype(int)
    out["LS"] = out[f"Q{n}"] - out["Q1"]

    # turnover of the two legs (long Qn, short Q1); first month counts as a full build
    legs = df[df["q"].isin([1, n])]
    w = legs.pivot_table(index="month", columns=["q", "firm_id"], values="w", fill_value=0.0)
    w = w.reindex(out.index, fill_value=0.0)
    out["turnover"] = w.diff().abs().sum(axis=1)
    if len(out):
        out.loc[out.index[0], "turnover"] = w.iloc[0].abs().sum()
    out["LS_net"] = out["LS"] - out["turnover"] * spec.cost_bps / 1e4
    out.index.name = "month"
    return out
