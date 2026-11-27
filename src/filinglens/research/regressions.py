"""Factor regressions, Sharpe ratios and Fama-MacBeth (specs/07, steps 5 and 7).

All t-statistics use Newey-West (HAC) standard errors. Overlapping 3- to 12-month holding
windows make monthly portfolio returns autocorrelated, so plain OLS t-stats would
overstate significance. The default lag count is the Newey-West (1994) rule
``floor(4 * (T/100)^(2/9))``, raised to at least ``hold_months - 1``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from filinglens.research.factors import MODELS


def nw_lags(n_obs: int, hold_months: int = 1) -> int:
    """Newey-West lag rule, at least ``hold_months - 1`` to cover overlapping windows."""
    rule: int = math.floor(4 * (n_obs / 100) ** (2 / 9))
    return max(rule, hold_months - 1, 1)


@dataclass(frozen=True)
class RegResult:
    """One time-series regression of a portfolio on a factor model."""

    model: str
    alpha: float  # monthly, decimal
    t_alpha: float
    betas: dict[str, float]
    r2: float
    n_months: int
    lags: int


def factor_regression(
    y: pd.Series,
    factors: pd.DataFrame,
    model: str,
    *,
    lags: int | None = None,
    hold_months: int = 1,
) -> RegResult | None:
    """Regress ``y`` on the model's factors with HAC errors. ``None`` if too few months.

    ``y`` must already be an excess return (quantile minus RF) or a zero-cost spread (L/S).
    The ``excess`` model is an intercept-only regression: the mean with a NW t-stat.
    """
    cols = list(MODELS[model])
    data = pd.concat([y.rename("y"), factors[cols]], axis=1, join="inner").dropna()
    if len(data) < max(12, len(cols) + 3):
        return None
    lags = lags if lags is not None else nw_lags(len(data), hold_months)
    x = sm.add_constant(data[cols], has_constant="add") if cols else np.ones((len(data), 1))
    fit = sm.OLS(data["y"].to_numpy(), x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    params = np.asarray(fit.params)
    tvals = np.asarray(fit.tvalues)
    return RegResult(
        model=model,
        alpha=float(params[0]),
        t_alpha=float(tvals[0]),
        betas={c: float(b) for c, b in zip(cols, params[1:], strict=True)},
        r2=float(fit.rsquared) if cols else 0.0,
        n_months=len(data),
        lags=lags,
    )


def sharpe(returns: pd.Series, rf: pd.Series | None = None) -> float:
    """Annualized Sharpe ratio of monthly returns (minus ``rf`` when given)."""
    r = returns if rf is None else (returns - rf.reindex(returns.index))
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return math.nan
    return float(r.mean() / r.std(ddof=1) * math.sqrt(12))


def quantile_table(
    port: pd.DataFrame,
    factors: pd.DataFrame,
    models: Sequence[str],
    *,
    n_quantiles: int = 5,
    hold_months: int = 3,
    ls_column: str = "LS",
) -> pd.DataFrame:
    """Table II layout: rows (model, stat) x columns Q1..Qn, L/S.

    Quantiles are measured in excess of RF; the L/S spread is already zero-cost.
    """
    rf = factors["RF"]
    cols = [f"Q{k}" for k in range(1, n_quantiles + 1)]
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for model in models:
        alpha, tstat = {}, {}
        for c in [*cols, ls_column]:
            y = port[c] if c == ls_column else port[c] - rf.reindex(port.index)
            res = factor_regression(y, factors, model, hold_months=hold_months)
            alpha[c] = res.alpha if res else math.nan
            tstat[c] = res.t_alpha if res else math.nan
        rows[(model, "alpha")] = alpha
        rows[(model, "t")] = tstat
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["model", "stat"])
    return table.rename(columns={ls_column: "L/S"})


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip at the given quantiles of ``s`` (used cross-sectionally, month by month)."""
    lo, hi = float(s.quantile(lower)), float(s.quantile(upper))
    return s.clip(lo, hi)


@dataclass(frozen=True)
class FMResult:
    """Time-series averages of monthly cross-sectional slopes."""

    coef: dict[str, float]
    t: dict[str, float]
    n_months: int
    avg_n: float
    lags: int


def fama_macbeth(
    panel: pd.DataFrame,
    y: str,
    xs: Sequence[str],
    *,
    min_obs: int = 30,
    winsor: tuple[float, float] | None = (0.01, 0.99),
    lags: int | None = None,
    hold_months: int = 1,
) -> FMResult | None:
    """Monthly OLS of ``y`` on ``xs``; the slope series' means get Newey-West t-stats.

    Regressors are winsorized cross-sectionally each month (the paper winsorizes the
    similarity measures at 1/99), which uses no future information.
    """
    slopes, sizes = [], []
    for month, cs in panel.dropna(subset=[y, *xs]).groupby("month"):
        if len(cs) < max(min_obs, len(xs) + 2):
            continue
        xmat = cs[list(xs)].astype(float)
        if winsor:
            xmat = xmat.apply(lambda s: winsorize(s, *winsor))
        if (xmat.std(ddof=0) == 0).any():
            continue
        fit = sm.OLS(
            cs[y].astype(float).to_numpy(), sm.add_constant(xmat, has_constant="add")
        ).fit()
        slopes.append(pd.Series(fit.params, name=month))
        sizes.append(len(cs))
    if len(slopes) < 12:
        return None
    slope_df = pd.DataFrame(slopes)
    lags = lags if lags is not None else nw_lags(len(slope_df), hold_months)
    coef, t = {}, {}
    for c in slope_df.columns:
        fit = sm.OLS(slope_df[c].to_numpy(), np.ones(len(slope_df))).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags}
        )
        coef[str(c)], t[str(c)] = float(fit.params[0]), float(fit.tvalues[0])
    return FMResult(coef=coef, t=t, n_months=len(slope_df), avg_n=float(np.mean(sizes)), lags=lags)
