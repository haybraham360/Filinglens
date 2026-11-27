"""Run the full Lazy Prices study: portfolios, robustness, Fama-MacBeth (specs/07).

``run_backtest`` evaluates a grid of :class:`PortfolioSpec` variants. **Every** variant and
period is logged in ``variants`` (multiple-testing awareness), and the report prints that
table in full, not just the best result. Parameters are fixed up front in
``config/research.yaml``. Nothing is tuned on the post-2014 period.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from filinglens.research.factors import available_models
from filinglens.research.portfolios import PortfolioSpec, portfolio_returns
from filinglens.research.regressions import (
    FMResult,
    factor_regression,
    fama_macbeth,
    quantile_table,
    sharpe,
)

logger = logging.getLogger(__name__)

CLASSICAL = ("sim_cosine_tf", "sim_jaccard", "sim_minedit", "sim_simple")
NEW = ("sim_embed", "sim_embed_aligned", "n_new_risks", "n_removed_risks", "lm_neg_delta")
CONTROLS = ("log_me", "log_bm", "ret_lag1", "ret_12_2")

# Signals where a *larger* value means *more* change. They are negated before sorting, so
# Q1 is always the "changers".
CHANGE_LIKE = {"n_new_risks", "n_removed_risks", "lm_neg_delta", "llm_change_severity"}


@dataclass
class BacktestConfig:
    """The pre-registered grid. Load it from ``config/research.yaml``."""

    signals: list[str] = field(default_factory=lambda: [*CLASSICAL, "sim_embed", "n_new_risks"])
    main_section: str = "full"
    sections: list[str] = field(default_factory=lambda: ["full", "1A", "3", "7"])
    holds: list[int] = field(default_factory=lambda: [3, 6, 12])
    main_hold: int = 3
    weightings: list[str] = field(default_factory=lambda: ["ew", "vw"])
    breakpoints: list[str] = field(default_factory=lambda: ["prior_year", "trailing_12m"])
    models: list[str] = field(default_factory=lambda: ["excess", "ff3", "paper5", "ff5_mom"])
    cost_bps: list[float] = field(default_factory=lambda: [0.0, 10.0, 25.0])
    split_month: str = "2015-01"  # the paper's sample ends in 2014: later months are out of sample
    size_terciles: bool = True
    exclude_microcaps: bool = True
    min_breakpoint_obs: int = 25

    @classmethod
    def from_yaml(cls, path: Path) -> BacktestConfig:
        """Load overrides from YAML (unknown keys are an error)."""
        data = yaml.safe_load(path.read_text()) or {}
        return cls(**data)


@dataclass
class BacktestResult:
    """Everything the report needs."""

    config: BacktestConfig
    meta: dict[str, Any]
    quantile_tables: dict[str, pd.DataFrame]
    ls_series: dict[str, pd.Series]
    variants: pd.DataFrame
    fama_macbeth: pd.DataFrame


def orient(signals: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Negate change-like columns so lower always means "changed more"."""
    out = signals.copy()
    for c in columns:
        if c in CHANGE_LIKE and c in out:
            out[c] = -out[c]
    return out


def _periods(index: pd.PeriodIndex, split: pd.Period) -> dict[str, pd.Series]:
    s = pd.Series(True, index=index)
    return {
        "full": s,
        "pre": pd.Series(index < split, index=index),
        "post": pd.Series(index >= split, index=index),
    }


def _log_variant(
    rows: list[dict[str, Any]],
    spec: PortfolioSpec,
    port: pd.DataFrame,
    factors: pd.DataFrame,
    models: Sequence[str],
    split: pd.Period,
    family: str,
) -> None:
    for column in ["LS", "LS_net"] if spec.cost_bps else ["LS"]:
        for period, mask in _periods(pd.PeriodIndex(port.index), split).items():
            y = port.loc[mask.to_numpy(), column]
            for model in models:
                res = factor_regression(y, factors, model, hold_months=spec.hold_months)
                rows.append(
                    {
                        "family": family,
                        "variant": spec.label(),
                        **{k: v for k, v in asdict(spec).items() if k != "min_breakpoint_obs"},
                        "series": column,
                        "period": period,
                        "model": model,
                        "alpha_bps": res.alpha * 1e4 if res else math.nan,
                        "t": res.t_alpha if res else math.nan,
                        "n_months": res.n_months if res else int(y.notna().sum()),
                        "sharpe": sharpe(y),
                        "avg_n_Q1": float(port["n_Q1"].mean()) if len(port) else math.nan,
                        "avg_turnover": float(port["turnover"].mean()) if len(port) else math.nan,
                    }
                )


def fm_panel(
    signals: pd.DataFrame, panel: pd.DataFrame, section: str, hold: int, columns: Sequence[str]
) -> pd.DataFrame:
    """Firm-months with the firm's latest signal released 1..``hold`` months earlier."""
    cols = [c for c in columns if c in signals and signals[c].notna().any()]
    sig = signals[signals["section"] == section][
        ["cik", "filing_month", "acceptance_datetime", *cols]
    ]
    parts = [sig.assign(month=sig["filing_month"] + a) for a in range(1, hold + 1)]
    if not parts:
        return pd.DataFrame()
    held = pd.concat(parts, ignore_index=True).sort_values(
        ["cik", "month", "filing_month", "acceptance_datetime"]
    )
    held = held.drop_duplicates(["cik", "month"], keep="last").rename(columns={"cik": "firm_id"})
    return held.merge(panel, on=["firm_id", "month"], how="inner")


def run_fama_macbeth(
    signals: pd.DataFrame, panel: pd.DataFrame, cfg: BacktestConfig
) -> pd.DataFrame:
    """Each signal alone with controls, then classical + new signals together."""
    available = [c for c in cfg.signals if c in signals and signals[c].notna().any()]
    data = fm_panel(signals, panel, cfg.main_section, cfg.main_hold, available)
    if data.empty:
        return pd.DataFrame()
    controls = [c for c in CONTROLS if c in data and data[c].notna().any()]
    specs: list[tuple[str, list[str]]] = [(s, [s]) for s in available]
    if len(available) > 1:
        specs.append(("incremental", available))
    rows = []
    for name, xs in specs:
        res: FMResult | None = fama_macbeth(
            data, "ret", [*xs, *controls], hold_months=cfg.main_hold
        )
        if res is None:
            continue
        for var in [*xs, *controls]:
            rows.append(
                {
                    "spec": name,
                    "variable": var,
                    "coef": res.coef.get(var, math.nan),
                    "t": res.t.get(var, math.nan),
                    "n_months": res.n_months,
                    "avg_n": res.avg_n,
                }
            )
    return pd.DataFrame(rows)


def run_backtest(
    signals: pd.DataFrame, panel: pd.DataFrame, factors: pd.DataFrame, cfg: BacktestConfig
) -> BacktestResult:
    """Main tables, the full robustness grid, and Fama-MacBeth."""
    models = [m for m in cfg.models if m in available_models(factors)]
    missing_models = [m for m in cfg.models if m not in models]
    signals = orient(signals, cfg.signals)
    present = [s for s in cfg.signals if s in signals and signals[s].notna().any()]
    split = pd.Period(cfg.split_month, freq="M")
    base = PortfolioSpec(
        signal=present[0] if present else "sim_cosine_tf",
        section=cfg.main_section,
        hold_months=cfg.main_hold,
        min_breakpoint_obs=cfg.min_breakpoint_obs,
    )
    tables: dict[str, pd.DataFrame] = {}
    ls: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []

    def run(spec: PortfolioSpec, family: str) -> pd.DataFrame:
        port = portfolio_returns(signals, panel, spec)
        _log_variant(rows, spec, port, factors, models, split, family)
        return port

    for signal in present:
        for w in cfg.weightings:
            spec = base.with_(signal=signal, weighting=w)
            port = run(spec, "main")
            tables[spec.label()] = quantile_table(
                port, factors, models, hold_months=spec.hold_months
            )
            ls[spec.label()] = port["LS"]
            for hold in [h for h in cfg.holds if h != cfg.main_hold]:
                run(spec.with_(hold_months=hold), "holding period")
            for bp in [b for b in cfg.breakpoints if b != base.breakpoints]:
                run(spec.with_(breakpoints=bp), "breakpoints")
            for section in [s for s in cfg.sections if s != cfg.main_section]:
                run(spec.with_(section=section), "section")
            if cfg.exclude_microcaps:
                run(spec.with_(exclude_microcaps=True), "size")
            if cfg.size_terciles:
                for t in (1, 2, 3):
                    run(spec.with_(size_tercile=t), "size")
            for cost in [c for c in cfg.cost_bps if c]:
                run(spec.with_(cost_bps=cost), "transaction costs")

    meta = {
        "source": panel.attrs.get("source", "unknown"),
        "n_filings": int(signals["accession_no"].nunique()) if "accession_no" in signals else 0,
        "n_firms": int(signals["cik"].nunique()) if "cik" in signals else 0,
        "first_month": str(signals["filing_month"].min()) if len(signals) else None,
        "last_month": str(signals["filing_month"].max()) if len(signals) else None,
        "forms": sorted(signals["form"].unique().tolist()) if "form" in signals else [],
        "models": models,
        "missing_models": missing_models,
        "signals": present,
        "missing_signals": [s for s in cfg.signals if s not in present],
    }
    return BacktestResult(
        config=cfg,
        meta=meta,
        quantile_tables=tables,
        ls_series=ls,
        variants=pd.DataFrame(rows),
        fama_macbeth=run_fama_macbeth(signals, panel, cfg),
    )
