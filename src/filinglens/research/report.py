"""Render a :class:`BacktestResult` as markdown tables, a variants CSV and an L/S chart."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from filinglens.research.backtest import BacktestResult

SURVIVORSHIP = {
    "crsp": (
        "Returns are from CRSP, including delisted firms, with delisting returns compounded "
        "into the final month. The sample is survivorship-free."
    ),
    "yfinance": (
        "**Survivorship bias.** Returns are from Yahoo Finance, which mostly lacks delisted "
        "stocks. Firms that went bankrupt or were acquired drop out, and the paper finds these "
        "are disproportionately big changers. The missing firms would most likely have had "
        "very negative returns in Q1, so the short leg, and with it the L/S spread, is probably "
        "**understated**. Survivorship can also flatter the long leg, so the net direction "
        "isn't certain. Treat these results as indicative."
    ),
    "synthetic": (
        "**Synthetic data.** This run uses simulated firms with a known, planted effect. It checks "
        "the pipeline and says nothing about real markets."
    ),
}
MODEL_NAMES = {
    "excess": "Excess return",
    "capm": "CAPM alpha",
    "ff3": "FF3 alpha",
    "carhart": "FF3 + momentum alpha",
    "paper5": "Paper 5-factor alpha (FF3 + MOM + LIQ)",
    "ff5": "FF5 alpha",
    "ff5_mom": "FF5 + momentum alpha",
}


def stars(t: float) -> str:
    """Significance stars for a t-stat (two-sided, normal approximation)."""
    if t is None or math.isnan(t):
        return ""
    a = abs(t)
    return "***" if a >= 2.576 else "**" if a >= 1.96 else "*" if a >= 1.645 else ""


def _cell(alpha: float, t: float, scale: float = 1e4) -> str:
    if alpha is None or math.isnan(alpha):
        return "n/a"
    return f"{alpha * scale:.1f}{stars(t)} ({t:.2f})"


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def quantile_markdown(table: pd.DataFrame) -> str:
    """One Table II-style block: rows per model, alpha in bps/month with NW t-stats."""
    cols = list(table.columns)
    rows = []
    for model in table.index.get_level_values("model").unique():
        a, t = table.loc[(model, "alpha")], table.loc[(model, "t")]
        rows.append([MODEL_NAMES.get(model, model), *[_cell(a[c], t[c]) for c in cols]])
    return _md_table(["", *cols], rows)


def preferred_model(models: list[str]) -> str:
    """The paper's own model when its factors are available, else the richest one we have."""
    for m in ("paper5", "carhart", "ff3", "capm", "excess"):
        if m in models:
            return m
    return models[0] if models else "excess"


def variants_markdown(variants: pd.DataFrame, model: str, family: str | None = None) -> str:
    """L/S alpha by variant for one model, full / pre-2015 / post-2014 side by side."""
    v = variants[(variants["model"] == model)]
    if family:
        v = v[v["family"] == family]
    if v.empty:
        return "_No variants._"
    rows = []
    for (label, series), g in v.groupby(["variant", "series"], sort=False):
        by = {r["period"]: r for r in g.to_dict("records")}
        cells = [
            _cell(by[p]["alpha_bps"] / 1e4, by[p]["t"]) if p in by else "n/a"
            for p in ("full", "pre", "post")
        ]
        n = by.get("full", {}).get("n_months", "")
        rows.append([f"`{label}`" + (" net" if series == "LS_net" else ""), *cells, str(n)])
    return _md_table(["Variant", "Full sample", "Pre-2015", "Post-2014", "Months"], rows)


def fm_markdown(fm: pd.DataFrame) -> str:
    """Fama-MacBeth slopes (x100 for readability) with NW t-stats, one column per spec."""
    if fm.empty:
        return "_Fama-MacBeth not run (too few months or firms)._"
    specs = list(dict.fromkeys(fm["spec"]))
    variables = list(dict.fromkeys(fm["variable"]))
    rows = []
    for var in variables:
        row = [f"`{var}`"]
        for s in specs:
            hit = fm[(fm["spec"] == s) & (fm["variable"] == var)]
            row.append(
                _cell(float(hit["coef"].iloc[0]), float(hit["t"].iloc[0]), 100) if len(hit) else ""
            )
        rows.append(row)
    months = fm.groupby("spec")["n_months"].first()
    rows.append(["Months", *[str(int(months[s])) for s in specs]])
    return _md_table(["", *specs], rows)


def plot_cumulative(result: BacktestResult, path: Path, labels: list[str] | None = None) -> Path:
    """Cumulative L/S (sum of monthly returns) with a marker at the out-of-sample split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    labels = labels or list(result.ls_series)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    for label in labels:
        s = result.ls_series[label].dropna()
        if s.empty:
            continue
        ax.plot(
            pd.PeriodIndex(s.index).to_timestamp(), s.cumsum() * 100, linewidth=1.4, label=label
        )
    split_ts = pd.Period(result.config.split_month, freq="M").to_timestamp()
    split = float(mdates.date2num(split_ts))  # type: ignore[no-untyped-call]
    ax.axvline(split, color="0.35", linestyle="--", linewidth=1)
    ax.text(
        split, ax.get_ylim()[1], "  post-2014 (out of sample)", va="top", fontsize=8, color="0.3"
    )
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set_ylabel("Cumulative L/S return (%, non-compounded)")
    ax.set_title("Lazy Prices: long non-changers (Q5), short changers (Q1)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def render(result: BacktestResult, out_dir: Path) -> Path:
    """Write ``results.md``, ``variants.csv``, ``fama_macbeth.csv`` and the chart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, cfg = result.meta, result.config
    model = preferred_model(meta["models"])
    result.variants.to_csv(out_dir / "variants.csv", index=False)
    result.fama_macbeth.to_csv(out_dir / "fama_macbeth.csv", index=False)
    main_ew = [k for k in result.ls_series if k.endswith("/ew/prior_year")]
    chart = plot_cumulative(result, out_dir / "cumulative_ls.png", main_ew or None)

    v = result.variants
    tested = v[(v["model"] == model) & (v["series"] == "LS")]
    n_tests = len(tested)
    n_sig = int((tested["t"].abs() >= 1.96).sum())
    lines = [
        "# Lazy Prices replication: results",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `filinglens research backtest`.",
        "",
    ]
    if meta["source"] == "synthetic":
        lines += ["> **PIPELINE CHECK ON SYNTHETIC DATA. NOT A RESULT.**", ""]
    lines += [
        "## Data",
        "",
        f"- Filings: {meta['n_filings']:,} ({', '.join(meta['forms'])}), "
        f"{meta['n_firms']:,} firms, released {meta['first_month']} to {meta['last_month']}.",
        f"- Returns: `{meta['source']}`. {SURVIVORSHIP.get(meta['source'], '')}",
        f"- Factor models run: {', '.join(meta['models']) or 'none'}."
        + (
            f" Not available: {', '.join(meta['missing_models'])} (missing factor data)."
            if meta["missing_models"]
            else ""
        ),
        f"- Signals: {', '.join(f'`{s}`' for s in meta['signals'])}."
        + (
            f" Missing in the input: {', '.join(meta['missing_signals'])}."
            if meta["missing_signals"]
            else ""
        ),
        "",
        "## Main results",
        "",
        f"Section `{cfg.main_section}`, held {cfg.main_hold} months, breakpoints from the prior "
        "year's distribution, entry the month after release. Alphas are in bps per month, "
        "with Newey-West t-stats in parentheses. Q1 = changers, Q5 = non-changers, "
        "L/S = Q5 - Q1.",
        "",
    ]
    for label, table in result.quantile_tables.items():
        lines += [f"### `{label}`", "", quantile_markdown(table), ""]
    lines += [
        "## Out of sample: before and after the paper",
        "",
        f"The paper's sample ends in 2014. Months from {cfg.split_month} on are out of sample, "
        "which tests whether the anomaly decayed after publication. "
        f"Model: {MODEL_NAMES.get(model, model)}.",
        "",
        variants_markdown(v, model, "main"),
        "",
        "## Robustness",
        "",
    ]
    for family in ("holding period", "breakpoints", "section", "size", "transaction costs"):
        if (v["family"] == family).any():
            lines += [f"### {family.capitalize()}", "", variants_markdown(v, model, family), ""]
    lines += [
        "## Fama-MacBeth",
        "",
        "Monthly cross-sectional regressions of returns on each signal with controls (log "
        "lagged market equity, log B/M when book equity is available, last month's return, "
        "return from t-12 to t-2). The last column puts the classical and new signals in "
        "together: the incremental-value test. Slopes are x100, with Newey-West t-stats.",
        "",
        fm_markdown(result.fama_macbeth),
        "",
        "## Multiple testing",
        "",
        f"{n_tests} L/S tests were run with the {model} model (every variant x period). "
        f"{n_sig} have |t| >= 1.96. Under the null, about {0.05 * n_tests:.0f} would pass by "
        "chance, and the variants are highly correlated. Every test is in `variants.csv`, "
        "including the ones that failed.",
        "",
        f"![Cumulative L/S]({chart.name})",
        "",
    ]
    path = out_dir / "results.md"
    path.write_text("\n".join(lines))
    return path
