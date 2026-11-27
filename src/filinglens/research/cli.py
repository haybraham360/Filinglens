"""``filinglens research ...``: the Lazy Prices pipeline (specs/07).

    filinglens research signals                     # similarity signals from parsed filings
    filinglens research prices --source yfinance    # or --source crsp --crsp-file --link-file
    filinglens research factors --liquidity <url>   # Ken French (+ Pastor-Stambaugh)
    filinglens research backtest                    # portfolios, regressions, report
    filinglens research backtest --synthetic        # pipeline check with a planted effect

Outputs go to ``data/processed/research`` (inputs) and ``reports/research/<name>`` (results).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import typer

from filinglens.config import get_settings

research = typer.Typer(
    help="Lazy Prices research: signals, portfolios, regressions.", no_args_is_help=True
)

DEFAULT_CONFIG = Path("config/research.yaml")


def _dir() -> Path:
    return get_settings().processed_dir / "research"


def _split(value: str | None) -> list[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


@research.command()
def signals(
    forms: str = typer.Option("10-K", help="Comma-separated: 10-K and/or 10-Q."),
    tickers: str | None = typer.Option(None, help="Only these tickers."),
    embed: bool = typer.Option(False, help="Add embedding similarity (slower; local model)."),
    lm_dictionary: Path | None = typer.Option(
        None, help="Loughran-McDonald Master Dictionary CSV."
    ),
    min_edit: bool = typer.Option(True, "--min-edit/--no-min-edit", help="Compute Sim_MinEdit."),
    force: bool = typer.Option(False, help="Recompute rows that are already cached."),
) -> None:
    """Compute year-over-year similarity signals for every consecutive filing pair."""
    from filinglens.research.signals import SignalConfig, run_signals
    from filinglens.research.text_features import LMDictionary

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_settings()
    cfg = SignalConfig(with_min_edit=min_edit)
    if embed:
        from filinglens.retrieval.embedder import load_embedder

        cfg.embedder = load_embedder(s.embedding_model, s.embedding_backend)
    if lm_dictionary:
        cfg.lm = LMDictionary.from_csv(lm_dictionary)
    out = _dir() / "signals.parquet"
    table = run_signals(
        s.processed_dir, out, cfg, forms=_split(forms), tickers=_split(tickers) or None, force=force
    )
    typer.echo(f"{len(table)} signal rows -> {out}")
    if len(table):
        typer.echo(
            table.groupby("section")[["sim_cosine_tf", "sim_jaccard", "sim_simple"]]
            .mean()
            .round(3)
            .to_string()
        )


@research.command()
def prices(
    source: str = typer.Option("yfinance", help="yfinance (free, survivorship-biased) or crsp."),
    start: str = typer.Option("2000-01-01"),
    end: str = typer.Option(date.today().isoformat()),
    crsp_file: Path | None = typer.Option(None, help="CRSP monthly stock file export (CSV)."),
    link_file: Path | None = typer.Option(None, help="CSV with cik,permno[,start,end]."),
) -> None:
    """Build the monthly returns panel for every firm in the filings index."""
    from filinglens.parsing.pipeline import load_filings_index
    from filinglens.research.prices import load_crsp_csv, save_panel, yfinance_monthly

    s = get_settings()
    if source == "crsp":
        if not crsp_file or not link_file:
            raise typer.BadParameter("--crsp-file and --link-file are required for crsp.")
        panel = load_crsp_csv(crsp_file, link_file)
    else:
        index = load_filings_index(s.processed_dir)
        if index.empty:
            raise typer.BadParameter("No filings index; run `filinglens ingest` first.")
        latest = index.sort_values("filing_date").drop_duplicates("cik", keep="last")
        tickers = {str(r.cik): str(r.ticker) for r in latest.itertuples() if r.ticker}
        facts_files = sorted((s.processed_dir / "facts").glob("*.parquet"))
        facts = pd.concat([pd.read_parquet(f) for f in facts_files]) if facts_files else None
        panel = yfinance_monthly(tickers, start, end, facts=facts, cache_dir=s.cache_dir)
        typer.echo("Note: yfinance excludes most delisted stocks (survivorship bias).")
    out = _dir() / "prices.parquet"
    save_panel(panel, out)
    typer.echo(f"{panel['firm_id'].nunique()} firms, {len(panel)} firm-months -> {out}")


@research.command()
def factors(
    liquidity: str | None = typer.Option(
        None,
        help="Pastor-Stambaugh liquidity file (URL or path), for the paper's 5-factor model.",
    ),
) -> None:
    """Download and merge monthly factors (cached)."""
    from filinglens.research.factors import available_models, load_factors

    s = get_settings()
    table = load_factors(s.cache_dir, liquidity=liquidity)
    out = _dir() / "factors.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.assign(month=table.index.astype(str)).to_parquet(out, index=False)
    typer.echo(f"{len(table)} months ({table.index.min()} to {table.index.max()}) -> {out}")
    typer.echo(f"Models available: {', '.join(available_models(table))}")


def _load_factors(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.PeriodIndex(df.pop("month"), freq="M")
    return df


@research.command()
def backtest(
    config: Path = typer.Option(DEFAULT_CONFIG, help="Pre-registered variant grid."),
    synthetic: bool = typer.Option(False, help="Run on simulated data with a planted effect."),
    name: str | None = typer.Option(None, help="Output folder under reports/research."),
) -> None:
    """Portfolios, factor regressions, robustness, Fama-MacBeth, then the report."""
    from filinglens.research.backtest import BacktestConfig, run_backtest
    from filinglens.research.prices import add_characteristics, book_equity_panel, load_panel
    from filinglens.research.report import render
    from filinglens.research.signals import load_signals

    cfg = BacktestConfig.from_yaml(config) if config.exists() else BacktestConfig()
    if synthetic:
        from filinglens.research.synthetic import make_world

        world = make_world()
        sig, panel, fac = world.signals, add_characteristics(world.panel), world.factors
        cfg.signals = [c for c in cfg.signals if c in sig]
        cfg.sections = ["full"]
    else:
        d = _dir()
        missing = [
            f
            for f in ("signals.parquet", "prices.parquet", "factors.parquet")
            if not (d / f).exists()
        ]
        if missing:
            raise typer.BadParameter(
                f"Missing {', '.join(missing)} in {d}. Run `filinglens research signals`, "
                "`prices` and `factors` first."
            )
        sig = load_signals(d / "signals.parquet")
        sev_path = d / "llm_severity.parquet"
        if sev_path.exists():
            sev = pd.read_parquet(sev_path)
            # in-sample (pre-cutoff) ratings never enter the main results
            sev = sev[~sev["in_sample"]][["accession_no", "llm_change_severity"]]
            sig = sig.merge(sev, on="accession_no", how="left")
        raw = load_panel(d / "prices.parquet")
        facts_files = sorted((get_settings().processed_dir / "facts").glob("*.parquet"))
        be = None
        if facts_files and raw.attrs.get("source") != "crsp":
            facts = pd.concat([pd.read_parquet(f) for f in facts_files])
            be = book_equity_panel(facts, raw)
        panel = add_characteristics(raw, be)
        fac = _load_factors(d / "factors.parquet")
    result = run_backtest(sig, panel, fac, cfg)
    out = Path("reports/research") / (name or ("synthetic" if synthetic else "latest"))
    path = render(result, out)
    typer.echo(f"Report -> {path}")


@research.command()
def severity(
    cutoff: str = typer.Option(..., help="The model's training cutoff (YYYY-MM-DD)."),
    provider: str = typer.Option("ollama", help="ollama (free, local) or anthropic (paid)."),
    allow_in_sample: bool = typer.Option(
        False, help="Also rate pre-cutoff filings, flagged in_sample."
    ),
) -> None:
    """LLM change-severity signal on anonymized Item 1A changes (see llm_severity.py)."""
    from filinglens.llm.calllog import MemoryCallLog
    from filinglens.llm.factory import make_llm
    from filinglens.parsing.pipeline import load_filings_index
    from filinglens.research.llm_severity import SeverityConfig, run_llm_severity
    from filinglens.research.signals import pair_filings

    s = get_settings()
    if provider not in ("ollama", "anthropic"):
        raise typer.BadParameter("provider must be ollama or anthropic")
    llm = make_llm(s, provider, MemoryCallLog())  # type: ignore[arg-type]
    assert llm is not None
    pairs = pair_filings(load_filings_index(s.processed_dir))
    cfg = SeverityConfig(
        training_cutoff=date.fromisoformat(cutoff), allow_in_sample=allow_in_sample
    )
    table = run_llm_severity(pairs, s.processed_dir, llm, cfg)
    out = _dir() / "llm_severity.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    typer.echo(f"{len(table)} rated filings -> {out}")
