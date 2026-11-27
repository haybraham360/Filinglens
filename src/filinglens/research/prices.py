"""Monthly stock returns and market equity, in one panel format for the backtest.

Panel columns: ``firm_id`` (zero-padded CIK, to match signals), ``month`` (Period[M]),
``ret`` (total return, decimal), ``me`` (market equity at month end, USD). ``panel.attrs``
records the source, so the report can state survivorship caveats automatically.

Two sources:

* **CRSP** (survivorship-free; via WRDS if the program provides it): ``load_crsp_csv``
  reads a monthly stock file export and compounds delisting returns into the last month.
  A link table maps CIK to PERMNO.
* **yfinance** (free; optional dependency, ``uv sync --extra research``): adjusted-close
  returns for tickers that still trade. **Survivorship-biased:** delisted firms, often the
  bankrupt "changers", are missing, which probably *understates* short-leg returns.
  Market equity is the unadjusted close times XBRL shares outstanding, point in time.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PANEL_COLUMNS = ["firm_id", "month", "ret", "me"]
SHARES_CONCEPTS = ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding")
EQUITY_CONCEPTS = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)


def _to_month(values: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(pd.to_datetime(values), freq="M")


def load_crsp_csv(path: Path, link_path: Path, *, common_only: bool = True) -> pd.DataFrame:
    """CRSP monthly file export (PERMNO, date, RET, DLRET, PRC, SHROUT, SHRCD) -> panel.

    ``link_path`` is a CSV with ``cik,permno`` (optionally ``start,end`` dates for each link).
    Non-numeric return codes become missing. ``me = |PRC| * SHROUT * 1000``. With
    ``common_only``, only share codes 10 and 11 (ordinary common stock) are kept.
    """
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = [c.upper() for c in raw.columns]
    if common_only and "SHRCD" in raw:
        raw = raw[raw["SHRCD"].isin([10, 11])]
    ret = pd.to_numeric(raw["RET"], errors="coerce")
    dlret = pd.to_numeric(raw["DLRET"], errors="coerce") if "DLRET" in raw else None
    if dlret is not None:
        # delisting month: compound the delisting return (missing RET counts as 0)
        has_dl = dlret.notna()
        ret = ret.where(~has_dl, (1 + ret.fillna(0)) * (1 + dlret) - 1)
    panel = pd.DataFrame(
        {
            "permno": raw["PERMNO"].astype(int),
            "date": pd.to_datetime(raw["DATE"].astype(str)),
            "ret": ret,
            "me": pd.to_numeric(raw["PRC"], errors="coerce").abs()
            * pd.to_numeric(raw["SHROUT"], errors="coerce")
            * 1000,
        }
    )
    link = pd.read_csv(link_path, dtype={"cik": str})
    link["cik"] = link["cik"].str.zfill(10)
    panel = panel.merge(link, on="permno", how="inner")
    if {"start", "end"} <= set(link.columns):
        start = pd.to_datetime(panel["start"]).fillna(pd.Timestamp.min)
        end = pd.to_datetime(panel["end"]).fillna(pd.Timestamp.max)
        panel = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    panel["month"] = _to_month(panel["date"])
    out = (
        panel.rename(columns={"cik": "firm_id"})[PANEL_COLUMNS]
        .drop_duplicates(["firm_id", "month"])
        .sort_values(["firm_id", "month"])
        .reset_index(drop=True)
    )
    out.attrs["source"] = "crsp"
    return out


def facts_asof(
    facts: pd.DataFrame, concepts: tuple[str, ...], months: pd.PeriodIndex, cik: str
) -> pd.Series:
    """Latest value of the first available concept *filed before* each month starts.

    Point in time: a value filed on 2023-02-03 is first usable in March 2023.
    """
    f = facts[(facts["cik"] == cik) & facts["concept"].isin(concepts)].copy()
    if f.empty:
        return pd.Series(np.nan, index=months)
    f["rank"] = f["concept"].map({c: i for i, c in enumerate(concepts)})
    f["usable"] = _to_month(f["filed_date"]) + 1
    f = f.sort_values(["usable", "period_end", "rank"]).drop_duplicates("usable", keep="last")
    s = pd.Series(f["value"].astype(float).to_numpy(), index=pd.PeriodIndex(f["usable"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(months)).ffill().reindex(months)


def yfinance_monthly(
    tickers: Mapping[str, str],
    start: str,
    end: str,
    *,
    facts: pd.DataFrame | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Monthly panel from yfinance for ``{cik: ticker}`` (survivorship-biased; see module doc)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("yfinance is not installed: `uv sync --extra research`.") from exc

    frames = []
    for cik, ticker in tickers.items():
        cache = cache_dir / "prices" / f"{ticker}.parquet" if cache_dir else None
        if cache is not None and cache.exists():
            daily = pd.read_parquet(cache)
        else:
            daily = yf.download(
                ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False
            )
            if isinstance(daily.columns, pd.MultiIndex):
                daily.columns = daily.columns.get_level_values(0)
            if cache is not None and not daily.empty:
                cache.parent.mkdir(parents=True, exist_ok=True)
                daily.to_parquet(cache)
        if daily.empty:
            logger.warning("no prices for %s", ticker)
            continue
        monthly = daily[["Adj Close", "Close"]].resample("ME").last().dropna()
        months = pd.PeriodIndex(monthly.index, freq="M")
        ret = monthly["Adj Close"].pct_change().to_numpy()
        me = np.full(len(monthly), np.nan)
        if facts is not None:
            shares = facts_asof(facts, SHARES_CONCEPTS, months, cik).to_numpy()
            me = monthly["Close"].to_numpy() * shares
        frames.append(pd.DataFrame({"firm_id": cik, "month": months, "ret": ret, "me": me}))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PANEL_COLUMNS)
    out = out.dropna(subset=["ret"]).sort_values(["firm_id", "month"]).reset_index(drop=True)
    out.attrs["source"] = "yfinance"
    return out


def add_characteristics(panel: pd.DataFrame, book_equity: pd.Series | None = None) -> pd.DataFrame:
    """Lagged controls for Fama-MacBeth: ``me_lag``, ``ret_lag1`` and ``ret_12_2``.

    ``ret_12_2`` compounds months t-12 to t-2. Gaps in the monthly series give NaN rather
    than silently spanning missing months. ``book_equity`` (indexed by firm_id and month,
    point in time) adds ``log_bm``.
    """
    p = panel.sort_values(["firm_id", "month"]).reset_index(drop=True)
    p["_mi"] = pd.PeriodIndex(p["month"], freq="M").asi8  # month ordinals
    g = p.groupby("firm_id", sort=False)
    contiguous = g["_mi"].diff() == 1
    p["me_lag"] = g["me"].shift(1).where(contiguous)
    p["ret_lag1"] = g["ret"].shift(1).where(contiguous)
    p["ret_12_2"] = np.nan
    for _, idx in g.groups.items():
        sub = p.loc[idx]
        s = pd.Series(np.log1p(sub["ret"].to_numpy()), index=sub["_mi"].to_numpy())
        full = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
        mom = np.expm1(full.shift(2).rolling(11, min_periods=11).sum())
        p.loc[idx, "ret_12_2"] = mom.reindex(sub["_mi"]).to_numpy()
    p["log_me"] = np.log(p["me_lag"].where(p["me_lag"] > 0))
    if book_equity is not None:
        be = book_equity.reindex(pd.MultiIndex.from_frame(p[["firm_id", "month"]])).to_numpy()
        p["log_bm"] = np.log(np.where((be > 0) & (p["me_lag"] > 0), be / p["me_lag"], np.nan))
    out = p.drop(columns="_mi")
    out.attrs = panel.attrs
    return out


def book_equity_panel(facts: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Point-in-time book equity per (firm_id, month) from XBRL 10-K facts.

    Only annual (10-K) values are used, and each is usable from the month after it was
    filed. The paper uses Compustat book equity; XBRL ``StockholdersEquity`` is the free
    approximation, without the deferred-tax and preferred-stock adjustments.
    """
    annual = facts[facts["form"].astype(str).str.startswith("10-K")]
    parts = []
    for firm, grp in panel.groupby("firm_id"):
        months = pd.PeriodIndex(grp["month"], freq="M")
        be = facts_asof(annual, EQUITY_CONCEPTS, months, str(firm))
        parts.append(
            pd.Series(
                be.to_numpy(), index=pd.MultiIndex.from_arrays([[firm] * len(months), months])
            )
        )
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts)
    out.index.names = ["firm_id", "month"]
    return out


def save_panel(panel: pd.DataFrame, path: Path) -> None:
    """Write a returns panel and its source (kept in a ``source`` column)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = panel.assign(
        month=panel["month"].astype(str), source=panel.attrs.get("source", "unknown")
    )
    out.to_parquet(path, index=False)


def load_panel(path: Path) -> pd.DataFrame:
    """Read a panel written by :func:`save_panel`."""
    df = pd.read_parquet(path)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    source = str(df["source"].iloc[0]) if "source" in df and len(df) else "unknown"
    df = df.drop(columns=["source"], errors="ignore")
    df.attrs["source"] = source
    return df
