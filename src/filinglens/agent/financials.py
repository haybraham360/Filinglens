"""`get_financials`: canonical financial concepts from XBRL companyfacts (specs/04).

Selection rules (all deterministic, no LLM):

- **Fiscal year comes from the period end date**, not companyfacts' ``fy`` field, which is
  the fiscal year of the *filing* (a FY2023 10-K also reports FY2021-2022 comparatives
  tagged ``fy=2023``). A period ending in the first week of January belongs to the prior
  fiscal year (52/53-week retailers).
- **Annual** duration facts span 350-380 days; **instant** facts (balance sheet) count only
  on a 10-K balance-sheet date (the latest instant date in some 10-K).
- **Quarterly** facts are 80-100-day durations from 10-Qs, labelled with that 10-Q's
  fiscal year/period, keeping only the quarter the 10-Q itself reports.
- ``as_of="original"`` takes the value as first filed; ``"latest"`` the most recent
  restatement. Both carry the source accession and filed date.
- Each canonical concept tries its XBRL tags in ``config/concept_map.yaml`` order, per year.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd
import psycopg
import yaml

from filinglens.ingestion.facts import FACT_COLUMNS

AsOf = Literal["original", "latest"]
Period = Literal["FY", "Q"]
_UNIT_PREFERENCE = ("USD", "USD/shares", "shares", "pure")


@cache
def load_concept_map(path: Path = Path("config/concept_map.yaml")) -> dict[str, tuple[str, ...]]:
    """Canonical concept -> ordered XBRL tags."""
    raw = yaml.safe_load(path.read_text()) or {}
    return {str(k): tuple(str(t) for t in v) for k, v in raw.items()}


class FactStore(Protocol):
    """Source of raw companyfacts rows."""

    def facts(self, cik: str, tags: Sequence[str]) -> pd.DataFrame:
        """Rows (``FACT_COLUMNS``) for ``cik`` whose concept is one of ``tags``."""
        ...


class PostgresFactStore:
    """Reads the ``financial_facts`` table."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        """Use an open connection."""
        self.conn = conn

    def facts(self, cik: str, tags: Sequence[str]) -> pd.DataFrame:
        """Query facts for one company and a set of tags."""
        rows = self.conn.execute(
            f"SELECT {', '.join(FACT_COLUMNS)} FROM financial_facts "
            "WHERE cik = %s AND concept = ANY(%s)",
            (cik, list(tags)),
        ).fetchall()
        return pd.DataFrame(rows, columns=FACT_COLUMNS)


class FrameFactStore:
    """Facts from a DataFrame (tests, or ingestion's facts parquet without a database)."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Wrap a DataFrame with ``FACT_COLUMNS``."""
        self.df = df

    def facts(self, cik: str, tags: Sequence[str]) -> pd.DataFrame:
        """Filter the frame."""
        mask = (self.df["cik"] == cik) & (self.df["concept"].isin(list(tags)))
        out: pd.DataFrame = self.df[mask].copy()
        return out


@dataclass(frozen=True)
class FactValue:
    """One selected value with its provenance."""

    concept: str
    xbrl_tag: str
    fiscal_year: int
    fiscal_period: str
    period_start: date | None
    period_end: date
    value: Decimal
    unit: str
    accession_no: str
    form: str
    filed_date: date

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (value as a string to keep it exact)."""
        out = asdict(self)
        out["value"] = str(self.value)
        for key in ("period_start", "period_end", "filed_date"):
            out[key] = None if out[key] is None else out[key].isoformat()
        return out


def fiscal_year_of(period_end: date) -> int:
    """Fiscal year a period end belongs to (first week of January -> previous year)."""
    if period_end.month == 1 and period_end.day <= 7:
        return period_end.year - 1
    return period_end.year


def _to_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("period_start", "period_end", "filed_date"):
        df[col] = df[col].map(_to_date)
    df = df[df["period_end"].notna() & df["filed_date"].notna()]
    df["days"] = [
        (e - s).days if s is not None else None
        for s, e in zip(df["period_start"], df["period_end"], strict=True)
    ]
    df["base_form"] = df["form"].fillna("").str.replace("/A", "", regex=False)
    return df


def _pick_unit(rows: pd.DataFrame) -> pd.DataFrame:
    units = list(dict.fromkeys(rows["unit"]))
    chosen = next((u for u in _UNIT_PREFERENCE if u in units), units[0])
    out: pd.DataFrame = rows[rows["unit"] == chosen]
    return out


def _choose(rows: pd.DataFrame, as_of: AsOf) -> pd.Series:
    rows = rows.sort_values(["filed_date", "accession_no"], kind="stable")
    return rows.iloc[0] if as_of == "original" else rows.iloc[-1]


def _annual_candidates(df: pd.DataFrame) -> pd.DataFrame:
    tenk = df[df["base_form"] == "10-K"]
    durations = tenk[tenk["days"].notna() & tenk["days"].between(350, 380)]
    instants = tenk[tenk["period_start"].isna()]
    # Balance-sheet dates: the latest instant reported in each 10-K.
    bs_dates = set(instants.groupby("accession_no")["period_end"].max())
    instants = instants[instants["period_end"].isin(bs_dates)]
    out = pd.concat([durations, instants])
    out["fiscal_year"] = out["period_end"].map(fiscal_year_of)
    out["label_fp"] = "FY"
    return out


def _quarter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    tenq = df[(df["base_form"] == "10-Q") & df["days"].notna() & df["days"].between(80, 100)]
    if tenq.empty:
        return tenq.assign(fiscal_year=[], label_fp=[])
    own_end = tenq.groupby("accession_no")["period_end"].transform("max")
    own = tenq[tenq["period_end"] == own_end].copy()
    own["fiscal_year"] = own["fy"].astype("Int64")
    own["label_fp"] = own["fp"]
    return own[own["fiscal_year"].notna()]


def select_values(
    df: pd.DataFrame,
    concept: str,
    tags: Sequence[str],
    fiscal_years: Sequence[int],
    *,
    period: Period = "FY",
    as_of: AsOf = "original",
) -> list[FactValue]:
    """Pick one value per requested fiscal year (per quarter for ``period="Q"``)."""
    if df.empty:
        return []
    prepared = _prepare(df)
    candidates = _annual_candidates(prepared) if period == "FY" else _quarter_candidates(prepared)
    out: list[FactValue] = []
    for year in fiscal_years:
        year_rows = candidates[candidates["fiscal_year"] == year]
        periods = sorted(set(year_rows["label_fp"])) if period == "Q" else ["FY"]
        for fp in periods:
            for tag in tags:
                rows = year_rows[(year_rows["concept"] == tag) & (year_rows["label_fp"] == fp)]
                if rows.empty:
                    continue
                row = _choose(_pick_unit(rows), as_of)
                out.append(
                    FactValue(
                        concept=concept,
                        xbrl_tag=tag,
                        fiscal_year=int(year),
                        fiscal_period=str(fp),
                        period_start=row["period_start"],
                        period_end=row["period_end"],
                        value=Decimal(str(row["value"])),
                        unit=str(row["unit"]),
                        accession_no=str(row["accession_no"]),
                        form=str(row["form"]),
                        filed_date=row["filed_date"],
                    )
                )
                break
    return out


def get_financials(
    store: FactStore,
    cik: str,
    concepts: Sequence[str],
    fiscal_years: Sequence[int],
    *,
    period: Period = "FY",
    as_of: AsOf = "original",
    concept_map: dict[str, tuple[str, ...]] | None = None,
) -> tuple[list[FactValue], list[str]]:
    """Values for canonical ``concepts``; returns ``(values, missing)`` where ``missing``
    lists ``concept FYyear`` pairs with no data."""
    cmap = concept_map or load_concept_map()
    unknown = [c for c in concepts if c not in cmap]
    if unknown:
        raise ValueError(f"unknown concepts {unknown}; known: {sorted(cmap)}")
    tags = [t for c in concepts for t in cmap[c]]
    df = store.facts(cik, tags)
    values: list[FactValue] = []
    missing: list[str] = []
    for concept in concepts:
        found = select_values(df, concept, cmap[concept], fiscal_years, period=period, as_of=as_of)
        values.extend(found)
        have = {v.fiscal_year for v in found}
        missing.extend(f"{concept} FY{y}" for y in fiscal_years if y not in have)
    return values, missing
