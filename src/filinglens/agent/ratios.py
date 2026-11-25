"""`compute_ratio`: financial ratios in exact decimal arithmetic (specs/04).

Pure functions over already-selected inputs, so they can be checked against hand
calculations. Every result carries its formula and the inputs it used.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, DivisionByZero, InvalidOperation

Inputs = Mapping[str, Decimal]
PLACES = Decimal("0.000001")


@dataclass(frozen=True)
class RatioSpec:
    """How to compute one ratio for fiscal year t."""

    formula: str
    concepts: tuple[str, ...]
    fn: Callable[[Inputs, Inputs | None], Decimal | None]
    needs_prior_year: bool = False


def _div(n: Decimal | None, d: Decimal | None) -> Decimal | None:
    if n is None or d is None or d == 0:
        return None
    try:
        return n / d
    except (DivisionByZero, InvalidOperation):
        return None


def _gross_margin(cur: Inputs, _prior: Inputs | None) -> Decimal | None:
    revenue = cur.get("revenue")
    gross = cur.get("gross_profit")
    if gross is None and revenue is not None and cur.get("cost_of_revenue") is not None:
        gross = revenue - cur["cost_of_revenue"]
    return _div(gross, revenue)


def _simple(num: str, den: str) -> Callable[[Inputs, Inputs | None], Decimal | None]:
    def fn(cur: Inputs, _prior: Inputs | None) -> Decimal | None:
        return _div(cur.get(num), cur.get(den))

    return fn


def _growth(concept: str) -> Callable[[Inputs, Inputs | None], Decimal | None]:
    def fn(cur: Inputs, prior: Inputs | None) -> Decimal | None:
        ratio = _div(cur.get(concept), (prior or {}).get(concept))
        return None if ratio is None else ratio - 1

    return fn


RATIOS: dict[str, RatioSpec] = {
    "gross_margin": RatioSpec(
        "gross_profit / revenue (gross_profit = revenue - cost_of_revenue if not reported)",
        ("gross_profit", "revenue", "cost_of_revenue"),
        _gross_margin,
    ),
    "operating_margin": RatioSpec(
        "operating_income / revenue",
        ("operating_income", "revenue"),
        _simple("operating_income", "revenue"),
    ),
    "net_margin": RatioSpec(
        "net_income / revenue", ("net_income", "revenue"), _simple("net_income", "revenue")
    ),
    "roe": RatioSpec(
        "net_income / stockholders_equity (year-end)",
        ("net_income", "stockholders_equity"),
        _simple("net_income", "stockholders_equity"),
    ),
    "roa": RatioSpec(
        "net_income / total_assets (year-end)",
        ("net_income", "total_assets"),
        _simple("net_income", "total_assets"),
    ),
    "current_ratio": RatioSpec(
        "current_assets / current_liabilities",
        ("current_assets", "current_liabilities"),
        _simple("current_assets", "current_liabilities"),
    ),
    "debt_to_equity": RatioSpec(
        "total_liabilities / stockholders_equity",
        ("total_liabilities", "stockholders_equity"),
        _simple("total_liabilities", "stockholders_equity"),
    ),
    "long_term_debt_to_equity": RatioSpec(
        "long_term_debt / stockholders_equity",
        ("long_term_debt", "stockholders_equity"),
        _simple("long_term_debt", "stockholders_equity"),
    ),
    "operating_cash_flow_margin": RatioSpec(
        "operating_cash_flow / revenue",
        ("operating_cash_flow", "revenue"),
        _simple("operating_cash_flow", "revenue"),
    ),
    "revenue_growth_yoy": RatioSpec(
        "revenue_t / revenue_(t-1) - 1", ("revenue",), _growth("revenue"), needs_prior_year=True
    ),
    "net_income_growth_yoy": RatioSpec(
        "net_income_t / net_income_(t-1) - 1",
        ("net_income",),
        _growth("net_income"),
        needs_prior_year=True,
    ),
}


def compute(
    ratio: str, inputs_by_year: Mapping[int, Inputs], fiscal_years: list[int]
) -> dict[int, Decimal | None]:
    """Ratio per fiscal year, rounded half-even to 6 decimal places (``None`` if inputs miss)."""
    spec = RATIOS[ratio]
    out: dict[int, Decimal | None] = {}
    for year in fiscal_years:
        cur = inputs_by_year.get(year, {})
        prior = inputs_by_year.get(year - 1) if spec.needs_prior_year else None
        value = spec.fn(cur, prior)
        out[year] = None if value is None else value.quantize(PLACES, rounding=ROUND_HALF_EVEN)
    return out
