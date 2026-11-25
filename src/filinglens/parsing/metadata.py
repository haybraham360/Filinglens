"""Fiscal year / period detection for a filing."""

from __future__ import annotations

import re
from datetime import date

_DEI = r'name="dei:{tag}"[^>]*>\s*(?:<[^>]+>\s*)*([^<\s]+)'
_FY = re.compile(_DEI.format(tag="DocumentFiscalYearFocus"), re.I)
_FP = re.compile(_DEI.format(tag="DocumentFiscalPeriodFocus"), re.I)
_VALID_FP = {"FY", "Q1", "Q2", "Q3", "Q4", "H1", "H2"}


def fiscal_from_ixbrl(html: str) -> tuple[int | None, str | None]:
    """Read ``dei:DocumentFiscalYearFocus`` / ``DocumentFiscalPeriodFocus`` from inline XBRL."""
    year: int | None = None
    period: str | None = None
    if m := _FY.search(html):
        digits = re.sub(r"\D", "", m.group(1))
        if len(digits) == 4:
            year = int(digits)
    if (m := _FP.search(html)) and m.group(1).upper() in _VALID_FP:
        period = m.group(1).upper()
    return year, period


def resolve_fiscal(
    html: str,
    form: str,
    period_of_report: date | None,
    facts_hint: tuple[int | None, str | None] = (None, None),
) -> tuple[int | None, str | None]:
    """Fiscal (year, period): inline XBRL first, then companyfacts, then period of report.

    The period-of-report fallback uses the calendar year of the period end, which matches
    how most companies name fiscal years (e.g. a year ending Jan 2024 is FY2024).
    """
    year, period = fiscal_from_ixbrl(html)
    year = year or facts_hint[0]
    period = period or facts_hint[1]
    if year is None and period_of_report is not None:
        year = period_of_report.year
    if period is None and form.upper().startswith("10-K"):
        period = "FY"
    return year, period
