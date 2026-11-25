"""Synthetic companyfacts rows shaped like the real feed, for financials/ratio tests.

Figures are USD millions modelled on reported annual results (Apple FY2021-23, Microsoft
FY2021-23, NVIDIA FY2022-24), entered by hand; they test FilingLens's selection and
arithmetic, not EDGAR. Like real companyfacts, each 10-K repeats two prior years as
comparatives (all tagged with the *filing's* fy), revenue tags differ by company, and
there is one synthetic restatement plus non-annual noise.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from filinglens.ingestion.facts import FACT_COLUMNS

M = 1_000_000
CIKS = {"AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810"}
REVENUE_TAG = {
    "AAPL": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "MSFT": "Revenues",
    "NVDA": "RevenueFromContractWithCustomerExcludingAssessedTax",
}
# fiscal year -> (revenue, gross_profit, operating_income, net_income)
INCOME = {
    "AAPL": {
        2021: (365817, 152836, 108949, 94680),
        2022: (394328, 170782, 119437, 99803),
        2023: (383285, 169148, 114301, 96995),
    },
    "MSFT": {
        2021: (168088, 115856, 69916, 61271),
        2022: (198270, 135620, 83383, 72738),
        2023: (211915, 146052, 88523, 72361),
    },
    "NVDA": {
        2022: (26914, 17475, 10041, 9752),
        2023: (26974, 15356, 4224, 4368),
        2024: (60922, 44301, 32972, 29760),
    },
}
FY_END = {
    "AAPL": {
        2020: date(2020, 9, 26),
        2021: date(2021, 9, 25),
        2022: date(2022, 9, 24),
        2023: date(2023, 9, 30),
    },
    "MSFT": {
        2020: date(2020, 6, 30),
        2021: date(2021, 6, 30),
        2022: date(2022, 6, 30),
        2023: date(2023, 6, 30),
    },
    "NVDA": {
        2021: date(2021, 1, 31),
        2022: date(2022, 1, 30),
        2023: date(2023, 1, 29),
        2024: date(2024, 1, 28),
    },
}
# Apple balance sheet: fiscal year -> (current_assets, current_liabilities, equity)
AAPL_BS = {2022: (135405, 153982, 50672), 2023: (143566, 145308, 62146)}
RESTATED_AAPL_NI_2022 = 99800  # synthetic restatement in the FY2023 10-K


def _row(
    cik: str,
    tag: str,
    value: float,
    start: date | None,
    end: date,
    fy: int,
    fp: str,
    form: str,
    accn: str,
    filed: date,
    unit: str = "USD",
) -> dict[str, object]:
    return {
        "cik": cik,
        "taxonomy": "us-gaap",
        "concept": tag,
        "unit": unit,
        "value": float(value),
        "period_start": start,
        "period_end": end,
        "fy": fy,
        "fp": fp,
        "form": form,
        "accession_no": accn,
        "filed_date": filed,
        "frame": None,
    }


def facts_frame() -> pd.DataFrame:
    """All synthetic facts for AAPL, MSFT and NVDA."""
    rows: list[dict[str, object]] = []
    for ticker, years in INCOME.items():
        cik = CIKS[ticker]
        ends = FY_END[ticker]
        for filing_year in years:
            accn = f"{cik}-{filing_year % 100:02d}-000010"
            filed = ends[filing_year] + timedelta(days=35)
            for y in (filing_year, filing_year - 1, filing_year - 2):
                if y not in years:
                    continue
                rev, gp, oi, ni = years[y]
                if ticker == "AAPL" and filing_year == 2023 and y == 2022:
                    ni = RESTATED_AAPL_NI_2022
                start = ends[y - 1] + timedelta(days=1)
                for tag, val in (
                    (REVENUE_TAG[ticker], rev),
                    ("GrossProfit", gp),
                    ("OperatingIncomeLoss", oi),
                    ("NetIncomeLoss", ni),
                ):
                    rows.append(
                        _row(
                            cik,
                            tag,
                            val * M,
                            start,
                            ends[y],
                            filing_year,
                            "FY",
                            "10-K",
                            accn,
                            filed,
                        )
                    )
            # noise: a 3-month Q4 revenue figure inside the 10-K (not annual)
            rows.append(
                _row(
                    cik,
                    REVENUE_TAG[ticker],
                    1 * M,
                    ends[filing_year] - timedelta(days=90),
                    ends[filing_year],
                    filing_year,
                    "FY",
                    "10-K",
                    accn,
                    filed,
                )
            )
            if ticker == "AAPL":
                for y in (filing_year, filing_year - 1):
                    if y in AAPL_BS:
                        ca, cl, eq = AAPL_BS[y]
                        for tag, val in (
                            ("AssetsCurrent", ca),
                            ("LiabilitiesCurrent", cl),
                            ("StockholdersEquity", eq),
                        ):
                            rows.append(
                                _row(
                                    cik,
                                    tag,
                                    val * M,
                                    None,
                                    ends[y],
                                    filing_year,
                                    "FY",
                                    "10-K",
                                    accn,
                                    filed,
                                )
                            )
    # one Apple 10-Q (Q2 FY2024): own quarter plus a prior-year comparative quarter
    accn = "0000320193-24-000069"
    for start, end, val in (
        (date(2023, 12, 31), date(2024, 3, 30), 90753),
        (date(2023, 1, 1), date(2023, 4, 1), 94836),
    ):
        rows.append(
            _row(
                CIKS["AAPL"],
                REVENUE_TAG["AAPL"],
                val * M,
                start,
                end,
                2024,
                "Q2",
                "10-Q",
                accn,
                date(2024, 5, 3),
            )
        )
    return pd.DataFrame(rows, columns=FACT_COLUMNS)
