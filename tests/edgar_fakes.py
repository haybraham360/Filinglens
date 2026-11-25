"""Canned SEC responses and helpers shared by the ingestion tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import respx
from httpx import Response

from filinglens.ingestion.client import EdgarClient
from filinglens.ingestion.ratelimit import RateLimiter

IDENTITY = "FilingLens Tests tests@example.com"
AAPL_CIK = "0000320193"

TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "2": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
}


def _columnar(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    keys = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
    ]
    return {k: [r.get(k, "") for r in rows] for k in keys}


SUBMISSIONS_JSON: dict[str, Any] = {
    "cik": "320193",
    "filings": {
        "recent": _columnar(
            [
                {
                    "accessionNumber": "0000320193-23-000106",
                    "filingDate": "2023-11-03",
                    "reportDate": "2023-09-30",
                    "acceptanceDateTime": "2023-11-02T18:08:27.000Z",
                    "form": "10-K",
                    "primaryDocument": "aapl-20230930.htm",
                },
                {
                    "accessionNumber": "0000320193-23-000077",
                    "filingDate": "2023-08-04",
                    "reportDate": "2023-07-01",
                    "acceptanceDateTime": "2023-08-03T18:04:43.000Z",
                    "form": "10-Q",
                    "primaryDocument": "aapl-20230701.htm",
                },
                {
                    "accessionNumber": "0001140361-23-049360",
                    "filingDate": "2023-10-30",
                    "reportDate": "",
                    "acceptanceDateTime": "2023-10-30T16:30:00.000Z",
                    "form": "4",
                    "primaryDocument": "form4.xml",
                },
                {
                    "accessionNumber": "0000320193-23-000110",
                    "filingDate": "2023-12-01",
                    "reportDate": "2023-09-30",
                    "acceptanceDateTime": "2023-12-01T09:00:00.000Z",
                    "form": "10-K/A",
                    "primaryDocument": "aapl-10ka.htm",
                },
            ]
        ),
        "files": [
            {
                "name": "CIK0000320193-submissions-001.json",
                "filingFrom": "2015-01-01",
                "filingTo": "2016-12-31",
            },
            {
                "name": "CIK0000320193-submissions-002.json",
                "filingFrom": "1994-01-01",
                "filingTo": "2001-12-31",
            },
        ],
    },
}

SUBMISSIONS_PAGE_JSON = _columnar(
    [
        {
            "accessionNumber": "0001628280-16-020309",
            "filingDate": "2016-10-26",
            "reportDate": "2016-09-24",
            "acceptanceDateTime": "2016-10-26T16:42:16.000Z",
            "form": "10-K",
            "primaryDocument": "a201610-k9242016.htm",
        },
        {
            "accessionNumber": "0001193125-15-356351",
            "filingDate": "2015-10-28",
            "reportDate": "2015-09-26",
            "acceptanceDateTime": "2015-10-28T16:31:09.000Z",
            "form": "10-K",
            "primaryDocument": "d17062d10k.htm",
        },
    ]
)

COMPANYFACTS_JSON: dict[str, Any] = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {
                    "shares": [
                        {
                            "end": "2023-10-20",
                            "val": 15550061000,
                            "accn": "0000320193-23-000106",
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2023-11-03",
                        }
                    ]
                }
            }
        },
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2022-09-25",
                            "end": "2023-09-30",
                            "val": 383285000000,
                            "accn": "0000320193-23-000106",
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2023-11-03",
                            "frame": "CY2023",
                        },
                        {
                            "start": "2022-09-25",
                            "end": "2023-09-30",
                            "val": 383285000000,
                            "accn": "0000320193-24-000123",
                            "fy": 2024,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2024-11-01",
                        },
                    ]
                }
            },
            "EarningsPerShareBasic": {
                "units": {
                    "USD/shares": [
                        {
                            "start": "2022-09-25",
                            "end": "2023-09-30",
                            "val": 6.16,
                            "accn": "0000320193-23-000106",
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2023-11-03",
                        }
                    ]
                }
            },
        },
    },
}


def make_client(tmp_path: Path, **kwargs: Any) -> EdgarClient:
    """A client with a fast limiter and no real sleeping."""
    kwargs.setdefault("rate_limiter", RateLimiter(10_000))
    kwargs.setdefault("sleep", lambda _s: None)
    return EdgarClient(IDENTITY, tmp_path / "cache", **kwargs)


def mount_sec(router: respx.MockRouter) -> None:
    """Register canned responses for one company (AAPL) on ``router``."""
    router.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=Response(200, json=TICKERS_JSON)
    )
    router.get(f"https://data.sec.gov/submissions/CIK{AAPL_CIK}.json").mock(
        return_value=Response(200, json=SUBMISSIONS_JSON)
    )
    router.get("https://data.sec.gov/submissions/CIK0000320193-submissions-001.json").mock(
        return_value=Response(200, json=SUBMISSIONS_PAGE_JSON)
    )
    router.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{AAPL_CIK}.json").mock(
        return_value=Response(200, json=COMPANYFACTS_JSON)
    )
    router.get(url__startswith="https://www.sec.gov/Archives/edgar/data/320193/").mock(
        side_effect=lambda request: Response(200, content=f"<html>{request.url}</html>".encode())
    )
