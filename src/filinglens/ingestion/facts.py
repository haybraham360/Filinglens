"""Download XBRL companyfacts JSON and flatten it into a long table."""

from __future__ import annotations

from typing import Any

import pandas as pd

from filinglens.ingestion.client import EdgarClient, EdgarNotFoundError

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

FACT_COLUMNS = [
    "cik",
    "taxonomy",
    "concept",
    "unit",
    "value",
    "period_start",
    "period_end",
    "fy",
    "fp",
    "form",
    "accession_no",
    "filed_date",
    "frame",
]


def fetch_company_facts(
    client: EdgarClient, cik: str, *, refresh: bool = False
) -> dict[str, Any] | None:
    """Return the companyfacts payload, or ``None`` if the company has no XBRL facts (404)."""
    try:
        payload: dict[str, Any] = client.get_json(COMPANYFACTS_URL.format(cik=cik), refresh=refresh)
    except EdgarNotFoundError:
        return None
    return payload


def flatten_company_facts(payload: dict[str, Any], cik: str) -> pd.DataFrame:
    """Flatten ``facts -> taxonomy -> concept -> units -> unit -> [obs]`` into one row per obs.

    ``filed_date`` is kept on every row: the same period is often reported again (and
    sometimes restated) in later filings, and point-in-time queries depend on it.
    """
    rows: list[tuple[Any, ...]] = []
    for taxonomy, concepts in payload.get("facts", {}).items():
        for concept, body in concepts.items():
            for unit, observations in body.get("units", {}).items():
                for obs in observations:
                    if obs.get("val") is None or not obs.get("end") or not obs.get("filed"):
                        continue
                    rows.append(
                        (
                            cik,
                            taxonomy,
                            concept,
                            unit,
                            obs["val"],
                            obs.get("start"),
                            obs["end"],
                            obs.get("fy"),
                            obs.get("fp"),
                            obs.get("form"),
                            obs.get("accn"),
                            obs["filed"],
                            obs.get("frame"),
                        )
                    )
    df = pd.DataFrame(rows, columns=FACT_COLUMNS)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    df = df.dropna(subset=["value"])
    for col in ("period_start", "period_end", "filed_date"):
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce").dt.date
    df["fy"] = df["fy"].astype("Int64")
    return df.reset_index(drop=True)
