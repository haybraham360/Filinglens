"""Load a ticker universe and map tickers to zero-padded CIKs."""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from filinglens.ingestion.cache import atomic_write_bytes
from filinglens.ingestion.client import EdgarClient

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


@dataclass(frozen=True)
class Company:
    """A company in the universe. ``cik`` is the stable key; ``ticker`` is as of ingestion."""

    ticker: str
    cik: str
    name: str
    sector: str | None = None


def pad_cik(cik: int | str) -> str:
    """Zero-pad a CIK to 10 digits, as data.sec.gov requires."""
    return f"{int(cik):010d}"


def archive_cik(cik: int | str) -> str:
    """CIK without leading zeros, as used in ``/Archives/edgar/data/{cik}/`` paths."""
    return str(int(cik))


def normalize_ticker(ticker: str) -> str:
    """Upper-case and use SEC's class separator (``BRK.B`` -> ``BRK-B``)."""
    return ticker.strip().upper().replace(".", "-")


def resolve_universe_file(universe: str, config_dir: Path = Path("config")) -> Path:
    """Accept a CSV path, or a short name like ``pilot`` meaning ``config/pilot_universe.csv``."""
    candidate = Path(universe)
    if candidate.suffix == ".csv" or candidate.exists():
        return candidate
    return config_dir / f"{universe}_universe.csv"


def load_universe_file(path: Path) -> list[tuple[str, str | None]]:
    """Read ``(ticker, sector)`` pairs from a CSV with a ``ticker`` column."""
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "ticker" not in reader.fieldnames:
            raise ValueError(f"{path} must have a 'ticker' column")
        rows = [
            (normalize_ticker(row["ticker"]), (row.get("sector") or "").strip() or None)
            for row in reader
            if row["ticker"].strip()
        ]
    return rows


def fetch_ticker_map(client: EdgarClient, *, refresh: bool = False) -> dict[str, tuple[str, str]]:
    """Return ``{ticker: (cik10, company_name)}`` from SEC's ``company_tickers.json``."""
    payload = client.get_json(TICKERS_URL, refresh=refresh)
    mapping: dict[str, tuple[str, str]] = {}
    for entry in payload.values():
        mapping[normalize_ticker(str(entry["ticker"]))] = (
            pad_cik(entry["cik_str"]),
            str(entry["title"]),
        )
    return mapping


def resolve_universe(
    client: EdgarClient,
    universe_file: Path,
    *,
    snapshot_dir: Path | None = None,
    refresh: bool = False,
) -> tuple[list[Company], list[str]]:
    """Map every ticker in ``universe_file`` to a :class:`Company`.

    Returns ``(companies, unresolved_tickers)``. If ``snapshot_dir`` is given, the resolved
    mapping is written there as ``<universe>_ciks.csv`` so later stages can read it offline.
    """
    ticker_map = fetch_ticker_map(client, refresh=refresh)
    companies: list[Company] = []
    unresolved: list[str] = []
    for ticker, sector in load_universe_file(universe_file):
        hit = ticker_map.get(ticker)
        if hit is None:
            logger.warning("ticker %s not found in company_tickers.json", ticker)
            unresolved.append(ticker)
            continue
        companies.append(Company(ticker=ticker, cik=hit[0], name=hit[1], sector=sector))

    if snapshot_dir is not None:
        name = universe_file.stem.removesuffix("_universe")
        _write_snapshot(snapshot_dir / f"{name}_ciks.csv", companies)
    return companies, unresolved


def _write_snapshot(path: Path, companies: list[Company]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["ticker", "cik", "name", "sector"])
    writer.writeheader()
    writer.writerows(asdict(company) for company in companies)
    atomic_write_bytes(path, buffer.getvalue().encode())
