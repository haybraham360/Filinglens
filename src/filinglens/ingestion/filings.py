"""List 10-K/10-Q filings from the submissions API and download their primary documents."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from filinglens.ingestion.client import EdgarClient, EdgarError, EdgarForbiddenError
from filinglens.ingestion.universe import Company, archive_cik

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# EDGAR records acceptance times in US Eastern time. The submissions JSON appends a "Z",
# but the clock value is Eastern (a 18:08 acceptance gets the *next* day's filing date,
# which only makes sense after the 17:30 ET cutoff). Treating it as UTC would shift every
# timestamp 4-5 hours early, a lookahead error in the research phase.
EDGAR_TZ = ZoneInfo("America/New_York")

FILING_COLUMNS = [
    "accession_no",
    "cik",
    "ticker",
    "form",
    "filing_date",
    "acceptance_datetime",
    "period_of_report",
    "primary_doc_url",
    "local_path",
    "is_amendment",
]


@dataclass(frozen=True)
class FilingRecord:
    """One row of the ``filings`` table (see specs/01)."""

    accession_no: str
    cik: str
    ticker: str
    form: str
    filing_date: date
    acceptance_datetime: datetime | None
    period_of_report: date | None
    primary_doc_url: str | None
    local_path: str | None
    is_amendment: bool

    def to_row(self) -> dict[str, Any]:
        """Return the record as a plain dict keyed by column name."""
        return asdict(self)


@dataclass
class DownloadReport:
    """Outcome of downloading primary documents for a set of filings."""

    records: list[FilingRecord]
    downloaded: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


def parse_acceptance(value: str | None) -> datetime | None:
    """Parse an EDGAR ``acceptanceDateTime`` (Eastern clock time) into an aware UTC datetime."""
    if not value:
        return None
    text = value.strip().removesuffix("Z")
    naive = datetime.fromisoformat(text).replace(tzinfo=None)
    return naive.replace(tzinfo=EDGAR_TZ).astimezone(UTC)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value[:10])


def _wanted_forms(forms: Iterable[str], include_amendments: bool) -> set[str]:
    wanted = {f.strip().upper() for f in forms if f.strip()}
    if include_amendments:
        wanted |= {f"{f}/A" for f in wanted if not f.endswith("/A")}
    return wanted


def _iter_columnar(block: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Turn the submissions API's column-oriented arrays into row dicts."""
    accessions = block.get("accessionNumber", [])
    keys = list(block)
    for i in range(len(accessions)):
        yield {k: (block[k][i] if i < len(block[k]) else None) for k in keys}


def primary_doc_url(cik: str, accession_no: str, document: str) -> str:
    """Archive URL for a filing document. Archive paths use the *unpadded* CIK."""
    return ARCHIVE_DOC_URL.format(
        cik=archive_cik(cik), accession=accession_no.replace("-", ""), document=document
    )


def list_filings(
    client: EdgarClient,
    company: Company,
    forms: Sequence[str],
    since: int,
    *,
    include_amendments: bool = True,
    refresh: bool = False,
) -> list[FilingRecord]:
    """List a company's filings of ``forms`` whose period of report is in ``since`` or later.

    Follows the extra ``filings.files`` pages for older history, skipping pages that end
    before ``since``. Amendments (``10-K/A``) are included and flagged when requested.
    """
    wanted = _wanted_forms(forms, include_amendments)
    data = client.get_json(SUBMISSIONS_URL.format(cik=company.cik), refresh=refresh)
    blocks: list[dict[str, list[Any]]] = [data["filings"]["recent"]]
    for page in data["filings"].get("files", []):
        filing_to = _parse_date(page.get("filingTo"))
        if filing_to is not None and filing_to.year < since:
            continue
        url = SUBMISSIONS_PAGE_URL.format(name=page["name"])
        blocks.append(client.get_json(url, refresh=refresh))

    records: dict[str, FilingRecord] = {}
    for block in blocks:
        for row in _iter_columnar(block):
            form = str(row.get("form") or "").upper()
            if form not in wanted:
                continue
            filing_date = _parse_date(row.get("filingDate"))
            if filing_date is None:
                continue
            period = _parse_date(row.get("reportDate"))
            if (period or filing_date).year < since:
                continue
            accession = str(row["accessionNumber"])
            document = (row.get("primaryDocument") or "").strip()
            records[accession] = FilingRecord(
                accession_no=accession,
                cik=company.cik,
                ticker=company.ticker,
                form=form,
                filing_date=filing_date,
                acceptance_datetime=parse_acceptance(row.get("acceptanceDateTime")),
                period_of_report=period,
                primary_doc_url=primary_doc_url(company.cik, accession, document)
                if document
                else None,
                local_path=None,
                is_amendment=form.endswith("/A"),
            )
    return sorted(records.values(), key=lambda r: (r.filing_date, r.accession_no))


def local_doc_path(raw_dir: Path, record: FilingRecord) -> Path | None:
    """Where a filing's primary document is stored: ``raw/filings/<cik>/<accession>/<doc>``."""
    if record.primary_doc_url is None:
        return None
    document = Path(record.primary_doc_url.rsplit("/", 1)[-1]).name
    return raw_dir / "filings" / record.cik / record.accession_no / document


def download_filings(
    client: EdgarClient,
    records: Sequence[FilingRecord],
    raw_dir: Path,
    *,
    workers: int = 4,
) -> DownloadReport:
    """Download each primary document unless it is already on disk.

    Workers share the client's global rate limiter, so concurrency hides latency without
    raising the request rate. A failed document is recorded and the rest continue; a 403
    aborts the whole run because it means SEC is rejecting us.
    """
    report = DownloadReport(records=[])

    def fetch(record: FilingRecord) -> tuple[FilingRecord, bool | None, str | None]:
        dest = local_doc_path(raw_dir, record)
        if dest is None or record.primary_doc_url is None:
            return record, None, "no primary document listed"
        try:
            downloaded = client.download(record.primary_doc_url, dest)
        except EdgarForbiddenError:
            raise
        except EdgarError as exc:
            return record, None, str(exc)
        return replace(record, local_path=dest.as_posix()), downloaded, None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for record, downloaded, error in pool.map(fetch, records):
            report.records.append(record)
            if error is not None:
                logger.warning("failed %s: %s", record.accession_no, error)
                report.failed.append((record.accession_no, error))
            elif downloaded:
                report.downloaded += 1
            else:
                report.skipped += 1
    return report
