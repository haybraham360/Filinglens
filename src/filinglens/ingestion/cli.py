"""``filinglens ingest``: download filings and XBRL facts from SEC EDGAR (specs/01)."""

from __future__ import annotations

import logging
from contextlib import ExitStack

import psycopg
import typer

from filinglens.config import get_settings
from filinglens.db.repositories import PostgresSink
from filinglens.ingestion.client import EdgarClient, EdgarForbiddenError
from filinglens.ingestion.pipeline import IngestSummary, run_ingest
from filinglens.ingestion.universe import resolve_universe_file


def _split(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def ingest(
    universe: str = typer.Option(
        "pilot", help="Universe name (config/<name>_universe.csv) or a CSV path."
    ),
    forms: str | None = typer.Option(None, help="Comma-separated forms. Default: FORMS env."),
    since: int | None = typer.Option(
        None, help="First fiscal year (period of report). Default: START_YEAR env."
    ),
    tickers: str | None = typer.Option(None, help="Only these tickers from the universe."),
    db: bool = typer.Option(True, "--db/--no-db", help="Load filings and facts into Postgres."),
    refresh: bool = typer.Option(
        False, help="Re-fetch submissions/companyfacts JSON to pick up new filings."
    ),
    amendments: bool = typer.Option(
        True, "--amendments/--no-amendments", help="Include and flag 10-K/A, 10-Q/A."
    ),
    workers: int = typer.Option(4, min=1, max=8, help="Parallel document downloads."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download 10-K/10-Q filings and XBRL facts from SEC EDGAR, cached and idempotent."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("filinglens").setLevel(logging.INFO)
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    form_list = _split(forms) or settings.forms
    first_year = since if since is not None else settings.start_year
    universe_file = resolve_universe_file(universe, settings.universe_file.parent)

    try:
        settings.require_edgar_identity()
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    with ExitStack() as stack:
        sink: PostgresSink | None = None
        if db:
            try:
                sink = stack.enter_context(PostgresSink(settings.database_url))
            except psycopg.OperationalError as exc:
                typer.secho(
                    f"Cannot connect to Postgres ({exc}).\nStart it with `make db`, "
                    "or pass --no-db to write parquet only.",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=2) from exc

        client = stack.enter_context(EdgarClient.from_settings(settings))
        try:
            summary = run_ingest(
                client,
                universe_file,
                forms=form_list,
                since=first_year,
                raw_dir=settings.raw_dir,
                processed_dir=settings.processed_dir,
                sink=sink,
                tickers=_split(tickers) or None,
                workers=workers,
                refresh=refresh,
                include_amendments=amendments,
            )
        except EdgarForbiddenError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=3) from exc

        _print_summary(summary)
        if sink is not None:
            typer.echo(
                f"DB rows: filings={sink.count('filings')}, "
                f"financial_facts={sink.count('financial_facts')}"
            )

    if not summary.ok:
        raise typer.Exit(code=1)


def _print_summary(summary: IngestSummary) -> None:
    typer.echo(
        f"{'ticker':<7} {'cik':<11} {'filings':>7} {'new':>5} {'cached':>6} "
        f"{'failed':>6} {'facts':>8}"
    )
    for c in summary.companies:
        line = (
            f"{c.ticker:<7} {c.cik:<11} {c.filings:>7} {c.downloaded:>5} {c.skipped:>6} "
            f"{c.failed_docs:>6} {c.fact_rows:>8}"
        )
        typer.echo(line + (f"  ERROR: {c.error}" if c.error else ""))
    typer.echo(
        f"Total: {summary.total('filings')} filings, {summary.total('downloaded')} new docs, "
        f"{summary.total('fact_rows')} fact rows | {summary.network_requests} requests, "
        f"{summary.cache_hits} cache hits, {summary.bytes_downloaded / 1e6:.1f} MB"
    )
    if summary.unresolved_tickers:
        typer.secho(
            f"Unresolved tickers: {', '.join(summary.unresolved_tickers)}", fg=typer.colors.YELLOW
        )
