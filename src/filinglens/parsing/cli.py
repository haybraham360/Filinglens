"""``filinglens parse`` and ``filinglens show-section`` (specs/02)."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import pandas as pd
import psycopg
import typer

from filinglens.config import get_settings
from filinglens.parsing.chunker import ChunkConfig
from filinglens.parsing.pipeline import (
    FilingParseResult,
    chunks_path,
    coverage_table,
    quality_frame,
    run_parse,
    sections_path,
)


def _split(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def parse(
    tickers: str | None = typer.Option(None, help="Only these tickers."),
    forms: str | None = typer.Option(None, help="Comma-separated forms. Default: FORMS env."),
    force: bool = typer.Option(False, help="Re-parse HTML even if cached sections exist."),
    db: bool = typer.Option(True, "--db/--no-db", help="Load chunks into Postgres."),
    edgartools: bool = typer.Option(
        True, "--edgartools/--no-edgartools", help="Cross-check and gap-fill with edgartools."
    ),
    strategy: str | None = typer.Option(None, help="structure | fixed. Default: CHUNK_STRATEGY."),
    workers: int = typer.Option(
        max(1, min(8, (os.cpu_count() or 2) - 1)), min=1, help="Parallel parser processes."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Split ingested filings into Items, chunk them, and report extraction coverage."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    chosen = strategy or settings.chunk_strategy
    if chosen not in ("structure", "fixed"):
        raise typer.BadParameter("strategy must be 'structure' or 'fixed'")
    cfg = ChunkConfig(
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_tokens=settings.chunk_min_tokens,
        strategy=chosen,  # type: ignore[arg-type]
    )

    results = run_parse(
        settings.processed_dir,
        cfg,
        tickers=_split(tickers) or None,
        forms=_split(forms) or settings.forms,
        force=force,
        use_edgartools=edgartools,
        workers=workers,
    )
    if not results:
        typer.secho(
            "No ingested filings found. Run `filinglens ingest` first.", fg=typer.colors.YELLOW
        )
        raise typer.Exit(code=1)

    quality = quality_frame(results)
    reports = settings.processed_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    quality.to_csv(reports / "parse_quality.csv", index=False)
    coverage = coverage_table(quality)
    (reports / "parse_coverage.md").write_text(_markdown_table(coverage))

    typer.echo(coverage.to_string(index=False))
    errors = [r for r in results if r.error]
    typer.echo(
        f"\n{len(results)} filings: {sum(r.cached for r in results)} from cache, "
        f"{len(errors)} errors, {sum(r.n_chunks for r in results)} chunks "
        f"({cfg.strategy}, max {cfg.max_tokens} tokens)\nReports: {reports}"
    )
    for r in errors[:10]:
        typer.secho(f"  {r.ticker} {r.accession_no}: {r.error}", fg=typer.colors.RED)

    if db:
        _load_chunks(settings.database_url, settings.processed_dir, results)
    if errors:
        raise typer.Exit(code=1)


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a markdown table (avoids a tabulate dependency)."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def _load_chunks(
    database_url: str, processed_dir: os.PathLike[str] | str, results: Sequence[FilingParseResult]
) -> None:
    from pathlib import Path

    from filinglens.db.repositories import PostgresSink

    try:
        sink = PostgresSink(database_url)
    except psycopg.OperationalError as exc:
        typer.secho(
            f"Cannot connect to Postgres ({exc}). Start it with `make db`, or pass --no-db.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from exc
    written = unchanged = missing = 0
    with sink:
        for r in results:
            if r.error:
                continue
            if not sink.filing_exists(r.accession_no):
                missing += 1
                continue
            df = pd.read_parquet(chunks_path(Path(processed_dir), r.cik, r.accession_no))
            if sink.replace_chunks(r.accession_no, df):
                written += 1
            else:
                unchanged += 1
        total = sink.count("chunks")
    typer.echo(
        f"DB: {written} filings rewritten, {unchanged} unchanged, {total} chunks total"
        + (f", {missing} skipped (filing not in DB; run ingest with --db)" if missing else "")
    )


def show_section(
    accession_no: str = typer.Argument(..., help="e.g. 0000320193-23-000106"),
    item: str = typer.Argument(..., help="e.g. 1A, 7, II-1A"),
    chunks: bool = typer.Option(False, help="Show chunk boundaries instead of the full text."),
) -> None:
    """Print a parsed section, for manual spot-checks of extraction quality."""
    settings = get_settings()
    matches = list((settings.processed_dir / "sections").glob(f"*/{accession_no}.parquet"))
    if not matches:
        typer.secho(f"No parsed sections for {accession_no}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    cik = matches[0].parent.name
    if chunks:
        df = pd.read_parquet(chunks_path(settings.processed_dir, cik, accession_no))
        for row in df[df["item"] == item.upper()].itertuples():
            typer.secho(f"--- {row.chunk_id} ({row.token_count} tokens)", fg=typer.colors.CYAN)
            typer.echo(row.context_header)
            typer.echo(row.text)
        return
    sections = pd.read_parquet(sections_path(settings.processed_dir, cik, accession_no))
    hit = sections[sections["item"] == item.upper()]
    if hit.empty:
        typer.secho(
            f"Item {item} not found. Available: {', '.join(sections['item'])}", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)
    sec = hit.iloc[0].to_dict()
    typer.secho(
        f"{sec['ticker']} {sec['form']} FY{sec['fiscal_year']} Item {sec['item']} "
        f"({sec['char_length']} chars, {sec['source']})",
        fg=typer.colors.CYAN,
    )
    typer.echo(sec["text"])
