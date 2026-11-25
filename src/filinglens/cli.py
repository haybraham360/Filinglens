"""Command-line entry point: `filinglens <command>`."""

from __future__ import annotations

import typer

from filinglens import __version__
from filinglens.config import get_settings

app = typer.Typer(help="FilingLens: an AI analyst for SEC filings.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the installed FilingLens version."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Check that local configuration is complete."""
    settings = get_settings()
    checks = {
        "EDGAR_IDENTITY set": bool(settings.edgar_identity.strip()),
        "EDGAR_MAX_RPS under 10": settings.edgar_max_rps < 10,
        "ANTHROPIC_API_KEY set": settings.anthropic_api_key is not None,
        "Universe file exists": settings.universe_file.exists(),
    }
    for name, ok in checks.items():
        typer.echo(f"{'ok  ' if ok else 'MISS'}  {name}")
    if not all(checks.values()):
        raise typer.Exit(code=1)


@app.command()
def ingest(
    universe: str = typer.Option("pilot", help="Universe name."),
    forms: str = typer.Option("10-K,10-Q", help="Comma-separated form types."),
    since: int = typer.Option(2015, help="First fiscal year to ingest."),
) -> None:
    """Download filings and XBRL facts from SEC EDGAR (see specs/01)."""
    raise NotImplementedError("Ingestion is the Week 1 milestone; see specs/01-data-ingestion.md")


@app.command()
def ask(question: str) -> None:
    """Answer a question about the ingested filings, with citations (see specs/03)."""
    raise NotImplementedError("RAG is the Week 3 milestone; see specs/03-retrieval-and-rag.md")


if __name__ == "__main__":
    app()
