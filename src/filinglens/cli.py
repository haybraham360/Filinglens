"""Command-line entry point: `filinglens <command>`."""

from __future__ import annotations

import typer

from filinglens import __version__
from filinglens.agent.cli import agent
from filinglens.config import get_settings
from filinglens.db.cli import seed_export, seed_import
from filinglens.ingestion.cli import ingest
from filinglens.parsing.cli import parse, show_section
from filinglens.rag.cli import ask, embed

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


app.command(name="ingest")(ingest)
app.command(name="parse")(parse)
app.command(name="show-section")(show_section)
app.command(name="embed")(embed)
app.command(name="ask")(ask)
app.command(name="agent")(agent)
app.command(name="seed-export")(seed_export)
app.command(name="seed-import")(seed_import)


if __name__ == "__main__":
    app()
