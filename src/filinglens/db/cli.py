"""``filinglens seed-export`` / ``seed-import`` (specs/06)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from filinglens.config import get_settings


def seed_export(
    out: Path = typer.Option(Path("seed.tar.gz"), help="Bundle to write."),
) -> None:
    """Bundle the ingested corpus (tables + parsed sections) for a deployed database."""
    from filinglens.db.connection import connect
    from filinglens.db.seed import export_seed

    settings = get_settings()
    with connect(settings.database_url) as conn:
        manifest = export_seed(conn, settings.data_dir, out)
    typer.echo(json.dumps(manifest, indent=2))
    typer.echo(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def seed_import(
    bundle: Path | None = typer.Argument(None, dir_okay=False, help="seed.tar.gz"),
    url: str | None = typer.Option(None, help="Download the bundle from this URL instead."),
    replace: bool = typer.Option(False, help="Overwrite tables that already have data."),
    if_empty: bool = typer.Option(
        False, help="Always extract data files; load tables only if the database is empty."
    ),
) -> None:
    """Load a seed bundle into DATABASE_URL and DATA_DIR."""
    import tempfile

    import httpx

    from filinglens.db.connection import connect
    from filinglens.db.seed import import_seed

    settings = get_settings()
    if url:
        tmp = Path(tempfile.mkdtemp()) / "seed.tar.gz"
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for part in resp.iter_bytes():
                    fh.write(part)
        bundle = tmp
    if bundle is None or not bundle.exists():
        typer.secho("Give a bundle path or --url.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    with connect(settings.database_url) as conn:
        try:
            manifest = import_seed(
                conn, bundle, settings.data_dir, replace=replace, if_empty=if_empty
            )
        except RuntimeError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {"tables_loaded": manifest["tables_loaded"], "rows": manifest["imported"]}, indent=2
        )
    )
