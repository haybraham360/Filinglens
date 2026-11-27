"""Files-only seed extraction (used when building the Vercel image; no database)."""

import io
import json
import tarfile
from pathlib import Path

from typer.testing import CliRunner

from filinglens.cli import app
from filinglens.db.seed import extract_files


def _bundle(path: Path) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in {
            "manifest.json": json.dumps({"filinglens": "test"}).encode(),
            "data/processed/sections/0000320193/a.parquet": b"PAR1",
            "db/chunks.copy": b"",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_extract_files_unpacks_only_data(tmp_path: Path) -> None:
    _bundle(tmp_path / "seed.tar.gz")
    manifest = extract_files(tmp_path / "seed.tar.gz", tmp_path / "out")
    assert manifest["filinglens"] == "test"
    assert (tmp_path / "out/processed/sections/0000320193/a.parquet").read_bytes() == b"PAR1"
    assert not (tmp_path / "out/db").exists()


def test_seed_import_files_only_needs_no_database(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _bundle(tmp_path / "seed.tar.gz")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
    from filinglens.config import get_settings

    get_settings.cache_clear()
    try:
        result = CliRunner().invoke(
            app, ["seed-import", str(tmp_path / "seed.tar.gz"), "--files-only"]
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    assert (tmp_path / "data/processed/sections/0000320193/a.parquet").exists()
