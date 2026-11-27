"""Monthly factor returns: Fama-French (Kenneth French Data Library) and Pastor-Stambaugh.

Kenneth French publishes zipped CSVs with no key. Each has a text preamble, then a header
row starting with a comma (``,Mkt-RF,SMB,HML,RF``), then ``YYYYMM`` rows in percent. An
annual block follows after a blank line, and that is where parsing stops. The momentum file
has the same layout with one ``Mom`` column.

The paper's "5-factor" model is market, size, value, momentum *and liquidity* (Table II
caption), not Fama-French's 2015 five factors. For liquidity we use the Pastor-Stambaugh
traded factor (``LIQ_V``). Its text file has ``%`` comment lines, then rows of month, level,
innovation and traded factor, with ``-99`` for missing values. The file name changes every
year, so it is passed in as a URL or path.

Downloaded files are cached under ``<cache_dir>/factors``.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

import httpx
import pandas as pd

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
}

# Factor sets per model, all in decimal monthly returns. "paper5" is the paper's model.
MODELS: dict[str, tuple[str, ...]] = {
    "excess": (),
    "capm": ("MKT",),
    "ff3": ("MKT", "SMB", "HML"),
    "carhart": ("MKT", "SMB", "HML", "MOM"),
    "paper5": ("MKT", "SMB", "HML", "MOM", "LIQ"),
    "ff5": ("MKT", "SMB", "HML", "RMW", "CMA"),
    "ff5_mom": ("MKT", "SMB", "HML", "RMW", "CMA", "MOM"),
}

_MONTH_ROW = re.compile(r"^\s*(\d{6})\s*,")
_RENAME = {"Mkt-RF": "MKT", "Mom": "MOM", "UMD": "MOM"}


def parse_french_csv(text: str) -> pd.DataFrame:
    """Monthly block of a Ken French CSV as decimals, indexed by month Period."""
    lines = text.splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith(",") and i + 1 < len(lines) and _MONTH_ROW.match(lines[i + 1])
        ),
        None,
    )
    if start is None:
        raise ValueError("No monthly table found in the factor file.")
    cols = [c.strip() for c in lines[start].split(",")[1:]]
    records = []
    for line in lines[start + 1 :]:
        m = _MONTH_ROW.match(line)
        if not m:
            break
        values = [float(v) for v in line.split(",")[1 : len(cols) + 1]]
        records.append([m.group(1), *values])
    df = pd.DataFrame(records, columns=["month", *cols])
    df["month"] = pd.PeriodIndex(pd.to_datetime(df["month"], format="%Y%m"), freq="M")
    df = df.set_index("month").rename(columns=lambda c: _RENAME.get(c, c))
    return df / 100.0


def parse_ps_liquidity(text: str) -> pd.Series:
    """Pastor-Stambaugh traded liquidity factor (last numeric column), decimals by month."""
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%") or not line[:6].isdigit():
            continue
        parts = re.split(r"[\s,]+", line)
        month, traded = parts[0], float(parts[-1])
        if traded <= -99:  # missing (the traded factor starts in 1968)
            continue
        records.append((month[:6], traded))
    if not records:
        raise ValueError("No liquidity rows found.")
    s = pd.Series(
        [v for _, v in records],
        index=pd.PeriodIndex(pd.to_datetime([m for m, _ in records], format="%Y%m"), freq="M"),
        name="LIQ",
    )
    return s


def _fetch(url: str, cache_dir: Path, client: httpx.Client | None = None) -> bytes:
    path = cache_dir / "factors" / url.rsplit("/", 1)[-1]
    if path.exists():
        return path.read_bytes()
    own = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True)
    try:
        resp = client.get(url)
        resp.raise_for_status()
    finally:
        if own:
            client.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return resp.content


def _unzip_text(blob: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith((".csv", ".txt")))
        return zf.read(name).decode("latin-1")


def load_factors(
    cache_dir: Path,
    *,
    sets: Iterable[str] = ("ff3", "ff5", "mom"),
    liquidity: str | Path | None = None,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Merged monthly factors (MKT, SMB, HML, RF, RMW, CMA, MOM, LIQ when available).

    ``liquidity`` is a URL or local path to the Pastor-Stambaugh file. When it is omitted,
    the ``paper5`` model can't run and the report says so.
    """
    frames = []
    for key in sets:
        frames.append(
            parse_french_csv(
                _unzip_text(_fetch(FRENCH_BASE + FRENCH_FILES[key], cache_dir, client))
            )
        )
    out = frames[0]
    for f in frames[1:]:
        out = out.combine_first(f)  # FF5's MKT/SMB/HML differ slightly; FF3's take priority
    if liquidity is not None:
        src = str(liquidity)
        text = (
            _fetch(src, cache_dir, client).decode("latin-1")
            if src.startswith(("http://", "https://"))
            else Path(src).read_text(encoding="latin-1")
        )
        out = out.join(parse_ps_liquidity(text), how="left")
    return out.sort_index()


def available_models(factors: pd.DataFrame) -> list[str]:
    """Models whose factor columns are all present (with data) in ``factors``."""
    have = {c for c in factors.columns if factors[c].notna().any()}
    return [m for m, cols in MODELS.items() if set(cols) <= have]
