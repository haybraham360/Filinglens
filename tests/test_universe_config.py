import csv
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_pilot_universe_has_20_unique_tickers() -> None:
    with (CONFIG / "pilot_universe.csv").open() as f:
        tickers = [row["ticker"] for row in csv.DictReader(f)]
    assert len(tickers) == 20
    assert len(set(tickers)) == 20


def test_concept_map_entries_are_nonempty_tag_lists() -> None:
    concept_map = yaml.safe_load((CONFIG / "concept_map.yaml").read_text())
    assert {"revenue", "net_income", "total_assets", "operating_income"} <= concept_map.keys()
    assert all(isinstance(tags, list) and tags for tags in concept_map.values())
