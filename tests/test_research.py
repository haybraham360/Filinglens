"""Lazy Prices research pipeline (specs/07): timing, no-lookahead, estimators, parsers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from filinglens.cli import app
from filinglens.llm.provider import LLMResult
from filinglens.parsing.blocks import Block
from filinglens.parsing.chunker import FilingInfo
from filinglens.parsing.pipeline import sections_frame
from filinglens.parsing.sections import Section
from filinglens.research.backtest import BacktestConfig, orient, run_backtest
from filinglens.research.factors import available_models, parse_french_csv, parse_ps_liquidity
from filinglens.research.llm_severity import Severity, SeverityConfig, anonymize, run_llm_severity
from filinglens.research.portfolios import (
    PortfolioSpec,
    assign_quantiles,
    holdings,
    portfolio_returns,
)
from filinglens.research.prices import add_characteristics, facts_asof, load_crsp_csv
from filinglens.research.regressions import factor_regression, fama_macbeth, nw_lags
from filinglens.research.report import render
from filinglens.research.signals import SignalConfig, pair_filings, release_month, run_signals
from filinglens.research.synthetic import make_world
from filinglens.research.text_features import LMDictionary, mentions_officer_change


def _index_row(acc: str, form: str, period: str, accepted: str, **kw: Any) -> dict[str, Any]:
    return {
        "cik": "0000000001",
        "ticker": "ACME",
        "form": form,
        "accession_no": acc,
        "filing_date": pd.Timestamp(accepted[:10]).date(),
        "acceptance_datetime": pd.Timestamp(accepted, tz="UTC"),
        "period_of_report": pd.Timestamp(period).date(),
        "is_amendment": form.endswith("/A"),
        "local_path": f"{acc}.htm",
        **kw,
    }


# --- pairing and timing ---------------------------------------------------------------


def test_pairs_consecutive_originals_and_skips_amendments_and_gaps() -> None:
    index = pd.DataFrame(
        [
            _index_row("k20", "10-K", "2020-12-31", "2021-02-20 21:00"),
            _index_row("k21", "10-K", "2021-12-31", "2022-02-18 21:00"),
            _index_row("k21a", "10-K/A", "2021-12-31", "2022-04-01 21:00"),
            _index_row("k23", "10-K", "2023-12-31", "2024-02-16 21:00"),  # 2022 missing
            _index_row("q21", "10-Q", "2021-03-31", "2021-05-01 20:00"),
            _index_row("q21b", "10-Q", "2021-06-30", "2021-08-01 20:00"),
            _index_row("q22", "10-Q", "2022-03-31", "2022-05-01 20:00"),
        ]
    )
    pairs = pair_filings(index, ("10-K", "10-Q"))
    got = set(zip(pairs["accession_no"], pairs["prev_accession_no"], strict=True))
    # same fiscal quarter a year earlier for 10-Qs; no pair across a missing year
    assert got == {("k21", "k20"), ("q22", "q21")}


def test_release_month_uses_eastern_clock() -> None:
    # 01:30 UTC on April 1 is 9:30 pm March 31 in New York: a March release
    s = pd.Series(
        [pd.Timestamp("2023-04-01 01:30", tz="UTC"), pd.Timestamp("2023-04-01 05:00", tz="UTC")]
    )
    assert release_month(s).astype(str).tolist() == ["2023-03", "2023-04"]


def _section(item: str, blocks: list[tuple[str, str]]) -> Section:
    return Section.from_blocks(item, "x", [Block(k, t) for k, t in blocks], "native")  # type: ignore[arg-type]


RISKS_OLD = [
    ("heading", "Item 1A. Risk Factors"),
    ("heading", "We depend on a limited number of suppliers."),
    (
        "para",
        "Many components come from single sources and supply disruption could harm our results. "
        * 4,
    ),
    ("heading", "We face risks from our former joint venture."),
    (
        "para",
        "Our former joint venture could expose us to indemnification claims about tax matters. "
        * 4,
    ),
]
RISKS_NEW = [
    ("heading", "Item 1A. Risk Factors"),
    ("heading", "We depend on a limited number of suppliers."),
    (
        "para",
        "Many components come from single sources and supply disruption could harm our results. "
        * 4,
    ),
    ("heading", "New export restrictions could limit sales in China."),
    (
        "para",
        "Export restrictions announced in January could prevent sales to customers in China. " * 4,
    ),
    ("heading", "Our new CEO may change strategy."),
    (
        "para",
        "We appointed a new Chief Executive Officer in March who may change our strategy. " * 2,
    ),
]


def _write(processed: Path, acc: str, fy: int, blocks: list[tuple[str, str]]) -> None:
    info = FilingInfo(acc, "0000000001", "ACME", "10-K", fy, "FY")
    path = processed / "sections" / "0000000001" / f"{acc}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    sections_frame(
        [
            _section("1A", blocks),
            _section("7", [("heading", "Item 7. MD&A"), ("para", "Revenue grew. " * 30)]),
        ],
        info,
    ).to_parquet(path)


def test_run_signals_end_to_end_and_incremental(tmp_path: Path) -> None:
    index = pd.DataFrame(
        [
            _index_row("k21", "10-K", "2021-12-31", "2022-02-18 21:00"),
            _index_row("k22", "10-K", "2022-12-31", "2023-02-17 21:00"),
        ]
    )
    (tmp_path / "filings").mkdir()
    index.to_parquet(tmp_path / "filings" / "0000000001.parquet")
    _write(tmp_path, "k21", 2021, RISKS_OLD)
    _write(tmp_path, "k22", 2022, RISKS_NEW)
    lm = LMDictionary.from_words(
        Negative=["disruption", "harm", "claims", "restrictions", "prevent"]
    )
    out = tmp_path / "research" / "signals.parquet"
    table = run_signals(tmp_path, out, SignalConfig(min_words=10, lm=lm))
    rows = table.set_index("section")
    assert set(rows.index) == {"full", "1A", "3", "7"} - {"3"}  # no Item 3 in either filing
    risk = rows.loc["1A"]
    assert risk["n_new_risks"] == 2 and risk["n_removed_risks"] == 1
    assert (
        0 < risk["sim_cosine_tf"] < 1 and 0 < risk["sim_simple"] < 1 and 0 < risk["sim_minedit"] < 1
    )
    assert rows.loc["7", "sim_cosine_tf"] == pytest.approx(1.0)  # MD&A unchanged
    assert rows.loc["full", "ceo_change"] and not rows.loc["full", "cfo_change"]
    assert np.isfinite(risk["lm_neg_delta"])
    assert str(rows.loc["1A", "filing_month"]) == "2023-02"
    again = run_signals(tmp_path, out, SignalConfig(min_words=10))  # cached: nothing new
    assert len(again) == len(table)


# --- portfolio formation: no lookahead ------------------------------------------------


def _signals(values: dict[tuple[str, str], float]) -> pd.DataFrame:
    rows = []
    for (cik, month), v in values.items():
        rows.append(
            {
                "cik": cik,
                "section": "full",
                "filing_month": pd.Period(month, freq="M"),
                "acceptance_datetime": pd.Period(month, freq="M").to_timestamp().tz_localize("UTC"),
                "sim": v,
            }
        )
    return pd.DataFrame(rows)


def test_breakpoints_come_only_from_the_prior_year() -> None:
    prior = {(f"{i:010d}", "2019-03"): i / 100 for i in range(1, 101)}  # uniform 0.01..1.00
    current = {("A", "2020-03"): 0.15, ("B", "2020-03"): 0.95, ("C", "2020-05"): 0.5}
    spec = PortfolioSpec(signal="sim", min_breakpoint_obs=50)
    q = assign_quantiles(_signals(prior | current), spec).set_index("firm_id")["q"]
    assert q["A"] == 1 and q["B"] == 5 and q["C"] == 3
    # wildly different current-year values don't move anyone else's breakpoints
    crazy = current | {(f"X{i}", "2020-04"): 0.0 for i in range(500)}
    q2 = assign_quantiles(_signals(prior | crazy), spec).set_index("firm_id")["q"]
    assert q2["A"] == 1 and q2["B"] == 5 and q2["C"] == 3
    # 2019 filings have no 2018 reference set: unassigned rather than sorted on themselves
    assert not any(k.startswith("00") for k in q.index)


def test_holdings_start_the_month_after_release_and_newest_filing_wins() -> None:
    assigned = pd.DataFrame(
        {
            "firm_id": ["A", "A"],
            "filing_month": pd.PeriodIndex(["2020-03", "2020-05"], freq="M"),
            "acceptance_datetime": pd.to_datetime(["2020-03-10", "2020-05-10"], utc=True),
            "q": [1, 5],
        }
    )
    held = holdings(assigned, 3).set_index("month")["q"]
    assert str(held.index.min()) == "2020-04"  # never the release month itself
    assert held[pd.Period("2020-05", "M")] == 1 and held[pd.Period("2020-06", "M")] == 5
    assert str(held.index.max()) == "2020-08"


def test_value_weights_use_last_months_market_equity() -> None:
    prior = {(f"R{i}", "2019-03"): i / 100 for i in range(1, 101)}
    sig = _signals(prior | {("A", "2020-03"): 0.01, ("B", "2020-03"): 0.02})
    months = pd.period_range("2020-03", "2020-04", freq="M")
    panel = pd.DataFrame(
        {
            "firm_id": ["A", "A", "B", "B"],
            "month": list(months) * 2,
            "ret": [0.0, 0.10, 0.0, -0.10],
            # B is tiny last month but huge this month: VW must use the lagged value
            "me": [300.0, 300.0, 100.0, 1e9],
        }
    )
    panel["month"] = pd.PeriodIndex(panel["month"], freq="M")
    port = portfolio_returns(
        sig,
        add_characteristics(panel),
        PortfolioSpec(signal="sim", weighting="vw", hold_months=1, min_breakpoint_obs=50),
    )
    assert port.loc[pd.Period("2020-04", "M"), "Q1"] == pytest.approx(0.75 * 0.10 + 0.25 * -0.10)


def test_release_month_returns_are_not_captured() -> None:
    """A huge effect in the release month itself must not show up (entry is the month after)."""
    world = make_world(
        n_firms=250, effect=0.0, filing_month_effect=0.05, start="2008-01", end="2016-12"
    )
    port = portfolio_returns(
        world.signals, add_characteristics(world.panel), PortfolioSpec(signal="sim_cosine_tf")
    )
    res = factor_regression(port["LS"], world.factors, "ff3")
    assert res is not None and abs(res.t_alpha) < 2.5 and abs(res.alpha) < 0.004


def test_planted_post_filing_effect_is_recovered() -> None:
    world = make_world(n_firms=250, effect=0.004, start="2008-01", end="2016-12")
    port = portfolio_returns(
        world.signals, add_characteristics(world.panel), PortfolioSpec(signal="sim_cosine_tf")
    )
    res = factor_regression(port["LS"], world.factors, "ff3", hold_months=3)
    assert res is not None and res.alpha > 0.005 and res.t_alpha > 3


# --- estimators ------------------------------------------------------------------------


def test_factor_regression_recovers_alpha_and_beta() -> None:
    rng = np.random.default_rng(0)
    idx = pd.period_range("2000-01", periods=240, freq="M")
    f = pd.DataFrame(
        {
            "MKT": rng.normal(0.006, 0.04, 240),
            "SMB": rng.normal(0, 0.03, 240),
            "HML": rng.normal(0, 0.03, 240),
            "RF": 0.001,
        },
        index=idx,
    )
    y = 0.003 + 1.2 * f["MKT"] + rng.normal(0, 0.01, 240)
    res = factor_regression(y, f, "ff3")
    assert res is not None
    assert res.alpha == pytest.approx(0.003, abs=0.0015) and res.betas["MKT"] == pytest.approx(
        1.2, abs=0.05
    )
    assert res.lags == nw_lags(240) and nw_lags(240, hold_months=12) == 11
    assert factor_regression(y.iloc[:5], f, "ff3") is None


def test_fama_macbeth_recovers_slope() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for m in pd.period_range("2005-01", periods=60, freq="M"):
        x = rng.normal(0, 1, 200)
        rows.append(pd.DataFrame({"month": m, "x": x, "ret": 0.002 * x + rng.normal(0, 0.05, 200)}))
    res = fama_macbeth(pd.concat(rows), "ret", ["x"])
    assert res is not None and res.coef["x"] == pytest.approx(0.002, abs=0.001) and res.t["x"] > 3


def test_characteristics_momentum_and_gaps() -> None:
    months = pd.period_range("2020-01", periods=14, freq="M")
    panel = pd.DataFrame({"firm_id": "A", "month": months, "ret": [0.01] * 14, "me": [100.0] * 14})
    out = add_characteristics(panel).set_index("month")
    assert np.isnan(out.loc[months[11], "ret_12_2"])
    assert out.loc[months[12], "ret_12_2"] == pytest.approx(1.01**11 - 1)  # t-12..t-2
    gappy = panel.drop(index=5)
    out2 = add_characteristics(gappy).set_index("month")
    assert np.isnan(out2.loc[months[6], "me_lag"])  # no silent jump over the missing month
    assert np.isnan(out2.loc[months[13], "ret_12_2"])


def test_facts_asof_is_point_in_time() -> None:
    facts = pd.DataFrame(
        {
            "cik": "1",
            "concept": "StockholdersEquity",
            "value": [100.0, 120.0],
            "period_end": pd.to_datetime(["2020-12-31", "2021-12-31"]),
            "filed_date": pd.to_datetime(["2021-02-20", "2022-02-18"]),
            "form": "10-K",
        }
    )
    months = pd.period_range("2021-01", "2022-04", freq="M")
    s = facts_asof(facts, ("StockholdersEquity",), months, "1")
    assert np.isnan(s[pd.Period("2021-02", "M")])  # filed Feb 20: usable from March
    assert s[pd.Period("2021-03", "M")] == 100 and s[pd.Period("2022-02", "M")] == 100
    assert s[pd.Period("2022-03", "M")] == 120


def test_crsp_loader_compounds_delisting_returns(tmp_path: Path) -> None:
    (tmp_path / "msf.csv").write_text(
        "PERMNO,date,RET,DLRET,PRC,SHROUT,SHRCD\n"
        "10001,2020-01-31,0.05,,-10,1000,11\n"
        "10001,2020-02-28,C,-0.30,10,1000,11\n"
        "10001,2020-03-31,0.10,,10,1000,11\n"
        "10002,2020-01-31,0.02,,20,500,73\n"  # not common stock
    )
    (tmp_path / "link.csv").write_text("cik,permno,start,end\n320193,10001,2019-01-01,2020-02-29\n")
    panel = load_crsp_csv(tmp_path / "msf.csv", tmp_path / "link.csv")
    assert panel.attrs["source"] == "crsp"
    assert panel["firm_id"].unique().tolist() == ["0000320193"]
    assert panel["ret"].tolist() == pytest.approx([0.05, -0.30])  # March is outside the link
    assert panel["me"].iloc[0] == pytest.approx(10 * 1000 * 1000)


# --- factor files ----------------------------------------------------------------------

FRENCH = """This file was created by CMPT_ME_BEME_RETS using the 202409 CRSP database.
The 1-month TBill return is from Ibbotson and Associates, Inc.

,Mkt-RF,SMB,HML,RF
192607,    2.96,   -2.56,   -2.43,    0.22
192608,    2.64,   -1.17,    3.82,    0.25

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
  1927,   29.47,   -2.46,   -3.75,    3.12
"""
PS = """% Pastor-Stambaugh liquidity series
% Month  Agg Liq.  Innov Liq (eq8)  Traded Liq (LIQ_V)
196208  -0.017537   0.004219  -99
196801  -0.020000   0.010000   0.012300
"""


def test_factor_file_parsers() -> None:
    ff = parse_french_csv(FRENCH)
    assert list(ff.columns) == ["MKT", "SMB", "HML", "RF"] and len(ff) == 2
    assert ff.loc[pd.Period("1926-07", "M"), "MKT"] == pytest.approx(0.0296)
    liq = parse_ps_liquidity(PS)
    assert liq.index.astype(str).tolist() == ["1968-01"] and liq.iloc[0] == pytest.approx(0.0123)
    merged = ff.join(pd.Series(0.01, index=ff.index, name="MOM"))
    assert available_models(merged) == ["excess", "capm", "ff3", "carhart"]


# --- text features and LLM safeguards ---------------------------------------------------


def test_officer_change_rule_window() -> None:
    near = [
        "the",
        "board",
        "appointed",
        "jane",
        "doe",
        "as",
        "chief",
        "executive",
        "officer",
        "effective",
        "march",
    ]
    far = ("the ceo spoke " + "word " * 15 + "about new products").split()
    assert mentions_officer_change(near, "ceo") and not mentions_officer_change(near, "cfo")
    assert not mentions_officer_change(far, "ceo")


def test_anonymize_masks_company_ticker_and_people() -> None:
    text = "Apple Inc. (AAPL) said Mr. Timothy Cook and Luca Maestri reviewed Apple's risks."
    out = anonymize(text, company=["Apple Inc."], tickers=["AAPL"], people=["Luca Maestri"])
    assert "Apple" not in out and "AAPL" not in out and "Cook" not in out and "Maestri" not in out
    assert out.count("[COMPANY]") == 2 and "[PERSON]" in out


class _FakeLLM:
    model = "fake"

    def __init__(self) -> None:
        self.users: list[str] = []

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: Any,
        purpose: str,
        prompt_version: str,
        trace_id: str | None = None,
    ) -> LLMResult[Any]:
        self.users.append(user)
        return LLMResult(
            Severity(severity=4, rationale="new export risk"), "fake", "end_turn", 1, 1, 1, 0.0
        )


def test_llm_severity_skips_pre_cutoff_filings(tmp_path: Path) -> None:
    _write(tmp_path, "k21", 2021, RISKS_OLD)
    _write(tmp_path, "k22", 2022, RISKS_NEW)
    pairs = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "ticker": "ACME",
                "accession_no": "k22",
                "prev_accession_no": "k21",
                "acceptance_datetime": pd.Timestamp("2023-02-17", tz="UTC"),
            }
        ]
    )
    llm = _FakeLLM()
    assert run_llm_severity(
        pairs, tmp_path, llm, SeverityConfig(training_cutoff=date(2024, 1, 1))
    ).empty
    assert llm.users == []
    out = run_llm_severity(
        pairs, tmp_path, llm, SeverityConfig(training_cutoff=date(2024, 1, 1), allow_in_sample=True)
    )
    assert out["in_sample"].tolist() == [True] and out["llm_change_severity"].tolist() == [4]
    assert "export restrictions" in llm.users[0].lower() and "ACME" not in llm.users[0]


# --- orchestration and report -----------------------------------------------------------


def test_backtest_logs_every_variant_and_renders(tmp_path: Path) -> None:
    world = make_world(n_firms=150, start="2010-01", end="2018-12")
    cfg = BacktestConfig(
        signals=["sim_cosine_tf", "n_new_risks", "sim_minedit"],
        sections=["full"],
        holds=[3, 6],
        breakpoints=["prior_year"],
        models=["excess", "ff3", "paper5", "ff5_mom"],
        cost_bps=[0, 20],
        size_terciles=False,
    )
    res = run_backtest(
        world.signals, add_characteristics(world.panel), world.factors.drop(columns="LIQ"), cfg
    )
    assert res.meta["missing_models"] == ["paper5"] and res.meta["missing_signals"] == [
        "sim_minedit"
    ]
    v = res.variants
    assert set(v["family"]) == {"main", "holding period", "size", "transaction costs"}
    assert set(v["period"]) == {"full", "pre", "post"}
    net = v[(v["series"] == "LS_net") & (v["model"] == "ff3") & (v["period"] == "full")]
    gross = v[
        (v["series"] == "LS")
        & (v["variant"].isin(net["variant"]))
        & (v["model"] == "ff3")
        & (v["period"] == "full")
    ]
    assert (net["alpha_bps"].to_numpy() < gross["alpha_bps"].to_numpy()).all()
    assert set(res.fama_macbeth["spec"]) == {"sim_cosine_tf", "n_new_risks", "incremental"}
    path = render(res, tmp_path)
    text = path.read_text()
    assert "PIPELINE CHECK" in text and "Post-2014" in text and "variants.csv" in text
    assert (tmp_path / "cumulative_ls.png").stat().st_size > 10_000


def test_change_like_signals_are_flipped_so_q1_means_more_change() -> None:
    df = pd.DataFrame({"n_new_risks": [0.0, 5.0], "sim_jaccard": [0.9, 0.5]})
    out = orient(df, ["n_new_risks", "sim_jaccard"])
    assert out["n_new_risks"].tolist() == [-0.0, -5.0] and out["sim_jaccard"].tolist() == [0.9, 0.5]


def test_research_config_file_loads_and_cli_is_registered() -> None:
    cfg = BacktestConfig.from_yaml(Path("config/research.yaml"))
    assert cfg.main_hold == 3 and cfg.split_month == "2015-01" and "paper5" in cfg.models
    result = CliRunner().invoke(app, ["research", "--help"])
    assert result.exit_code == 0 and "backtest" in result.output
