"""Draft *candidate* golden questions from the ingested corpus (specs/05).

    python -m evals.draft --n 80

Drafts are unverified: nothing here goes into ``golden_v1.jsonl`` until a person checks it
with ``python -m evals.verify``. Per type:

- factual: Claude writes a question answerable from one sampled chunk (Item 1, 1A, 7).
- numeric: templated from XBRL via ``get_financials`` / ``compute_ratio``; the reviewer
  must check the value against the filing, since the tool under test produced it.
- yoy_change: risk-factor headings that ``compare_sections`` finds added between years.
- multi_hop: a ratio change plus Claude's summary of retrieved MD&A chunks.
- unanswerable: templated questions the filings can't answer (company outside the
  universe, future fiscal years, information 10-Ks don't contain).

Requires ingested + parsed + embedded data, and an Anthropic API key (paid calls for the
factual and multi-hop drafts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from typing import Any

from pydantic import BaseModel, Field

from evals.dataset import TARGET_MIX, Case, GoldSpan, dataset_path, read_jsonl, write_jsonl
from filinglens.llm.prompts import load_prompt

logger = logging.getLogger("evals.draft")


class DraftQA(BaseModel):
    """Drafter output for a factual question."""

    usable: bool
    question: str
    answer: str
    evidence_quote: str = Field(description="Exact span from the passage.")


class Drivers(BaseModel):
    """Drafter output for the explanation half of a multi-hop question."""

    drivers: str


NUMERIC_TEMPLATES = [
    (
        "get_financials",
        "revenue",
        "total revenue",
        "{company}'s total revenue in fiscal {fy} was {value}.",
    ),
    (
        "get_financials",
        "net_income",
        "net income",
        "{company}'s net income in fiscal {fy} was {value}.",
    ),
    (
        "compute_ratio",
        "gross_margin",
        "gross margin",
        "{company}'s gross margin in fiscal {fy} was {value}.",
    ),
    (
        "compute_ratio",
        "operating_margin",
        "operating margin",
        "{company}'s operating margin in fiscal {fy} was {value}.",
    ),
    (
        "compute_ratio",
        "net_margin",
        "net profit margin",
        "{company}'s net profit margin in fiscal {fy} was {value}.",
    ),
    (
        "compute_ratio",
        "revenue_growth_yoy",
        "year-over-year revenue growth",
        "{company}'s revenue grew {value} year over year in fiscal {fy}.",
    ),
]
UNANSWERABLE_TEMPLATES = [
    "What were Costco's main risk factors in its fiscal {fy} 10-K?",
    "What revenue guidance did {company} give for fiscal {future}?",
    "What does {company}'s fiscal {future} 10-K say about supply chain risk?",
    "What is {company}'s internal target for employee attrition in fiscal {fy}?",
    "How much did {company} pay its outside counsel for the {fy} annual report?",
    "What will {company}'s stock price be at the end of {future}?",
]


def _id(kind: str, *parts: Any) -> str:
    digest = hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:8]
    return f"{kind[:3]}_{digest}"


def _fmt_value(tool: str, key: str, value: float) -> str:
    if tool == "compute_ratio":
        return f"{value * 100:.1f}%"
    return f"${value / 1e9:,.2f} billion" if abs(value) >= 1e9 else f"${value / 1e6:,.1f} million"


class Drafter:
    """Generates candidate cases against the ingested corpus."""

    def __init__(self, conn: Any, client: Any, model: str, settings: Any, seed: int) -> None:
        """Wire DB, Claude and the agent's deterministic tools."""
        from filinglens.agent.diff import SectionStore
        from filinglens.agent.financials import PostgresFactStore
        from filinglens.rag.cli import build_retriever, retrieval_config

        self.conn, self.client, self.model, self.settings = conn, client, model, settings
        self.rng = random.Random(seed)
        self.retriever = build_retriever(settings, conn, retrieval_config(settings))
        self.sections = SectionStore(settings.processed_dir)
        self._ctx_args = (
            PostgresFactStore(conn),
            str(settings.universe_file.parent / "concept_map.yaml"),
        )
        rows = conn.execute(
            "SELECT DISTINCT ticker, cik, fiscal_year FROM chunks WHERE form = '10-K' "
            "AND fiscal_year IS NOT NULL ORDER BY ticker, fiscal_year"
        ).fetchall()
        self.filings = [(t, c, int(y)) for t, c, y in rows]
        self.names = self._company_names(settings)

    def _ctx(self) -> Any:
        """Fresh tool context (own source registry) for one draft."""
        from filinglens.agent.sources import SourceRegistry
        from filinglens.agent.tools import AgentContext

        facts, concept_map = self._ctx_args
        return AgentContext(
            retriever=self.retriever,
            sections=self.sections,
            facts=facts,
            directory=self.retriever.directory,
            registry=SourceRegistry(),
            embedder=self.retriever.embedder,
            concept_map_path=concept_map,
        )

    @staticmethod
    def _company_names(settings: Any) -> dict[str, str]:
        """Ticker -> SEC company name from ingestion's universe snapshot."""
        import csv

        universe = settings.universe_file.stem.removesuffix("_universe")
        path = settings.raw_dir / "universe" / f"{universe}_ciks.csv"
        if not path.exists():
            return {}
        with path.open(newline="") as fh:
            return {r["ticker"]: r["name"] for r in csv.DictReader(fh)}

    def _parse(self, system: str, user: str, schema: type[BaseModel]) -> Any:
        from filinglens.llm.provider import FALLBACK_BETA, FALLBACK_MODELS

        kwargs: dict[str, Any] = {}
        if self.model in FALLBACK_MODELS:
            kwargs = {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        resp = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": "medium"},
            **kwargs,
        )
        return resp.parsed_output

    def factual(self, n: int) -> list[Case]:
        """Questions answerable from one chunk."""
        rows = self.conn.execute(
            "SELECT chunk_id, ticker, fiscal_year, item, text, accession_no, char_start, char_end, "
            "context_header FROM chunks WHERE form = '10-K' AND item IN ('1', '1A', '7') "
            "AND token_count > 120 ORDER BY md5(chunk_id || %s) LIMIT %s",
            (str(self.rng.random()), n * 3),
        ).fetchall()
        out: list[Case] = []
        for cid, ticker, fy, item, text, acc, start, end, header in rows:
            if len(out) >= n:
                break
            qa = self._parse(
                load_prompt("draft_factual").text,
                f"{header}\n\n<passage>\n{text}\n</passage>",
                DraftQA,
            )
            if qa is None or not qa.usable:
                continue
            out.append(
                Case(
                    id=_id("factual", cid),
                    question=qa.question,
                    ticker=ticker,
                    fiscal_year=fy,
                    type="factual",
                    gold_answer=qa.answer,
                    gold_chunk_ids=[cid],
                    gold_spans=[
                        GoldSpan(accession_no=acc, item=item, char_start=start, char_end=end)
                    ],
                    expected_tools=["search_filings"],
                    source="draft:factual",
                    notes=f"evidence quote: {qa.evidence_quote!r}",
                )
            )
        return out

    def numeric(self, n: int) -> list[Case]:
        """Templated figures from XBRL (reviewer checks the value against the filing)."""
        from filinglens.agent.tools import run_tool

        out: list[Case] = []
        combos = [(f, t) for f in self.filings for t in NUMERIC_TEMPLATES]
        self.rng.shuffle(combos)
        for (ticker, _cik, fy), (tool, key, label, answer_tpl) in combos:
            if len(out) >= n:
                break
            args: dict[str, Any] = {"ticker": ticker, "fiscal_years": [fy]}
            args |= {"concepts": [key]} if tool == "get_financials" else {"ratio": key}
            result, is_error = run_tool(self._ctx(), tool, args)
            if is_error:
                continue
            if tool == "get_financials":
                values = result.get("values") or []
                if not values:
                    continue
                value = float(values[0]["value"])
                v0 = values[0]
                source = f"{v0['xbrl_tag']} in {v0['accession_no']} (filed {v0['filed_date']})"
            else:
                raw = result["values_by_year"].get(str(fy))
                if raw is None:
                    continue
                value = float(raw)
                source = "; ".join(
                    f"{i['concept']}={i['value']} ({i['accession_no']})"
                    for i in result["inputs_used"]
                    if i["fiscal_year"] == fy
                )
            company = self.names.get(ticker, ticker)
            out.append(
                Case(
                    id=_id("numeric", ticker, fy, key),
                    type="numeric",
                    ticker=ticker,
                    fiscal_year=fy,
                    question=f"What was {company}'s {label} in fiscal {fy}?",
                    gold_answer=answer_tpl.format(
                        company=company, fy=fy, value=_fmt_value(tool, key, value)
                    ),
                    gold_value=value,
                    expected_tools=[tool],
                    source="draft:numeric",
                    notes=f"VERIFY against the 10-K: {source}",
                )
            )
        return out

    def yoy(self, n: int) -> list[Case]:
        """Risk-factor headings added between consecutive 10-Ks."""
        from filinglens.agent.tools import run_tool

        out: list[Case] = []
        tickers = sorted({t for t, _, _ in self.filings})
        self.rng.shuffle(tickers)
        for ticker in tickers:
            if len(out) >= n:
                break
            ctx = self._ctx()
            years = self.sections.available_years(ctx.cik(ticker), "1A")
            for fy in reversed(years[1:]):
                result, is_error = run_tool(
                    ctx, "compare_sections", {"ticker": ticker, "item": "1A", "year_b": fy}
                )
                added = [] if is_error else result["added_risk_headings"][:5]
                if not added:
                    continue
                acc = result["filing_b"]["accession_no"]
                spans = []
                for h in added:
                    src = ctx.registry.get(h["ref"])
                    if src and src.char_start is not None and src.char_end is not None:
                        spans.append(
                            GoldSpan(
                                accession_no=acc,
                                item="1A",
                                char_start=src.char_start,
                                char_end=src.char_end,
                            )
                        )
                headings = "; ".join(h["text"] for h in added)
                out.append(
                    Case(
                        id=_id("yoy", ticker, fy),
                        type="yoy_change",
                        ticker=ticker,
                        fiscal_year=fy,
                        question=f"What new risk factors did {ticker} add in its fiscal {fy} 10-K "
                        f"compared with fiscal {result['filing_a']['fiscal_year']}?",
                        gold_answer=f"New risk factor headings include: {headings}.",
                        gold_spans=spans,
                        expected_tools=["compare_sections"],
                        source="draft:yoy",
                        notes=f"{result['counts']['headings_added']} headings added in total; "
                        "check they are genuinely new, not reworded.",
                    )
                )
                break
        return out

    def multi_hop(self, n: int) -> list[Case]:
        """Operating-margin change plus management's stated drivers."""
        from filinglens.agent.tools import run_tool
        from filinglens.retrieval.filters import QueryFilters

        out: list[Case] = []
        candidates = [
            f for f in self.filings if any(g[0] == f[0] and g[2] == f[2] - 1 for g in self.filings)
        ]
        self.rng.shuffle(candidates)
        for ticker, cik, fy in candidates:
            if len(out) >= n:
                break
            result, is_error = run_tool(
                self._ctx(),
                "compute_ratio",
                {"ticker": ticker, "ratio": "operating_margin", "fiscal_years": [fy - 1, fy]},
            )
            vals = {} if is_error else result["values_by_year"]
            if None in (vals.get(str(fy - 1)), vals.get(str(fy))):
                continue
            q = f"{ticker} operating income and margin drivers fiscal {fy}"
            chunks = self.retriever.retrieve(
                q, filters=QueryFilters(ciks=(cik,), fiscal_years=(fy,), items=("7",))
            ).chunks[:4]
            if not chunks:
                continue
            passages = "\n\n".join(f"<passage>\n{c.text}\n</passage>" for c in chunks)
            drivers = self._parse(
                load_prompt("draft_drivers").text,
                f"{ticker} FY{fy} operating margin\n\n{passages}",
                Drivers,
            )
            a, b = float(vals[str(fy - 1)]), float(vals[str(fy)])
            out.append(
                Case(
                    id=_id("multi_hop", ticker, fy),
                    type="multi_hop",
                    ticker=ticker,
                    fiscal_year=fy,
                    question=f"How did {ticker}'s operating margin change from fiscal {fy - 1} to "
                    f"fiscal {fy}, and what does management say drove the change?",
                    gold_answer=f"Operating margin went from {a * 100:.1f}% to {b * 100:.1f}%. "
                    f"{drivers.drivers if drivers else ''}",
                    gold_value=b,
                    gold_chunk_ids=[c.chunk_id for c in chunks],
                    gold_spans=[
                        GoldSpan(
                            accession_no=c.accession_no,
                            item=c.item,
                            char_start=c.char_start or 0,
                            char_end=c.char_end or 0,
                        )
                        for c in chunks
                    ],
                    expected_tools=["compute_ratio", "search_filings"],
                    source="draft:multi_hop",
                    notes="Check the drivers against Item 7; drop chunks that don't support them.",
                )
            )
        return out

    def unanswerable(self, n: int) -> list[Case]:
        """Questions the corpus can't answer."""
        out: list[Case] = []
        latest = max((y for _, _, y in self.filings), default=2024)
        tickers = sorted({t for t, _, _ in self.filings})
        for i in range(n):
            ticker = tickers[i % len(tickers)] if tickers else "AAPL"
            tpl = UNANSWERABLE_TEMPLATES[i % len(UNANSWERABLE_TEMPLATES)]
            q = tpl.format(company=ticker, fy=latest, future=latest + 6)
            out.append(
                Case(
                    id=_id("unanswerable", q),
                    type="unanswerable",
                    ticker=None,
                    question=q,
                    gold_answer="The ingested filings don't contain this information, so the "
                    "system should say it can't answer.",
                    source="draft:unanswerable",
                )
            )
        return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=80, help="Total candidates to draft.")
    parser.add_argument("--dataset", default="golden_v1")
    parser.add_argument("--model", default="claude-opus-5-5", help="Drafting model.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--types", default=",".join(TARGET_MIX))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import anthropic

    from filinglens.config import get_settings
    from filinglens.db.connection import connect

    settings = get_settings()
    conn = connect(settings.database_url)
    drafter = Drafter(conn, anthropic.Anthropic(), args.model, settings, args.seed)
    wanted = args.types.split(",")
    cases: list[Case] = []
    for kind in wanted:
        k = max(1, round(args.n * TARGET_MIX[kind]))
        fn = {
            "factual": drafter.factual,
            "numeric": drafter.numeric,
            "yoy_change": drafter.yoy,
            "multi_hop": drafter.multi_hop,
            "unanswerable": drafter.unanswerable,
        }[kind]
        got = fn(k)
        logger.info("%s: drafted %d/%d", kind, len(got), k)
        cases += got

    path = dataset_path(f"{args.dataset}.candidates")
    existing = {c.id: c for c in read_jsonl(path)} if path.exists() else {}
    for c in cases:
        existing.setdefault(c.id, c)
    write_jsonl(path, list(existing.values()))
    print(
        f"{len(cases)} drafted, {len(existing)} candidates in {path}. "
        f"Next: python -m evals.verify --reviewer <your name>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
