"""``filinglens agent``: answer with tools (search, section diff, financials, ratios)."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import typer

from filinglens.agent.loop import Agent, AgentResult
from filinglens.config import Settings, get_settings
from filinglens.llm.calllog import PostgresCallLog, SpendCapExceededError


def build_agent(settings: Settings, conn: Any, client: Any) -> Agent:
    """Wire the agent's tools to Postgres, parsed sections and the embedding model."""
    from filinglens.agent.diff import SectionStore
    from filinglens.agent.financials import PostgresFactStore
    from filinglens.agent.sources import SourceRegistry
    from filinglens.agent.tools import AgentContext
    from filinglens.rag.cli import build_retriever, retrieval_config

    retriever = build_retriever(settings, conn, retrieval_config(settings))
    sections = SectionStore(settings.processed_dir)
    facts = PostgresFactStore(conn)
    concept_map = str(settings.universe_file.parent / "concept_map.yaml")

    def context(registry: SourceRegistry) -> AgentContext:
        return AgentContext(
            retriever=retriever,
            sections=sections,
            facts=facts,
            directory=retriever.directory,
            registry=registry,
            embedder=retriever.embedder,
            concept_map_path=concept_map,
        )

    return Agent(
        client,
        settings.llm_model,
        PostgresCallLog(conn),
        context,
        effort=settings.llm_effort,
        max_steps=settings.agent_max_steps,
        max_tokens=settings.llm_max_tokens,
        daily_cap_usd=settings.llm_daily_spend_cap_usd,
        trace_log=settings.log_dir / "agent_traces.jsonl",
    )


def agent(
    question: str = typer.Argument(..., help="Question about the ingested filings."),
    as_json: bool = typer.Option(False, "--json", help="Print answer and trace as JSON."),
    trace: bool = typer.Option(False, "--trace", help="Show each step (tools, args, latency)."),
) -> None:
    """Answer a question with tools: filing search, YoY section diff, XBRL financials, ratios."""
    import anthropic

    from filinglens.rag.cli import _connect

    settings = get_settings()
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    conn = _connect(settings)
    with conn:
        runner = build_agent(settings, conn, client)
        try:
            result = runner.run(question)
        except SpendCapExceededError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=3) from exc
        except anthropic.AuthenticationError as exc:
            typer.secho(
                "Anthropic API authentication failed: set ANTHROPIC_API_KEY in .env.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2) from exc
        except anthropic.APIConnectionError as exc:
            typer.secho(f"Could not reach the Anthropic API: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
    _print(result, as_json=as_json, trace=trace)


def _print(result: AgentResult, *, as_json: bool, trace: bool) -> None:
    if as_json:
        payload = result.answer.model_dump()
        payload |= {
            "trace_id": result.trace_id,
            "tools_called": result.tools_called,
            "unsupported_numbers": result.unsupported_numbers,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "steps": [asdict(s) for s in result.steps],
        }
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    if trace:
        for s in result.steps:
            detail = (
                f"{s.input_tokens} in / {s.output_tokens} out, stop={s.stop_reason}"
                if s.kind == "llm"
                else json.dumps(s.args)
            )
            err = "  ERROR" if s.is_error else ""
            typer.secho(
                f"  [{s.index}] {s.kind:4} {s.name} ({s.latency_ms} ms) {detail}{err}",
                fg=typer.colors.BRIGHT_BLACK,
            )
        typer.echo()
    a = result.answer
    typer.echo(a.answer)
    if a.citations:
        typer.echo()
    for c in a.citations:
        mark = "" if c.verified else "  [UNVERIFIED quote]"
        where = (
            f"{c.tool_name}"
            if c.source == "tool"
            else f"{c.ticker} {c.form} FY{c.fiscal_year} Item {c.item}"
        )
        typer.echo(f"[{c.n}] {where}{mark}")
        typer.echo(f'    "{c.quote}"')
        if c.filing_url:
            typer.secho(f"    {c.filing_url}", fg=typer.colors.BRIGHT_BLACK)
    warn = (
        f" UNSUPPORTED NUMBERS: {', '.join(result.unsupported_numbers)}"
        if result.unsupported_numbers
        else ""
    )
    validity = "n/a" if a.citation_validity is None else f"{a.citation_validity:.0%}"
    typer.secho(
        f"\ntools={','.join(result.tools_called) or 'none'} confidence={a.confidence} "
        f"citations_valid={validity} | {result.latency_ms / 1000:.1f}s ${result.cost_usd:.4f}"
        f"{warn}",
        fg=typer.colors.BRIGHT_BLACK,
    )
