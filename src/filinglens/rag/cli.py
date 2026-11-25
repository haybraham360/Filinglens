"""``filinglens embed`` and ``filinglens ask`` (specs/03)."""

from __future__ import annotations

import json
from typing import Any

import psycopg
import typer

from filinglens.config import Settings, get_settings
from filinglens.llm.calllog import PostgresCallLog, SpendCapExceededError
from filinglens.rag.answer import AnswerResult, answer_question
from filinglens.retrieval.embedder import load_embedder
from filinglens.retrieval.filters import CompanyDirectory
from filinglens.retrieval.retriever import RetrievalConfig, Retriever


def _connect(settings: Settings) -> psycopg.Connection[Any]:
    from filinglens.db.connection import connect

    try:
        return connect(settings.database_url)
    except psycopg.OperationalError as exc:
        typer.secho(
            f"Cannot connect to Postgres ({exc}). Start it with `make db`.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from exc


def embed(
    batch_size: int = typer.Option(256, min=1, help="Chunks per embedding batch."),
    limit: int | None = typer.Option(None, min=1, help="Embed at most this many chunks."),
) -> None:
    """Embed chunks (context header + text) that lack an embedding from EMBEDDING_MODEL."""
    from filinglens.retrieval.indexer import embed_pending_chunks, pending_count

    settings = get_settings()
    conn = _connect(settings)
    with conn:
        todo = pending_count(conn, settings.embedding_model)
        if todo == 0:
            typer.echo(f"All chunks already embedded with {settings.embedding_model}.")
            return
        typer.echo(
            f"Embedding {todo if limit is None else min(todo, limit)} chunks "
            f"with {settings.embedding_model}..."
        )
        embedder = load_embedder(settings.embedding_model)
        done = embed_pending_chunks(
            conn,
            embedder,
            expected_dim=settings.embedding_dim,
            batch_size=batch_size,
            limit=limit,
            progress=lambda d, t: typer.echo(f"  {d}/{t}"),
        )
        typer.echo(f"Embedded {done} chunks; {pending_count(conn, embedder.model_name)} left.")


def retrieval_config(settings: Settings, **overrides: Any) -> RetrievalConfig:
    """Retrieval knobs from settings, with CLI/eval overrides."""
    base: dict[str, Any] = {
        "top_k": settings.retrieval_top_k,
        "candidate_k": settings.retrieval_candidate_k,
        "use_hybrid": settings.use_hybrid,
        "rrf_k": settings.rrf_k,
        "use_rerank": settings.use_rerank,
        "rerank_top_n": settings.rerank_top_n,
    }
    base.update({k: v for k, v in overrides.items() if v is not None})
    return RetrievalConfig(**base)


def build_retriever(
    settings: Settings, conn: psycopg.Connection[Any], cfg: RetrievalConfig
) -> Retriever:
    """Wire embedder, company directory and (optional) reranker for ``cfg``."""
    from filinglens.retrieval.rerank import CrossEncoderReranker

    universe = settings.universe_file.stem.removesuffix("_universe")
    directory = CompanyDirectory.load(
        settings.raw_dir / "universe" / f"{universe}_ciks.csv",
        settings.universe_file.parent / "company_aliases.yaml",
    )
    reranker = CrossEncoderReranker(settings.rerank_model) if cfg.use_rerank else None
    return Retriever(conn, load_embedder(settings.embedding_model), directory, cfg, reranker)


def ask(
    question: str = typer.Argument(..., help="Question about the ingested filings."),
    top_k: int | None = typer.Option(None, min=1, help="Chunks given to the LLM."),
    hybrid: bool | None = typer.Option(None, "--hybrid/--dense-only", help="Add full-text + RRF."),
    rerank: bool | None = typer.Option(None, "--rerank/--no-rerank", help="Cross-encoder rerank."),
    filters: bool = typer.Option(True, "--filters/--no-filters", help="Company/year filtering."),
    as_json: bool = typer.Option(False, "--json", help="Print the structured answer as JSON."),
    show_context: bool = typer.Option(False, help="Also print the retrieved chunks."),
) -> None:
    """Answer a question about the ingested filings, with verified citations."""
    from filinglens.llm.provider import AnthropicLLM

    settings = get_settings()
    if settings.anthropic_api_key is None:
        typer.secho(
            "Note: ANTHROPIC_API_KEY is not set; using the SDK's other credential "
            "sources (ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile).",
            fg=typer.colors.YELLOW,
            err=True,
        )
    cfg = retrieval_config(
        settings,
        top_k=top_k,
        rerank_top_n=top_k,
        use_hybrid=hybrid,
        use_rerank=rerank,
        use_filters=filters,
    )
    conn = _connect(settings)
    with conn:
        retriever = build_retriever(settings, conn, cfg)
        import anthropic

        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        llm = AnthropicLLM(
            settings.llm_model,
            PostgresCallLog(conn),
            effort=settings.llm_effort,
            max_tokens=settings.llm_max_tokens,
            daily_cap_usd=settings.llm_daily_spend_cap_usd,
            client=anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic(),
        )
        try:
            result = answer_question(
                question, retriever, llm, query_log=settings.log_dir / "queries.jsonl"
            )
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
    _print(result, as_json=as_json, show_context=show_context)


def _print(result: AnswerResult, *, as_json: bool, show_context: bool) -> None:
    if as_json:
        payload = result.answer.model_dump()
        payload["trace_id"] = result.trace_id
        payload["timings_ms"] = {k: round(v) for k, v in result.timings_ms.items()}
        typer.echo(json.dumps(payload, indent=2))
        return
    f = result.retrieval.filters
    scope = ", ".join(
        p
        for p in (
            "/".join(f.tickers),
            "/".join(f"FY{y}" for y in f.fiscal_years),
            "/".join(f.forms),
            "/".join(f"Item {i}" for i in f.items),
        )
        if p
    )
    if scope or result.retrieval.relaxed:
        relaxed = (
            f" (relaxed: {', '.join(result.retrieval.relaxed)})" if result.retrieval.relaxed else ""
        )
        typer.secho(f"Filters: {scope or 'none'}{relaxed}", fg=typer.colors.BRIGHT_BLACK)
    typer.echo()
    typer.echo(result.answer.answer)
    if result.answer.citations:
        typer.echo()
        for c in result.answer.citations:
            mark = "" if c.verified else "  [UNVERIFIED quote]"
            typer.echo(
                f"[{c.n}] {c.ticker} {c.form} FY{c.fiscal_year} Item {c.item} - {c.chunk_id}{mark}"
            )
            typer.echo(f'    "{c.quote}"')
            typer.secho(f"    {c.filing_url}", fg=typer.colors.BRIGHT_BLACK)
    a = result.answer
    validity = "n/a" if a.citation_validity is None else f"{a.citation_validity:.0%}"
    cost = "" if result.cost_usd is None else f", ${result.cost_usd:.4f}"
    typer.secho(
        f"\nconfidence={a.confidence} insufficient_context={a.insufficient_context} "
        f"citations_valid={validity} | {result.timings_ms.get('total', 0) / 1000:.1f}s{cost}",
        fg=typer.colors.BRIGHT_BLACK,
    )
    if show_context:
        for i, chunk in enumerate(result.retrieval.chunks, 1):
            typer.secho(f"\n--- C{i} {chunk.chunk_id} ranks={chunk.ranks}", fg=typer.colors.CYAN)
            typer.echo(chunk.passage[:1200])
