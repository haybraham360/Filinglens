"""Question -> retrieval -> Claude -> verified, cited answer."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filinglens.ingestion.universe import archive_cik
from filinglens.llm.prompts import Prompt, load_prompt
from filinglens.llm.provider import StructuredLLM
from filinglens.rag.citations import QUOTE_THRESHOLD, quote_match_score
from filinglens.rag.schemas import Answer, AnswerDraft, Citation
from filinglens.retrieval.retriever import RetrievalResult, Retriever
from filinglens.retrieval.search import RetrievedChunk

_MARKER = re.compile(r"\[(C\d+)\]")


@dataclass
class AnswerResult:
    """The answer plus everything needed to debug or evaluate it."""

    answer: Answer
    retrieval: RetrievalResult
    trace_id: str
    llm_model: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    cost_usd: float | None = None


def filing_index_url(cik: str, accession_no: str) -> str:
    """EDGAR filing index page for an accession."""
    return (
        f"https://www.sec.gov/Archives/edgar/data/{archive_cik(cik)}/"
        f"{accession_no.replace('-', '')}/{accession_no}-index.htm"
    )


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, dict[str, RetrievedChunk]]:
    """Render chunks as ``<chunk ref="C1">`` blocks; return the text and the handle map."""
    handles: dict[str, RetrievedChunk] = {}
    parts = []
    for i, chunk in enumerate(chunks, 1):
        ref = f"C{i}"
        handles[ref] = chunk
        # Text stays verbatim (quotes are verified against it); only a stray closing tag
        # inside filing text is neutralized so it can't end the chunk early.
        body = chunk.passage.replace("</chunk>", "</ chunk>")
        parts.append(f'<chunk ref="{ref}">\n{body}\n</chunk>')
    return "\n\n".join(parts), handles


def _user_message(question: str, context: str) -> str:
    return f"<question>\n{question}\n</question>\n\n<context>\n{context}\n</context>"


def resolve_citations(
    draft: AnswerDraft, handles: dict[str, RetrievedChunk], threshold: float = QUOTE_THRESHOLD
) -> Answer:
    """Map handles to real chunks, verify quotes, and renumber markers as [1], [2], ...

    Citations to unknown handles are dropped; citations whose quote isn't found in the chunk
    are kept but flagged ``verified=False`` and counted against ``citation_validity``.
    """
    order: list[str] = []
    for ref in _MARKER.findall(draft.answer) + [c.ref.strip("[] ") for c in draft.citations]:
        if ref in handles and ref not in order:
            order.append(ref)
    number = {ref: i + 1 for i, ref in enumerate(order)}

    citations: list[Citation] = []
    for c in draft.citations:
        ref = c.ref.strip("[] ")
        chunk = handles.get(ref)
        if chunk is None:
            continue
        score = quote_match_score(c.quote, chunk.text)
        citations.append(
            Citation(
                n=number[ref],
                chunk_id=chunk.chunk_id,
                accession_no=chunk.accession_no,
                item=chunk.item,
                ticker=chunk.ticker,
                form=chunk.form,
                fiscal_year=chunk.fiscal_year,
                quote=c.quote,
                verified=score >= threshold,
                match_score=round(score, 3),
                filing_url=filing_index_url(chunk.cik, chunk.accession_no),
            )
        )
    citations.sort(key=lambda c: (c.n, c.chunk_id))

    def renumber(m: re.Match[str]) -> str:
        ref = m.group(1)
        return f"[{number[ref]}]" if ref in number else ""

    text = re.sub(r"\s+([.,;:])", r"\1", _MARKER.sub(renumber, draft.answer)).strip()
    validity = (sum(c.verified for c in citations) / len(citations)) if citations else None
    return Answer(
        answer=text,
        citations=citations,
        insufficient_context=draft.insufficient_context,
        confidence=draft.confidence,
        citation_validity=validity,
    )


def _insufficient(message: str) -> Answer:
    return Answer(
        answer=message,
        citations=[],
        insufficient_context=True,
        confidence="low",
        citation_validity=None,
    )


def answer_question(
    question: str,
    retriever: Retriever,
    llm: StructuredLLM,
    *,
    prompt: Prompt | None = None,
    query_log: Path | None = None,
) -> AnswerResult:
    """Answer ``question`` from retrieved filing chunks, with verified citations."""
    prompt = prompt or load_prompt("answer")
    trace_id = uuid.uuid4().hex
    start = time.perf_counter()
    retrieval = retriever.retrieve(question)
    timings = {f"retrieval_{k}": v for k, v in retrieval.timings_ms.items()}

    if not retrieval.chunks:
        result = AnswerResult(
            answer=_insufficient(
                "No matching filing text was found for this question. The company or period "
                "may not be in the ingested universe."
            ),
            retrieval=retrieval,
            trace_id=trace_id,
        )
    else:
        context, handles = build_context(retrieval.chunks)
        llm_result = llm.generate(
            system=prompt.text,
            user=_user_message(question, context),
            schema=AnswerDraft,
            purpose="answer",
            prompt_version=prompt.id,
            trace_id=trace_id,
        )
        timings["llm"] = float(llm_result.latency_ms)
        if llm_result.parsed is None:
            reason = (
                "the model declined to answer"
                if llm_result.stop_reason == "refusal"
                else f"the response was cut off ({llm_result.stop_reason})"
            )
            answer = _insufficient(f"No answer was produced: {reason}.")
        else:
            answer = resolve_citations(llm_result.parsed, handles)
        result = AnswerResult(
            answer=answer,
            retrieval=retrieval,
            trace_id=trace_id,
            llm_model=llm_result.model,
            cost_usd=llm_result.cost_usd,
        )
    timings["total"] = (time.perf_counter() - start) * 1000
    result.timings_ms = timings
    if query_log is not None:
        log_query(query_log, question, result)
    return result


def log_query(path: Path, question: str, result: AnswerResult) -> None:
    """Append one JSON line per question (latency, filters, chunks, citation validity)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "trace_id": result.trace_id,
        "question": question,
        "filters": {
            "tickers": result.retrieval.filters.tickers,
            "fiscal_years": result.retrieval.filters.fiscal_years,
            "forms": result.retrieval.filters.forms,
            "items": result.retrieval.filters.items,
        },
        "relaxed": result.retrieval.relaxed,
        "chunk_ids": [c.chunk_id for c in result.retrieval.chunks],
        "llm_model": result.llm_model,
        "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
        "cost_usd": result.cost_usd,
        "insufficient_context": result.answer.insufficient_context,
        "citation_validity": result.answer.citation_validity,
        "n_citations": len(result.answer.citations),
    }
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
