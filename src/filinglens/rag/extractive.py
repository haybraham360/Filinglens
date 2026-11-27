"""LLM-free answers: the most relevant filing sentences, quoted verbatim with citations.

Used when no LLM is configured (``LLM_PROVIDER=none``, or ``auto`` without Claude or
Ollama). Deterministic and free. Sentences are scored by how many of the question's
content words they contain, with a small bonus for higher-ranked chunks; at most one
sentence per chunk is used so several passages are represented.
"""

from __future__ import annotations

import re

from filinglens.rag.answer import filing_index_url
from filinglens.rag.schemas import Answer, Citation
from filinglens.retrieval.filters import QueryFilters, lexical_query
from filinglens.retrieval.search import RetrievedChunk

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z0-9]+")
MAX_SENTENCES = 4
MIN_CHARS = 40
MAX_CHARS = 600
EXTRACTIVE_NOTE = "Extractive answer (no LLM): the most relevant passages from the filings."


def _terms(question: str, filters: QueryFilters) -> set[str]:
    query = lexical_query(question, filters).replace('"', " ")
    words = {w for part in query.split(" OR ") for w in _WORD.findall(part.lower())}
    stems = {w[:6] for w in words if len(w) >= 4}
    return words | stems


def _score(sentence: str, terms: set[str]) -> int:
    words = set(_WORD.findall(sentence.lower()))
    return len(words & terms) + len({w[:6] for w in words if len(w) >= 4} & terms)


def _is_heading(paragraph: str) -> bool:
    """Short, single-line, not ending like a sentence: a block heading."""
    return (
        0 < len(paragraph) <= 160
        and "\n" not in paragraph
        and not paragraph.rstrip().endswith((".", ":", ";"))
    )


def best_sentences(
    question: str, texts: list[str], filters: QueryFilters, limit: int = MAX_SENTENCES
) -> list[tuple[int, str]]:
    """``(text index, sentence)`` pairs: the best sentence of each text, top ``limit``,
    returned in text order. Texts with no matching sentence are skipped."""
    terms = _terms(question, filters)
    candidates: list[tuple[float, int, str]] = []
    for rank, text in enumerate(texts):
        best: tuple[float, str] | None = None
        heading_score = 0.0  # words of the heading above count, at half weight
        for para in text.split("\n\n"):
            para = para.strip()
            if _is_heading(para):
                heading_score = 0.5 * _score(para, terms)
                continue
            for sentence in _SENTENCE.split(para):
                sentence = sentence.strip()
                if not (MIN_CHARS <= len(sentence) <= MAX_CHARS) or sentence.startswith("|"):
                    continue
                own = _score(sentence, terms)
                score = own + heading_score - 0.1 * rank
                if (own or heading_score) and (best is None or score > best[0]):
                    best = (score, sentence)
        if best is not None and best[0] > 0:
            candidates.append((best[0], rank, best[1]))
    chosen = sorted(candidates, key=lambda c: (-c[0], c[1]))[:limit]
    return [(rank, sentence) for _, rank, sentence in sorted(chosen, key=lambda c: c[1])]


def extractive_answer(question: str, chunks: list[RetrievedChunk], filters: QueryFilters) -> Answer:
    """Build an :class:`Answer` from the best matching sentences of the retrieved chunks."""
    terms = _terms(question, filters)
    picked = best_sentences(question, [c.text for c in chunks], filters)
    if not picked:
        return Answer(
            answer="No passage in the retrieved filings matches this question closely enough "
            "to answer it without an LLM.",
            citations=[],
            insufficient_context=True,
            confidence="low",
        )
    lines, citations = [EXTRACTIVE_NOTE], []
    for n, (rank, sentence) in enumerate(picked, 1):
        chunk = chunks[rank]
        where = f"{chunk.ticker} {chunk.form} FY{chunk.fiscal_year} Item {chunk.item}"
        lines.append(f"- {where}: {sentence} [{n}]")
        citations.append(
            Citation(
                n=n,
                chunk_id=chunk.chunk_id,
                accession_no=chunk.accession_no,
                item=chunk.item,
                ticker=chunk.ticker,
                form=chunk.form,
                fiscal_year=chunk.fiscal_year,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                quote=sentence,
                verified=True,  # copied verbatim from the chunk
                match_score=1.0,
                filing_url=filing_index_url(chunk.cik, chunk.accession_no),
            )
        )
    best_score = max(_score(sentence, terms) for _, sentence in picked)
    return Answer(
        answer="\n".join(lines),
        citations=citations,
        insufficient_context=False,
        confidence="medium" if best_score >= 3 else "low",
        citation_validity=1.0,
    )
