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
    chosen = sorted(_candidates(question, texts, filters), key=lambda c: (-c[0], c[1]))[:limit]
    return [(rank, sentence) for _, rank, sentence in sorted(chosen, key=lambda c: c[1])]


def _norm_sentence(sentence: str) -> str:
    return re.sub(r"\W+", " ", sentence.lower()).strip()


def pick_sentences(
    question: str,
    texts: list[str],
    years: list[int | None],
    filters: QueryFilters,
    limit: int = MAX_SENTENCES,
) -> list[tuple[int, str]]:
    """Like :func:`best_sentences`, for passages from several fiscal years.

    Filings repeat boilerplate year after year, so the same sentence often comes back
    from several 10-Ks. Each distinct sentence is kept once, from the most recent filing.
    Newer filings also get a small boost (at most half a matching word), so when no
    year is asked for, current disclosures win ties. Results are ordered newest first.
    """
    known = [y for y in years if y is not None]
    lo, hi = (min(known), max(known)) if known else (0, 0)
    best: dict[str, tuple[float, int, int, str]] = {}  # norm -> (score, year, idx, sentence)
    for score, idx, sentence in _candidates(question, texts, filters):
        year = years[idx] or lo
        boosted = score + (0.5 * (year - lo) / (hi - lo) if hi > lo else 0.0)
        key = _norm_sentence(sentence)
        seen = best.get(key)
        if seen is None or (year, boosted) > (seen[1], seen[0]):
            best[key] = (max(boosted, seen[0]) if seen else boosted, year, idx, sentence)
    chosen = sorted(best.values(), key=lambda c: (-c[0], c[2]))[:limit]
    return [(idx, sentence) for _, _, idx, sentence in sorted(chosen, key=lambda c: (-c[1], -c[0]))]


def _candidates(
    question: str, texts: list[str], filters: QueryFilters
) -> list[tuple[float, int, str]]:
    """``(score, text index, sentence)``: the best-scoring sentence of each text."""
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
    return candidates


def extractive_answer(question: str, chunks: list[RetrievedChunk], filters: QueryFilters) -> Answer:
    """Build an :class:`Answer` from the best matching sentences of the retrieved chunks."""
    terms = _terms(question, filters)
    picked = pick_sentences(
        question, [c.text for c in chunks], [c.fiscal_year for c in chunks], filters
    )
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
