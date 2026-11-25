"""Structure-aware chunking of sections into retrieval units with exact character offsets."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from filinglens.parsing.items import item_label
from filinglens.parsing.sections import Section, Span
from filinglens.parsing.text import count_tokens

Strategy = Literal["structure", "fixed"]

DEFAULT_CHUNK_ITEMS: dict[str, tuple[str, ...]] = {
    "10-K": ("1", "1A", "1C", "3", "7", "7A"),
    "10-Q": ("I-2", "I-3", "II-1", "II-1A"),
}


@dataclass(frozen=True)
class ChunkConfig:
    """Chunking parameters. Evals (specs/05) choose the final values.

    Defaults leave headroom under bge-small's 512-token limit for the context header and
    for the gap between our approximate token count and the model's tokenizer.
    """

    max_tokens: int = 448
    overlap_tokens: int = 48
    min_tokens: int = 64
    strategy: Strategy = "structure"

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")
        if not 0 <= self.min_tokens <= self.max_tokens:
            raise ValueError("min_tokens must be in [0, max_tokens]")


@dataclass(frozen=True)
class FilingInfo:
    """Filing-level metadata copied onto every chunk."""

    accession_no: str
    cik: str
    ticker: str
    form: str
    fiscal_year: int | None
    fiscal_period: str | None


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit. ``text == section.text[char_start:char_end]``."""

    chunk_id: str
    accession_no: str
    cik: str
    ticker: str
    form: str
    fiscal_year: int | None
    fiscal_period: str | None
    item: str
    heading: str | None
    text: str
    token_count: int
    char_start: int
    char_end: int
    context_header: str

    @property
    def embedding_text(self) -> str:
        """Text to embed: context header + chunk body."""
        return f"{self.context_header}\n{self.text}"


_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[\"'(\[A-Z0-9•])")
_WORD = re.compile(r"\S+")


def _split_long(
    text: str, start: int, max_tokens: int, counter: Callable[[str], int]
) -> list[tuple[int, int]]:
    """Split one oversize block into sentence (then word-window) pieces."""
    pieces: list[tuple[int, int]] = []
    cursor = 0
    bounds = [m.start() for m in _SENTENCE_END.finditer(text)] + [len(text)]
    for b in bounds:
        sentence = text[cursor:b]
        if sentence.strip():
            if counter(sentence) <= max_tokens:
                pieces.append((start + cursor, start + b))
            else:
                words = [(m.start(), m.end()) for m in _WORD.finditer(sentence)]
                step = max(1, max_tokens // 2)
                for i in range(0, len(words), step):
                    window = words[i : i + step]
                    pieces.append((start + cursor + window[0][0], start + cursor + window[-1][1]))
        cursor = b
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    return pieces


Piece = tuple[int, int, bool]  # (start, end, is_heading)


def _pieces(
    section: Section, spans: list[Span], cfg: ChunkConfig, counter: Callable[[str], int]
) -> list[Piece]:
    """Break spans into packable pieces: whole blocks, table rows, or sentences."""
    out: list[Piece] = []
    for span in spans:
        body = section.text[span.start : span.end]
        if counter(body) <= cfg.max_tokens:
            out.append((span.start, span.end, span.kind == "heading"))
        elif span.kind == "table":
            pos = span.start
            for line in body.split("\n"):
                out.append((pos, pos + len(line), False))
                pos += len(line) + 1
        else:
            out.extend(
                (a, b, False) for a, b in _split_long(body, span.start, cfg.max_tokens, counter)
            )
    return out


def _segments(
    section: Section, cfg: ChunkConfig, counter: Callable[[str], int]
) -> list[list[Span]]:
    """Group spans into heading-led segments (structure) or a single segment (fixed).

    A heading starts a segment; consecutive headings (category + risk heading) stay
    together. A segment under ``min_tokens`` is appended to the previous one, so every
    risk factor keeps its own heading at the start of its chunk.
    """
    spans = list(section.spans)
    if cfg.strategy == "fixed":
        return [spans]
    segments: list[list[Span]] = []
    for span in spans:
        starts_new = span.kind == "heading" and segments and segments[-1][-1].kind != "heading"
        if not segments or starts_new:
            segments.append([span])
        else:
            segments[-1].append(span)

    def tokens(seg: list[Span]) -> int:
        return counter(section.text[seg[0].start : seg[-1].end])

    merged: list[list[Span]] = []
    for seg in segments:
        if merged and tokens(seg) < cfg.min_tokens and tokens(merged[-1] + seg) <= cfg.max_tokens:
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)
    return merged


def _pack(
    pieces: list[Piece], section: Section, cfg: ChunkConfig, counter: Callable[[str], int]
) -> list[tuple[int, int]]:
    """Greedily pack contiguous pieces into windows of <= max_tokens with piece-level overlap.

    A window never ends on a heading (the heading moves to the next window), and a window
    that would be a heading alone takes the following piece too.
    """

    def fits(a: int, b: int, limit: int) -> bool:
        return counter(section.text[pieces[a][0] : pieces[b][1]]) <= limit

    windows: list[tuple[int, int]] = []
    i = 0
    while i < len(pieces):
        j = i
        while j + 1 < len(pieces) and fits(i, j + 1, cfg.max_tokens):
            j += 1
        while j > i and pieces[j][2] and j + 1 < len(pieces):
            j -= 1
        if j == i and pieces[i][2] and i + 1 < len(pieces):
            j = i + 1
        windows.append((pieces[i][0], pieces[j][1]))
        if j + 1 >= len(pieces):
            break
        k = j + 1
        while k - 1 > i and not pieces[k - 1][2] and fits(k - 1, j, cfg.overlap_tokens):
            k -= 1
        i = k
    return windows


def _heading_at(section: Section, start: int) -> str | None:
    """Nearest sub-heading for a chunk starting at ``start`` (skips the Item heading itself)."""
    spans = section.spans
    idx = max((n for n, s in enumerate(spans) if s.start <= start), default=0)
    if spans[idx].kind == "heading":
        while idx + 1 < len(spans) and spans[idx + 1].kind == "heading":
            idx += 1
    else:
        while idx >= 0 and spans[idx].kind != "heading":
            idx -= 1
    if idx <= 0:
        return None
    return section.text[spans[idx].start : spans[idx].end]


def context_header(info: FilingInfo, item: str, heading: str | None) -> str:
    """``[AAPL | 10-K | FY2023 | Item 1A Risk Factors | <heading>]``."""
    period = f"FY{info.fiscal_year}" if info.fiscal_year else "FY?"
    if info.fiscal_period and info.fiscal_period != "FY":
        period += f" {info.fiscal_period}"
    parts = [info.ticker, info.form, period, item_label(info.form, item)]
    if heading:
        parts.append(heading if len(heading) <= 160 else heading[:157].rstrip() + "...")
    return "[" + " | ".join(parts) + "]"


def chunk_section(
    section: Section,
    info: FilingInfo,
    cfg: ChunkConfig = ChunkConfig(),
    counter: Callable[[str], int] = count_tokens,
) -> list[Chunk]:
    """Chunk one section. Never crosses the section boundary; deterministic for given input."""
    if not section.spans:
        return []
    windows: list[tuple[int, int]] = []
    for seg in _segments(section, cfg, counter):
        whole = (seg[0].start, seg[-1].end)
        if counter(section.text[whole[0] : whole[1]]) <= cfg.max_tokens:
            windows.append(whole)
        else:
            windows.extend(_pack(_pieces(section, seg, cfg, counter), section, cfg, counter))

    chunks = []
    for n, (start, end) in enumerate(windows):
        text = section.text[start:end]
        heading = _heading_at(section, start)
        chunks.append(
            Chunk(
                chunk_id=f"{info.accession_no}_{section.item}_{n:04d}",
                accession_no=info.accession_no,
                cik=info.cik,
                ticker=info.ticker,
                form=info.form,
                fiscal_year=info.fiscal_year,
                fiscal_period=info.fiscal_period,
                item=section.item,
                heading=heading,
                text=text,
                token_count=counter(text),
                char_start=start,
                char_end=end,
                context_header=context_header(info, section.item, heading),
            )
        )
    return chunks
