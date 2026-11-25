"""Split a filing into Items: native heading splitter, with edgartools filling gaps."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from filinglens.parsing.blocks import Block, BlockKind, decode_html, html_to_blocks
from filinglens.parsing.clean import remove_page_furniture, running_furniture
from filinglens.parsing.items import (
    ITEM_HEADING,
    PART_HEADING,
    REQUIRED_ITEMS,
    base_form,
    canonical_items,
    item_key,
)
from filinglens.parsing.text import normalize_text

logger = logging.getLogger(__name__)

CANDIDATE_MAX_CHARS = 250
BLOCK_SEP = "\n\n"


@dataclass(frozen=True)
class Span:
    """Location of one block inside a section's text."""

    kind: BlockKind
    start: int
    end: int


@dataclass(frozen=True)
class Section:
    """One Item of a filing, as clean text with block offsets (for chunking and citations)."""

    item: str
    title: str
    text: str
    spans: tuple[Span, ...]
    source: str

    @classmethod
    def from_blocks(cls, item: str, title: str, blocks: list[Block], source: str) -> Section:
        """Join blocks with blank lines, recording each block's character span."""
        parts: list[str] = []
        spans: list[Span] = []
        pos = 0
        for block in blocks:
            if parts:
                pos += len(BLOCK_SEP)
            spans.append(Span(block.kind, pos, pos + len(block.text)))
            parts.append(block.text)
            pos += len(block.text)
        return cls(item, title, BLOCK_SEP.join(parts), tuple(spans), source)


@dataclass
class ExtractionResult:
    """Sections for one filing plus diagnostics for the quality report."""

    sections: dict[str, Section]
    edgartools_chars: dict[str, int] = field(default_factory=dict)
    edgartools_error: str | None = None


def _is_candidate(block: Block) -> bool:
    """Item headings are short headings, or short plain paragraphs in unstyled filings."""
    limit = CANDIDATE_MAX_CHARS if block.kind == "heading" else CANDIDATE_MAX_CHARS // 2
    return block.kind != "table" and len(block.text) <= limit


def _longest_ordered(cands: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Longest subsequence (by block position) whose canonical item ranks strictly increase."""
    if not cands:
        return []
    best = [1] * len(cands)
    prev = [-1] * len(cands)
    for i in range(len(cands)):
        for j in range(i):
            if cands[j][1] < cands[i][1] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    i = max(range(len(cands)), key=lambda k: (best[k], k))
    out = []
    while i != -1:
        out.append(cands[i])
        i = prev[i]
    return out[::-1]


def split_items(blocks: list[Block], form: str) -> dict[str, list[Block]]:
    """Find Item headings in ``blocks`` and return each Item's blocks (heading included).

    Tables of contents are already dropped when they are tables; plain-text ToCs are handled
    by taking the *last* heading-like occurrence of each item, then keeping the longest run
    of items in canonical filing order.
    """
    order = {key: rank for rank, key in enumerate(canonical_items(form))}
    is_10q = base_form(form) == "10-Q"
    part: str | None = None
    last_seen: dict[str, int] = {}

    for idx, block in enumerate(blocks):
        if not _is_candidate(block):
            continue
        match = ITEM_HEADING.match(block.text)
        if match is None:
            part_match = PART_HEADING.match(block.text)
            if part_match and is_10q:
                part = part_match.group("part").upper()
            continue
        if "continued" in match.group("title").lower():
            continue
        if match.group("part"):
            part = match.group("part").upper()
        key = item_key(form, match.group("num") + (match.group("letter") or ""), part)
        if key in order:
            last_seen[key] = idx

    cands = sorted((idx, order[key], key) for key, idx in last_seen.items())
    chosen = _longest_ordered(cands)
    out: dict[str, list[Block]] = {}
    for n, (start, _rank, key) in enumerate(chosen):
        end = chosen[n + 1][0] if n + 1 < len(chosen) else len(blocks)
        # A "PART II" heading just before the next item belongs to the next part.
        while end - 1 > start and PART_HEADING.match(blocks[end - 1].text):
            end -= 1
        out[key] = blocks[start:end]
    return out


_MD_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")
_MD_EMPHASIS = re.compile(r"(?<!\\)\*{1,3}")
_MD_HEADING = re.compile(r"^#{1,6}\s+")


def markdown_to_blocks(markdown: str) -> list[Block]:
    """Convert edgartools section markdown into blocks, undoing its markdown escaping."""
    blocks: list[Block] = []
    for para in re.split(r"\n\s*\n", markdown):
        para = para.strip()
        if not para:
            continue
        if para.startswith("|"):
            blocks.append(Block("table", _MD_ESCAPE.sub(r"\1", para)))
            continue
        heading = bool(_MD_HEADING.match(para)) or (
            para.startswith("**") and para.endswith("**") and "\n" not in para
        )
        text = normalize_text(
            _MD_ESCAPE.sub(r"\1", _MD_EMPHASIS.sub("", _MD_HEADING.sub("", para)))
        )
        if text:
            blocks.append(Block("heading" if heading else "para", text))
    return blocks


def edgartools_items(html: str, form: str) -> dict[str, list[Block]]:
    """Sections detected by edgartools' offline HTML parser, keyed like :func:`split_items`."""
    from edgar.documents import HTMLParser, ParserConfig

    document = HTMLParser(ParserConfig(form=base_form(form))).parse(html)
    out: dict[str, list[Block]] = {}
    for section in document.sections.values():
        code = getattr(section, "item", None)
        if not code:
            continue
        key = item_key(form, str(code), getattr(section, "part", None))
        if key in canonical_items(form) and key not in out:
            out[key] = markdown_to_blocks(section.markdown())
    return out


def extract_sections(raw: bytes, form: str, *, use_edgartools: bool = True) -> ExtractionResult:
    """Extract Items from one filing's primary document.

    The native splitter is primary because it keeps heading structure (needed for
    one-chunk-per-risk-factor) and verbatim text. edgartools runs as a cross-check and
    supplies any required item the native splitter missed.
    """
    titles = canonical_items(form)
    raw_blocks = html_to_blocks(raw)
    running = running_furniture(raw_blocks)
    blocks = remove_page_furniture(raw_blocks, running)
    native = split_items(blocks, form)
    sections = {
        key: Section.from_blocks(key, titles[key], item_blocks, "native")
        for key, item_blocks in native.items()
    }
    result = ExtractionResult(sections=sections)
    if not use_edgartools:
        return result

    try:
        et = edgartools_items(decode_html(raw), form)
    except Exception as exc:  # third-party parser: never let it fail the filing
        result.edgartools_error = f"{type(exc).__name__}: {exc}"
        logger.debug("edgartools failed: %s", exc)
        return result

    for key, et_blocks in et.items():
        cleaned = remove_page_furniture(et_blocks, running)
        result.edgartools_chars[key] = sum(len(b.text) for b in cleaned)
        if key not in sections and key in REQUIRED_ITEMS[base_form(form)] and cleaned:
            sections[key] = Section.from_blocks(key, titles[key], cleaned, "edgartools")

    ordered = {k: sections[k] for k in titles if k in sections}
    result.sections = ordered
    return result
