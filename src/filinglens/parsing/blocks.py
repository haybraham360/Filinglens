"""Convert filing HTML into an ordered list of text blocks (headings, paragraphs, tables)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from lxml import html as lxml_html
from lxml.etree import _Element

from filinglens.parsing.text import normalize_text

BlockKind = Literal["heading", "para", "table"]


@dataclass(frozen=True)
class Block:
    """One unit of filing text."""

    kind: BlockKind
    text: str


_BLOCK_TAGS = frozenset(
    [
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "center",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "ul",
        "html",
    ]
)
_SKIP_TAGS = frozenset({"script", "style", "head", "title", "noscript", "ix:header"})
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BOLD_TAGS = frozenset({"b", "strong"}) | _HEADING_TAGS
_ITALIC_TAGS = frozenset({"i", "em", "u"})

_FONT_WEIGHT = re.compile(r"font-weight\s*:\s*([a-z0-9]+)", re.I)
_ITALIC_STYLE = re.compile(r"font-style\s*:\s*(italic|oblique)|text-decoration[^;]*underline", re.I)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.I)
_ITEM_CELL = re.compile(r"^item\s*\d", re.I)
_CURRENCY = frozenset({"$", "\u20ac", "\u00a3", "\u00a5"})
_TRAILER = re.compile(r"^\)?\s*%?\)?$")

HEADING_MAX_CHARS = 400


def decode_html(raw: bytes) -> str:
    """Decode filing bytes: UTF-8 if valid, else Windows-1252 (common in pre-2010 filings)."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _tag(el: _Element) -> str:
    tag = el.tag
    return tag.lower() if isinstance(tag, str) else ""


def _emphasis(el: _Element, tag: str, bold: bool, italic: bool) -> tuple[bool, bool]:
    if tag in _BOLD_TAGS:
        bold = True
    if tag in _ITALIC_TAGS:
        italic = True
    style = el.get("style") or ""
    if style:
        weight = _FONT_WEIGHT.search(style)
        if weight:
            value = weight.group(1).lower()
            bold = value in {"bold", "bolder"} or (value.isdigit() and int(value) >= 600)
        if _ITALIC_STYLE.search(style):
            italic = True
    return bold, italic


def _is_hidden(el: _Element) -> bool:
    return bool(_HIDDEN_STYLE.search(el.get("style") or ""))


def _is_allcaps_heading(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(text) <= 150 and len(letters) >= 4 and not any(c.islower() for c in letters)


class _Builder:
    """Accumulates inline text runs and emits blocks at block boundaries."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self._runs: list[tuple[str, bool]] = []

    def add(self, text: str | None, emphasized: bool) -> None:
        if text:
            self._runs.append((text, emphasized))

    def flush(self) -> None:
        if not self._runs:
            return
        text = normalize_text("".join(t for t, _ in self._runs))
        if text:
            total = sum(len("".join(t.split())) for t, _ in self._runs)
            emph = sum(len("".join(t.split())) for t, e in self._runs if e)
            heading = len(text) <= HEADING_MAX_CHARS and (
                (total > 0 and emph / total >= 0.9) or _is_allcaps_heading(text)
            )
            self.blocks.append(Block("heading" if heading else "para", text))
        self._runs = []

    def walk(self, el: _Element, bold: bool = False, italic: bool = False) -> None:
        tag = _tag(el)
        if not tag or tag in _SKIP_TAGS or _is_hidden(el):
            return
        if tag == "table":
            self.flush()
            self.blocks.extend(_table_blocks(el))
            return
        bold, italic = _emphasis(el, tag, bold, italic)
        emphasized = bold or italic
        is_block = tag in _BLOCK_TAGS
        if is_block:
            self.flush()
        if tag == "br":
            self.add(" ", emphasized)
        self.add(el.text, emphasized)
        for child in el:
            self.walk(child, bold, italic)
            self.add(child.tail, emphasized)
        if is_block:
            self.flush()


def _cell_text(cell: _Element) -> tuple[str, bool]:
    builder = _Builder()
    builder.walk(cell)
    builder.flush()
    text = " ".join(b.text for b in builder.blocks)
    heading = bool(builder.blocks) and all(b.kind == "heading" for b in builder.blocks)
    return text, heading


def _merge_cells(cells: list[str]) -> list[str]:
    """Join split financial cells: ``$ | 3,105`` -> ``$3,105`` and ``(3 | )%`` -> ``(3)%``."""
    out: list[str] = []
    pending_currency = ""
    for cell in cells:
        if not cell:
            continue
        if cell in _CURRENCY:
            pending_currency = cell
            continue
        if out and _TRAILER.match(cell):
            out[-1] += cell
            continue
        out.append(pending_currency + cell)
        pending_currency = ""
    return out


def _table_blocks(table: _Element) -> list[Block]:
    """Render a table as a markdown block, or as paragraphs if it is a layout table.

    Tables of contents (several rows starting with "Item N") are dropped entirely, so ToC
    entries can never be mistaken for section headings.
    """
    rows: list[list[str]] = []
    row_heading: list[bool] = []
    for tr in table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr"):
        raw = [_cell_text(td) for td in tr if _tag(td) in {"td", "th"}]
        cells = _merge_cells([text for text, _ in raw])
        if cells:
            rows.append(cells)
            row_heading.append(all(h for text, h in raw if text))
    if not rows:
        return []

    item_rows = sum(1 for r in rows if _ITEM_CELL.match(r[0]))
    if item_rows >= 3 and item_rows / len(rows) >= 0.4:
        return []

    joined = [" ".join(r) for r in rows]
    if max(len(r) for r in rows) <= 1 or (len(rows) == 1 and len(rows[0]) <= 3):
        # Layout table: headings or paragraphs that happen to sit in cells.
        return [
            Block(
                "heading"
                if len(t) <= HEADING_MAX_CHARS and (h or _is_allcaps_heading(t))
                else "para",
                t,
            )
            for t, h in zip(joined, row_heading, strict=True)
        ]

    width = max(len(r) for r in rows)

    def fmt(r: list[str], left: bool = False) -> str:
        cells = [c.replace("|", "\\|") for c in r]
        pad = [""] * (width - len(r))
        return "| " + " | ".join(pad + cells if left else cells + pad) + " |"

    # Header rows usually omit the row-label column, so pad them on the left.
    lines = [fmt(rows[0], left=True), "| " + " | ".join(["---"] * width) + " |"]
    lines += [fmt(r) for r in rows[1:]]
    return [Block("table", "\n".join(lines))]


def html_to_blocks(raw: bytes | str) -> list[Block]:
    """Parse filing HTML (including inline XBRL) into ordered blocks.

    Hidden elements and the ``ix:header`` (hidden XBRL facts) are skipped; inline XBRL
    tags inside text are unwrapped naturally because only their text is kept.
    """
    text = decode_html(raw) if isinstance(raw, bytes) else raw
    text = _XML_DECL.sub("", text, count=1)
    if not text.strip():
        return []
    root = lxml_html.document_fromstring(text)
    builder = _Builder()
    builder.walk(root)
    builder.flush()
    return builder.blocks
