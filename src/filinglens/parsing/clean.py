"""Remove page furniture (page numbers, running headers/footers, ToC back-links) from blocks."""

from __future__ import annotations

import re
from collections import defaultdict

from filinglens.parsing.blocks import Block

_PAGE_NUMBER = re.compile(
    r"^(?:page\s+)?[-\u2013\u2014(]?\s*(?:\d{1,3}|[ivxlc]{1,6}|[a-z]-\d{1,3})\s*[-\u2013\u2014)]?$",
    re.I,
)
_TOC_LINK = re.compile(r"^(?:back to |return to )?(?:table of contents|index|toc)\.?$", re.I)
_FORM_NAME = re.compile(r"form\s+10-?[kq]\b", re.I)
_DIGITS = re.compile(r"\d+")
FURNITURE_MAX_CHARS = 150


def is_page_furniture(text: str) -> bool:
    """True for a bare page number (``12``, ``- 5 -``, ``ii``, ``F-3``) or a ToC back-link."""
    return bool(_PAGE_NUMBER.match(text) or _TOC_LINK.match(text))


def _mask(text: str) -> str:
    return _DIGITS.sub("#", text.lower())


def running_furniture(blocks: list[Block]) -> frozenset[str]:
    """Digit-masked texts of running headers/footers, detected across the whole document.

    Running footers are short lines repeated across pages with only the digits changing
    (``Acme Inc. | 2023 Form 10-K | 14``). A short line that repeats *identically* is only
    treated as furniture if it names the form, so legitimately repeated sub-headings such
    as ``Fiscal 2023 Compared to Fiscal 2022`` survive.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for block in blocks:
        if block.kind != "table" and len(block.text) <= FURNITURE_MAX_CHARS:
            groups[_mask(block.text)].append(block.text)
    running = set()
    for mask, texts in groups.items():
        if len(texts) < 3:
            continue
        varying_digits = "#" in mask and len(set(texts)) >= 3
        if varying_digits or _FORM_NAME.search(mask):
            running.add(mask)
    return frozenset(running)


def remove_page_furniture(
    blocks: list[Block], running: frozenset[str] | None = None
) -> list[Block]:
    """Drop page numbers, ToC back-links and running headers/footers.

    ``running`` comes from :func:`running_furniture` on the full document; pass it when
    cleaning a fragment (one section), where repetition can't be detected locally.
    """
    if running is None:
        running = running_furniture(blocks)
    return [
        b
        for b in blocks
        if b.kind == "table"
        or not (
            is_page_furniture(b.text)
            or (len(b.text) <= FURNITURE_MAX_CHARS and _mask(b.text) in running)
        )
    ]
