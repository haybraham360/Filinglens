"""Text normalization and token counting shared by the parser and chunker."""

from __future__ import annotations

import re
import unicodedata

# Legacy EDGAR HTML often encodes Windows-1252 punctuation as C1 control characters
# (e.g. ``&#146;`` for an apostrophe). Map them before NFKC.
_C1_MAP = {
    "\u0091": "'",
    "\u0092": "'",
    "\u0093": '"',
    "\u0094": '"',
    "\u0095": "\u2022",
    "\u0096": "\u2013",
    "\u0097": "\u2014",
    "\u0085": "...",
}
_QUOTE_MAP = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2032": "'"}
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"))
_TRANSLATE = str.maketrans({**_C1_MAP, **_QUOTE_MAP}) | _ZERO_WIDTH
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+|[^\w\s]")


def normalize_text(text: str) -> str:
    """Normalize unicode (NFKC, ASCII quotes, no zero-width chars) and collapse whitespace."""
    text = text.translate(_TRANSLATE)
    text = unicodedata.normalize("NFKC", text)
    return _WS.sub(" ", text).strip()


def count_tokens(text: str) -> int:
    """Approximate token count: words plus punctuation marks.

    Deterministic and dependency-free. It runs slightly below BPE/WordPiece counts on
    numeric-heavy text, so chunk limits are set with headroom below the embedding
    model's maximum sequence length.
    """
    return len(_TOKEN.findall(text))
