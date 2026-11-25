"""Citable sources collected during an agent run, each with a short handle.

Handles: ``C#`` = retrieved chunk, ``P#`` = paragraph or heading from a section diff,
``T#`` = a financial tool result. The model cites handles with verbatim quotes, and every
quote is checked against the source's text, so numbers from tools are cited (and
verified) exactly like filing text.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

SourceKind = Literal["chunk", "section", "tool"]
_PREFIX: dict[SourceKind, str] = {"chunk": "C", "section": "P", "tool": "T"}


@dataclass(frozen=True)
class Source:
    """One citable thing the agent has seen."""

    handle: str
    kind: SourceKind
    text: str
    cik: str | None = None
    ticker: str | None = None
    accession_no: str | None = None
    form: str | None = None
    fiscal_year: int | None = None
    item: str | None = None
    chunk_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None


@dataclass
class SourceRegistry:
    """Assigns stable handles; the same chunk or span always gets the same handle."""

    sources: dict[str, Source] = field(default_factory=dict)
    _keys: dict[tuple[Any, ...], str] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, kind: SourceKind, key: tuple[Any, ...], **fields: Any) -> str:
        """Register a source (idempotent on ``key``) and return its handle."""
        with self._lock:
            if key in self._keys:
                return self._keys[key]
            prefix = _PREFIX[kind]
            self._counts[prefix] = self._counts.get(prefix, 0) + 1
            handle = f"{prefix}{self._counts[prefix]}"
            self._keys[key] = handle
            self.sources[handle] = Source(handle=handle, kind=kind, **fields)
            return handle

    def add_tool_result(self, tool_name: str, args: dict[str, Any], result: Any) -> str:
        """Register a tool's output; its JSON text is what quotes are verified against."""
        text = json.dumps(result, sort_keys=True, default=str)
        key = ("tool", tool_name, json.dumps(args, sort_keys=True, default=str))
        return self.add(
            "tool",
            key,
            text=text,
            tool_name=tool_name,
            tool_args=args,
            ticker=args.get("ticker"),
        )

    def get(self, handle: str) -> Source | None:
        """Look up a handle (case-insensitive, tolerant of brackets)."""
        return self.sources.get(handle.strip("[] ").upper())
