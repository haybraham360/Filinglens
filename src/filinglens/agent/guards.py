"""Check that numbers in an answer come from tool results or filing text, not the LLM.

A figure in the answer is *supported* if some number in a source matches it after the
usual presentation changes: rounding, scaling to thousands/millions/billions, or showing
a ratio as a percentage. Anything else is reported, which is how evals check the rule
"no numeric answer without a tool call" (specs/04).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation

_NUMBER = re.compile(
    r"(?<![\w.\[])(?P<neg>[-\u2212])?\$?\(?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\)?"
    r"(?P<pct>\s?%|\s?percent)?"
    r"(?:\s?(?P<scale>trillion|billion|million|thousand|bn|mm|m|k)\b)?",
    re.I,
)
_SCALES = {
    "trillion": Decimal(10) ** 12,
    "billion": Decimal(10) ** 9,
    "bn": Decimal(10) ** 9,
    "million": Decimal(10) ** 6,
    "mm": Decimal(10) ** 6,
    "m": Decimal(10) ** 6,
    "thousand": Decimal(10) ** 3,
    "k": Decimal(10) ** 3,
}
_PRESENTATION = (
    Decimal(1),
    Decimal(100),
    Decimal("0.001"),
    Decimal("0.000001"),
    Decimal("1e-9"),
    Decimal("1e-12"),
)
TOLERANCE = Decimal("0.006")


def _numbers(text: str, *, for_answer: bool) -> list[tuple[str, Decimal]]:
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0).strip()
        try:
            value = Decimal(m.group("num").replace(",", ""))
        except InvalidOperation:
            continue
        if m.group("scale"):
            value *= _SCALES[m.group("scale").lower()]
        if m.group("neg") or (raw.startswith("(") and raw.endswith(")")):
            value = -value
        if for_answer:
            before = text[max(0, m.start() - 6) : m.start()].lower()
            plain_int = "," not in m.group("num") and "." not in m.group("num")
            if before.rstrip().endswith(("item", "fy", "q", "part", "note")):
                continue
            if plain_int and not m.group("pct") and not m.group("scale"):
                n = int(m.group("num"))
                if 1990 <= n <= 2040 or n <= 10:  # years, small counts/ordinals
                    continue
        out.append((raw, value))
    return out


def _matches(answer_value: Decimal, source_value: Decimal) -> bool:
    for factor in _PRESENTATION:
        target = source_value * factor
        if target == 0:
            if answer_value == 0:
                return True
            continue
        if abs(answer_value - target) / abs(target) <= TOLERANCE:
            return True
    return False


def unsupported_numbers(answer: str, sources: Iterable[str]) -> list[str]:
    """Numbers in ``answer`` that no source text supports (in order, deduplicated)."""
    answer = re.sub(r"\[\d+\]", " ", answer)  # citation markers
    source_values = [v for text in sources for _, v in _numbers(text, for_answer=False)]
    flagged: list[str] = []
    for raw, value in _numbers(answer, for_answer=True):
        if not any(_matches(value, s) for s in source_values) and raw not in flagged:
            flagged.append(raw)
    return flagged
