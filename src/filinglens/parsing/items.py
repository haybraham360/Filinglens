"""Canonical 10-K / 10-Q item keys, titles and heading patterns.

Item keys: 10-K items are bare codes (``"1A"``, ``"7"``). 10-Q item numbers repeat across
parts, so 10-Q keys are part-qualified (``"I-2"`` is MD&A, ``"II-1A"`` is Risk Factors).
"""

from __future__ import annotations

import re

TEN_K_ITEMS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

TEN_Q_ITEMS: dict[str, str] = {
    "I-1": "Financial Statements",
    "I-2": "Management's Discussion and Analysis",
    "I-3": "Quantitative and Qualitative Disclosures About Market Risk",
    "I-4": "Controls and Procedures",
    "II-1": "Legal Proceedings",
    "II-1A": "Risk Factors",
    "II-2": "Unregistered Sales of Equity Securities and Use of Proceeds",
    "II-3": "Defaults Upon Senior Securities",
    "II-4": "Mine Safety Disclosures",
    "II-5": "Other Information",
    "II-6": "Exhibits",
}

# Items whose absence or shortness signals a parse failure (see quality report).
REQUIRED_ITEMS: dict[str, tuple[str, ...]] = {"10-K": ("1A", "7"), "10-Q": ("I-2",)}
QUALITY_ITEMS: dict[str, tuple[str, ...]] = {
    "10-K": ("1", "1A", "3", "7", "7A"),
    "10-Q": ("I-2", "II-1A"),
}
MIN_CHARS: dict[str, int] = {"1A": 500, "7": 1000, "I-2": 500}

ITEM_HEADING = re.compile(
    r"^(?:part\s+(?P<part>iv|i{1,3})\b[\s,.:\u2013\u2014-]*)?"
    r"item\s*(?P<num>\d{1,2})\s*(?P<letter>[a-c])?(?![a-z0-9])\s*[.:\u2013\u2014-]?\s*(?P<title>.*)$",
    re.I,
)
PART_HEADING = re.compile(r"^part\s+(?P<part>iv|i{1,3})\b", re.I)


def base_form(form: str) -> str:
    """``10-K/A`` -> ``10-K``; also maps ``10-K405``/``10-KT`` to ``10-K``."""
    form = form.upper().removesuffix("/A")
    return "10-Q" if form.startswith("10-Q") else "10-K"


def canonical_items(form: str) -> dict[str, str]:
    """Ordered ``{item_key: title}`` for a form."""
    return TEN_Q_ITEMS if base_form(form) == "10-Q" else TEN_K_ITEMS


def item_key(form: str, code: str, part: str | None) -> str:
    """Build the item key for ``code`` (e.g. ``"1A"``) in ``part`` (``"I"``/``"II"``)."""
    code = code.upper()
    if base_form(form) == "10-Q":
        return f"{(part or 'I').upper()}-{code}"
    return code


def item_label(form: str, key: str) -> str:
    """Human label: ``Item 1A Risk Factors`` or ``Part II Item 1A Risk Factors``."""
    title = canonical_items(form).get(key, "")
    if "-" in key:
        part, code = key.split("-", 1)
        return f"Part {part} Item {code} {title}".strip()
    return f"Item {key} {title}".strip()
