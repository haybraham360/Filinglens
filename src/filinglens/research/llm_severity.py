"""``llm_change_severity``: an LLM rates how material the year-over-year changes are.

This signal is **at risk of pretraining leakage** (specs/07). A model may know what
happened to the company after the filing. Safeguards:

1. **Anonymization.** Company names, tickers and named people are masked before the text
   is sent (:func:`anonymize`). This is best-effort: distinctive details such as products,
   places and numbers can still identify a firm.
2. **Cutoff gate.** Filings released before the model's training cutoff are skipped,
   unless ``allow_in_sample`` is set. Then they are flagged ``in_sample`` and reported
   separately, never mixed into out-of-sample results.
3. Only the added and removed risk-factor passages are sent, not the whole filing.

Off by default and never run in CI. It costs money with Claude. With Ollama it is free
but slow.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from filinglens.agent.diff import align, units
from filinglens.llm.prompts import load_prompt
from filinglens.llm.provider import StructuredLLM
from filinglens.research.signals import load_sections

logger = logging.getLogger(__name__)

PROMPT_NAME = "change_severity"
MAX_CHARS = 12_000

_HONORIFIC = re.compile(r"\b(?:Mr|Ms|Mrs|Dr|Messrs)\.?\s+(?:[A-Z][a-z'\-]+\s?){1,3}")
_SUFFIX = r"(?:,?\s+(?:Inc|Incorporated|Corp|Corporation|Co|Company|Ltd|PLC|LLC|N\.V|S\.A)\.?)?"


class Severity(BaseModel):
    """Structured LLM output."""

    severity: int = Field(ge=1, le=5)
    rationale: str


def anonymize(
    text: str,
    *,
    company: Sequence[str] = (),
    tickers: Sequence[str] = (),
    people: Sequence[str] = (),
) -> str:
    """Mask company names (with legal suffixes), tickers and people's names."""
    out = text
    for name in sorted({n for n in company if n}, key=len, reverse=True):
        core = re.escape(re.sub(_SUFFIX + "$", "", name, flags=re.I).strip())
        out = re.sub(rf"\b{core}{_SUFFIX}", "[COMPANY]", out, flags=re.I)
    for t in {t for t in tickers if t}:
        out = re.sub(rf"\b{re.escape(t)}\b", "[TICKER]", out)
    for p in sorted({p for p in people if p}, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(p)}\b", "[PERSON]", out)
    return _HONORIFIC.sub("[PERSON] ", out)


@dataclass(frozen=True)
class SeverityConfig:
    """Model cutoff and how to treat earlier filings."""

    training_cutoff: date
    allow_in_sample: bool = False
    item: str = "1A"


def changed_passages(old_text_units: list[Any], new_text_units: list[Any]) -> tuple[str, str]:
    """Added and removed paragraphs (modified ones count as both), each truncated."""
    diff = align(old_text_units, new_text_units)
    added = [u.text for u in diff.added] + [n.text for _, n, _ in diff.modified]
    removed = [u.text for u in diff.removed] + [o.text for o, _, _ in diff.modified]
    return "\n\n".join(added)[:MAX_CHARS], "\n\n".join(removed)[:MAX_CHARS]


def run_llm_severity(
    pairs: pd.DataFrame,
    processed_dir: Any,
    llm: StructuredLLM,
    cfg: SeverityConfig,
    *,
    company_names: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Rate each pair. Returns ``accession_no, llm_change_severity, in_sample, model``."""
    prompt = load_prompt(PROMPT_NAME)
    cutoff = pd.Timestamp(cfg.training_cutoff, tz="UTC")
    rows = []
    for pair in pairs.to_dict("records"):
        in_sample = pd.Timestamp(pair["acceptance_datetime"]) < cutoff
        if in_sample and not cfg.allow_in_sample:
            continue
        old = load_sections(processed_dir, str(pair["cik"]), str(pair["prev_accession_no"]))
        new = load_sections(processed_dir, str(pair["cik"]), str(pair["accession_no"]))
        if not old or not new or cfg.item not in old or cfg.item not in new:
            continue
        added, removed = changed_passages(
            units(old[cfg.item], ("para",)), units(new[cfg.item], ("para",))
        )
        names = (company_names or {}).get(str(pair["cik"]), [])
        mask = {"company": names, "tickers": [str(pair.get("ticker") or "")]}
        user = (
            "<added_passages>\n" + anonymize(added, **mask) + "\n</added_passages>\n\n"
            "<removed_passages>\n" + anonymize(removed, **mask) + "\n</removed_passages>"
        )
        res = llm.generate(
            system=prompt.text,
            user=user,
            schema=Severity,
            purpose="research.change_severity",
            prompt_version=prompt.id,
        )
        if res.parsed is None:
            logger.warning("no severity for %s (%s)", pair["accession_no"], res.stop_reason)
            continue
        rows.append(
            {
                "accession_no": pair["accession_no"],
                "llm_change_severity": res.parsed.severity,
                "in_sample": in_sample,
                "model": res.model,
                "prompt_version": prompt.id,
            }
        )
    return pd.DataFrame(rows)
