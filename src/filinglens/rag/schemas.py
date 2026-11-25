"""Structured answer models: what the LLM returns, and what FilingLens returns."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["low", "medium", "high"]


class DraftCitation(BaseModel):
    """A citation as the model writes it: a context handle plus a verbatim quote."""

    ref: str = Field(description='Handle of the cited source, e.g. "C3", "P2" or "T1".')
    quote: str = Field(description="Short verbatim span copied exactly from that chunk.")


class AnswerDraft(BaseModel):
    """The model's structured output (validated by the SDK against this schema)."""

    answer: str = Field(description="Answer text with inline [C#]/[P#]/[T#] markers after claims.")
    citations: list[DraftCitation]
    insufficient_context: bool
    confidence: Confidence


class Citation(BaseModel):
    """A verified-or-flagged citation: a filing chunk, a section paragraph, or a tool result."""

    n: int = Field(description="Reference number used in the answer text, e.g. [1].")
    source: Literal["chunk", "section", "tool"] = "chunk"
    chunk_id: str | None = None
    accession_no: str | None = None
    item: str | None = None
    tool_name: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    ticker: str | None = None
    form: str | None = None
    fiscal_year: int | None = None
    quote: str
    verified: bool
    match_score: float
    filing_url: str | None = None


class Answer(BaseModel):
    """FilingLens's answer (specs/03): text, citations, and confidence."""

    answer: str
    citations: list[Citation]
    insufficient_context: bool
    confidence: Confidence
    citation_validity: float | None = Field(
        default=None, description="Share of citations whose quote was found in the chunk."
    )
