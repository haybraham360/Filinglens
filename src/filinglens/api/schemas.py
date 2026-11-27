"""Request/response models (these generate the OpenAPI docs at /docs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from filinglens.rag.schemas import Answer

Ticker = Field(default=None, pattern=r"^[A-Za-z][A-Za-z.\-]{0,9}$", description="e.g. AAPL")
Year = Field(default=None, ge=1994, le=2100)


class Health(BaseModel):
    """Liveness response."""

    status: str
    version: str


class Ready(BaseModel):
    """Readiness: database reachable, which LLM provider answers."""

    status: Literal["ok", "degraded"]
    database: bool
    provider: str
    chunks: int


class AskRequest(BaseModel):
    """A question, optionally pinned to one company and fiscal year."""

    question: str = Field(min_length=3, max_length=2000)
    ticker: str | None = Ticker
    fiscal_year: int | None = Year
    mode: Literal["agent", "rag"] = Field(
        default="agent",
        description="agent: tools (search, diff, financials, ratios); rag: retrieval + answer.",
    )
    quality: Literal["standard", "high"] = Field(
        default="standard",
        description="high uses the stronger model (stricter rate limit) when Claude is configured.",
    )


class StepOut(BaseModel):
    """One agent step, for the "How I got this answer" panel."""

    index: int
    kind: str
    name: str
    latency_ms: int
    args: dict[str, Any] | None = None
    is_error: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None


class AskResponse(BaseModel):
    """An answer with citations plus how it was produced."""

    answer: Answer
    trace_id: str
    mode: str
    provider: str
    model: str | None
    cached: bool = False
    degraded: bool = Field(default=False, description="True if the spend cap forced free mode.")
    latency_ms: int
    tools_called: list[str] = []
    steps: list[StepOut] = []
    unsupported_numbers: list[str] = []


class CompanyOut(BaseModel):
    """An ingested company and the fiscal years available."""

    ticker: str
    cik: str
    name: str | None
    years_10k: list[int]
    years_10q: list[int]


class CompareRequest(BaseModel):
    """Which 10-K item to diff between two fiscal years."""

    ticker: str = Field(pattern=r"^[A-Za-z][A-Za-z.\-]{0,9}$")
    item: str = Field(default="1A", pattern=r"^[0-9]{1,2}[A-Ca-c]?$")
    year_a: int | None = Year
    year_b: int | None = Year


class MetricRow(BaseModel):
    """One metric across years. Values are exact decimal strings from the tools."""

    metric: str
    label: str
    kind: Literal["usd", "ratio"]
    values: dict[str, str | None]
    formula: str | None = None


class FinancialsResponse(BaseModel):
    """Key metrics table for one company."""

    ticker: str
    years: list[int]
    rows: list[MetricRow]
    sources: list[dict[str, Any]]


class ChunkOut(BaseModel):
    """Source text behind a citation."""

    chunk_id: str
    accession_no: str
    ticker: str | None
    form: str
    fiscal_year: int | None
    item: str
    heading: str | None
    text: str
    char_start: int | None
    char_end: int | None
    filing_url: str


class SectionSpanOut(BaseModel):
    """A passage from a parsed section (for diff citations), with surrounding context."""

    accession_no: str
    item: str
    before: str
    text: str
    after: str
    filing_url: str


class TraceOut(BaseModel):
    """A stored request trace."""

    trace_id: str
    created_at: str
    question: str
    mode: str
    payload: dict[str, Any]
