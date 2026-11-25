"""Record every LLM call (model, tokens, latency, cost) and enforce the daily spend cap."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import psycopg


@dataclass(frozen=True)
class CallRecord:
    """One LLM request, as stored in ``llm_calls``."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float | None
    purpose: str
    prompt_version: str
    stop_reason: str | None
    trace_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class CallLog(Protocol):
    """Where call records go."""

    def record(self, call: CallRecord) -> None:
        """Persist one call."""
        ...

    def spent_today_usd(self) -> float:
        """Total logged cost since 00:00 UTC."""
        ...


class SpendCapExceededError(RuntimeError):
    """The daily LLM spend cap (``LLM_DAILY_SPEND_CAP_USD``) has been reached."""


class PostgresCallLog:
    """Writes to the ``llm_calls`` table."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        """Use an open connection (schema already applied)."""
        self.conn = conn

    def record(self, call: CallRecord) -> None:
        """Insert one row (autocommitted in its own transaction)."""
        with self.conn.transaction():
            self.conn.execute(
                "INSERT INTO llm_calls (created_at, trace_id, model, input_tokens, output_tokens,"
                " latency_ms, cost_usd, purpose, prompt_version, stop_reason)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    call.created_at,
                    call.trace_id,
                    call.model,
                    call.input_tokens,
                    call.output_tokens,
                    call.latency_ms,
                    call.cost_usd,
                    call.purpose,
                    call.prompt_version,
                    call.stop_reason,
                ),
            )

    def spent_today_usd(self) -> float:
        """Sum of ``cost_usd`` since midnight UTC."""
        row = self.conn.execute(
            "SELECT coalesce(sum(cost_usd), 0) FROM llm_calls "
            "WHERE created_at >= date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'"
        ).fetchone()
        return float(row[0]) if row else 0.0


class MemoryCallLog:
    """In-process log for tests and ``--no-db`` runs."""

    def __init__(self) -> None:
        """Start empty."""
        self.calls: list[CallRecord] = []

    def record(self, call: CallRecord) -> None:
        """Keep the record in memory."""
        self.calls.append(call)

    def spent_today_usd(self) -> float:
        """Sum of today's recorded costs."""
        today = datetime.now(UTC).date()
        return sum(c.cost_usd or 0.0 for c in self.calls if c.created_at.date() == today)
