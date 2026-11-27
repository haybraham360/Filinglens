"""Structured JSON logging with request and trace ids on every line."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import ClassVar

from starlette.requests import Request
from starlette.responses import Response

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
logger = logging.getLogger("filinglens.api")


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg, request_id, trace_id, extras."""

    _RESERVED: ClassVar[frozenset[str]] = frozenset(
        vars(logging.LogRecord("", 0, "", 0, "", None, None))
    )

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as JSON."""
        out: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "trace_id": trace_id_var.get(),
        }
        out |= {k: v for k, v in vars(record).items() if k not in self._RESERVED and k != "message"}
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Route all logs through the JSON formatter (idempotent)."""
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Assign a request id (honouring a safe incoming ``X-Request-ID``) and log the request."""
    incoming = request.headers.get("x-request-id", "")
    rid = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
    request_id_var.set(rid)
    trace_id_var.set(None)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        trace_id = getattr(request.state, "trace_id", None)
        if trace_id:
            trace_id_var.set(trace_id)
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": int((time.perf_counter() - start) * 1000),
            },
        )
    response.headers["X-Request-ID"] = rid
    trace_id = getattr(request.state, "trace_id", None)
    if trace_id:
        response.headers["X-Trace-ID"] = trace_id
    return response
