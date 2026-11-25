"""On-disk HTTP response cache keyed by URL."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically, so an interrupted run never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class CachedResponse:
    """A response body and status, either fresh from the network or from disk."""

    url: str
    status_code: int
    content: bytes
    from_cache: bool

    def json(self) -> object:
        """Decode the body as JSON."""
        return json.loads(self.content)


class DiskCache:
    """Stores response bodies under ``root/<sha[:2]>/<sha>.body`` with a JSON sidecar."""

    def __init__(self, root: Path) -> None:
        """Create a cache rooted at ``root`` (created lazily)."""
        self.root = root

    def _paths(self, url: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(url.encode()).hexdigest()
        base = self.root / digest[:2] / digest
        return base.with_suffix(".body"), base.with_suffix(".meta.json")

    def get(self, url: str) -> CachedResponse | None:
        """Return the cached response for ``url``, or ``None`` if absent."""
        body_path, meta_path = self._paths(url)
        if not (body_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        return CachedResponse(
            url=url,
            status_code=int(meta["status_code"]),
            content=body_path.read_bytes(),
            from_cache=True,
        )

    def put(self, url: str, status_code: int, content: bytes) -> None:
        """Store a response. The body is written before the sidecar that marks it complete."""
        body_path, meta_path = self._paths(url)
        atomic_write_bytes(body_path, content)
        meta = {
            "url": url,
            "status_code": status_code,
            "fetched_at": datetime.now(UTC).isoformat(),
            "bytes": len(content),
        }
        atomic_write_bytes(meta_path, json.dumps(meta).encode())

    def delete(self, url: str) -> None:
        """Remove a cached entry if present."""
        for path in self._paths(url):
            path.unlink(missing_ok=True)
