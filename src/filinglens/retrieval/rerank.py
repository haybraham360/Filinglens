"""Cross-encoder reranking of retrieval candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol

from filinglens.retrieval.search import RetrievedChunk


class Reranker(Protocol):
    """Scores (query, passage) pairs; higher is more relevant."""

    model_name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance score per passage."""
        ...


class CrossEncoderReranker:
    """Local cross-encoder (default ``BAAI/bge-reranker-base``) via sentence-transformers."""

    def __init__(self, model_name: str, *, batch_size: int = 16) -> None:
        """Load the model (downloads on first use)."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "sentence-transformers is not installed. Run `uv sync --extra local-embeddings`."
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = CrossEncoder(model_name)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score each passage against ``query``."""
        if not passages:
            return []
        scores = self._model.predict(
            [(query, p) for p in passages], batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


def rerank(
    reranker: Reranker, query: str, candidates: Sequence[RetrievedChunk], top_n: int
) -> list[RetrievedChunk]:
    """Reorder ``candidates`` by reranker score and keep ``top_n``."""
    scores = reranker.score(query, [c.passage for c in candidates])
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda cs: (-cs[1], cs[0].chunk_id))
    return [
        replace(chunk, score=score, ranks={**chunk.ranks, "rerank": i + 1})
        for i, (chunk, score) in enumerate(ranked[:top_n])
    ]
