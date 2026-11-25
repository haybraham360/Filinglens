"""Embedding models behind a small interface (local sentence-transformers by default)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]

# bge v1.5 models embed queries with this instruction; documents are embedded as-is.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Turns text into L2-normalized vectors."""

    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        """Embed passages (chunk context header + text). Shape ``(n, dim)``."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a search query. Shape ``(dim,)``."""
        ...


class SentenceTransformerEmbedder:
    """Local embedding model via ``sentence-transformers`` (the ``local-embeddings`` extra)."""

    def __init__(self, model_name: str, *, batch_size: int = 32, device: str | None = None) -> None:
        """Load ``model_name`` lazily from the Hugging Face cache (downloads on first use)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "sentence-transformers is not installed. Run `uv sync --extra local-embeddings`."
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = SentenceTransformer(model_name, device=device)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"{model_name} does not report an embedding dimension")
        self.dim = int(dim)
        self._query_prefix = BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        """Embed passages in batches."""
        out = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(out, dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        """Embed a query with the model's query instruction, if it has one."""
        vec: Vector = self.embed_documents([self._query_prefix + text])[0]
        return vec


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder for tests and offline smoke runs.

    Not semantically meaningful beyond word overlap; never use it for reported results.
    """

    def __init__(self, dim: int = 384, model_name: str = "hashing-bow-v1") -> None:
        """Create an embedder producing ``dim``-dimensional vectors."""
        self.dim = dim
        self.model_name = model_name

    def _embed(self, text: str) -> Vector:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            vec[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        return np.asarray(vec / norm if norm else vec, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        """Embed each text independently."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, text: str) -> Vector:
        """Embed a query exactly like a document."""
        return self._embed(text)


def load_embedder(model_name: str) -> Embedder:
    """Embedder for ``model_name``; ``hashing-*`` names give :class:`HashingEmbedder`."""
    if model_name.startswith("hashing"):
        return HashingEmbedder(model_name=model_name)
    return SentenceTransformerEmbedder(model_name)
