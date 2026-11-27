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


class FastEmbedEmbedder:
    """ONNX runtime embeddings via ``fastembed`` (default backend; no PyTorch).

    Produces the same vectors as sentence-transformers for ``BAAI/bge-small-en-v1.5``
    (cosine 1.0000 in a side-by-side check), with ~150 MB of dependencies instead of 1 GB+
    and ~360 MB peak RAM, so the API fits small hosting tiers.
    """

    def __init__(
        self, model_name: str, *, batch_size: int = 32, cache_dir: str | None = None
    ) -> None:
        """Load ``model_name`` (downloads ~65 MB on first use)."""
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = TextEmbedding(model_name, cache_dir=cache_dir)
        self.dim = int(self._model.embedding_size)
        self._query_prefix = BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        """Embed passages, L2-normalized."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.asarray(
            list(self._model.embed(list(texts), batch_size=self.batch_size)), dtype=np.float32
        )
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.asarray(out / np.where(norms == 0, 1, norms), dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        """Embed a query with the bge query instruction (as sentence-transformers does)."""
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


_CACHE: dict[tuple[str, str], Embedder] = {}


def load_embedder(model_name: str, backend: str | None = None) -> Embedder:
    """Embedder for ``model_name`` (cached per process).

    ``backend``: ``fastembed`` (default, ONNX) or ``sentence-transformers``; defaults to the
    ``EMBEDDING_BACKEND`` setting. ``hashing-*`` names give :class:`HashingEmbedder`.
    """
    if model_name.startswith("hashing"):
        return HashingEmbedder(model_name=model_name)
    if backend is None:
        from filinglens.config import get_settings

        backend = get_settings().embedding_backend
    key = (backend, model_name)
    if key not in _CACHE:
        _CACHE[key] = (
            SentenceTransformerEmbedder(model_name)
            if backend == "sentence-transformers"
            else FastEmbedEmbedder(model_name)
        )
    return _CACHE[key]
