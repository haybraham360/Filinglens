"""Eval run configuration (``evals/configs/*.yaml``)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from filinglens.parsing.pipeline import PARSER_VERSION


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IndexConfig(_Strict):
    """Index-time settings: changing any of these builds a separate index (own DB schema)."""

    chunk_max_tokens: int = 448
    chunk_overlap_tokens: int = 48
    chunk_min_tokens: int = 64
    chunk_strategy: Literal["structure", "fixed"] = "structure"
    context_header: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    def fingerprint(self) -> str:
        """Short stable id for this index (includes the parser version)."""
        payload = json.dumps(self.model_dump(), sort_keys=True) + PARSER_VERSION
        return hashlib.sha256(payload.encode()).hexdigest()[:10]


class RetrievalConfig(_Strict):
    """Query-time retrieval knobs (specs/03)."""

    top_k: int = 8
    candidate_k: int = 40
    use_hybrid: bool = True
    rrf_k: int = 60
    use_rerank: bool = False
    rerank_top_n: int = 8
    rerank_model: str = "BAAI/bge-reranker-base"
    use_filters: bool = True


class ModelConfig(_Strict):
    """A model, where it runs, and (Claude only) its effort level.

    ``provider: none`` means no LLM: extractive answers / the rule-based agent for ``llm``,
    and no judge (programmatic metrics only) for ``judge``.
    """

    model: str
    provider: Literal["anthropic", "ollama", "none"] = "anthropic"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"

    @property
    def served_name(self) -> str | None:
        """The model name responses should report (``None`` when there is no LLM)."""
        if self.provider == "none":
            return None
        return f"ollama/{self.model}" if self.provider == "ollama" else self.model


class EvalConfig(_Strict):
    """One eval configuration.

    ``mode``: ``retrieval`` scores retrieval only (no LLM, free), ``rag`` runs
    ``answer_question`` (specs/03), ``agent`` runs the tool agent (specs/04).
    """

    name: str
    description: str = ""
    mode: Literal["retrieval", "rag", "agent"] = "rag"
    index: IndexConfig = IndexConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: ModelConfig = ModelConfig(model="claude-sonnet-5-5")
    judge: ModelConfig = ModelConfig(model="claude-opus-5-5")
    reps: int = 1
    concurrency: int = 2
    case_timeout_s: float = 240.0
    retrieval_eval_k: int = 10


def load_config(path: Path) -> EvalConfig:
    """Read a YAML config; the file name stem is the default ``name``."""
    raw = yaml.safe_load(path.read_text()) or {}
    raw.setdefault("name", path.stem)
    return EvalConfig.model_validate(raw)
