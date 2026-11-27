"""Application settings, loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Never hardcode keys or emails; set them in `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SEC EDGAR
    edgar_identity: str = Field(default="", description="User-Agent: 'Name email@example.com'")
    edgar_max_rps: float = Field(default=8.0, gt=0, le=10)

    # Data
    data_dir: Path = Path("./data")
    universe_file: Path = Path("./config/pilot_universe.csv")
    start_year: int = Field(default=2015, ge=1994)
    forms: Annotated[list[str], NoDecode] = ["10-K", "10-Q"]

    # Chunking (specs/02; final values chosen by evals in specs/05)
    chunk_max_tokens: int = Field(default=448, gt=0)
    chunk_overlap_tokens: int = Field(default=48, ge=0)
    chunk_min_tokens: int = Field(default=64, ge=0)
    chunk_strategy: Literal["structure", "fixed"] = "structure"

    # Database
    database_url: str = "postgresql://filinglens:filinglens@localhost:5433/filinglens"

    # LLM
    # "auto": Claude if ANTHROPIC_API_KEY is set, else Ollama if it is running, else no LLM
    # (extractive answers and the rule-based tool router; free and deterministic).
    llm_provider: Literal["auto", "anthropic", "ollama", "none"] = "auto"
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    llm_daily_spend_cap_usd: float = Field(default=5.0, ge=0)

    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = Field(default=16000, gt=0)
    agent_max_steps: int = Field(default=6, ge=2, le=12)

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(default=384, gt=0)
    embedding_backend: Literal["fastembed", "sentence-transformers"] = "fastembed"

    # Retrieval (specs/03; eval-tunable)
    retrieval_top_k: int = Field(default=8, gt=0)
    retrieval_candidate_k: int = Field(default=40, gt=0)
    use_hybrid: bool = True
    rrf_k: int = Field(default=60, gt=0)
    use_rerank: bool = False
    rerank_top_n: int = Field(default=8, gt=0)
    rerank_model: str = "BAAI/bge-reranker-base"

    # API
    api_cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:8501"]
    # Extra allowed origins as one regex, e.g. Vercel preview deployments of the frontend.
    api_cors_origin_regex: str | None = None
    api_rate_limit_per_minute: int = Field(default=12, gt=0)
    api_rate_limit_high_per_minute: int = Field(default=3, gt=0)
    api_rate_limit_other_per_minute: int = Field(default=120, gt=0)
    api_trust_proxy: bool = False  # use X-Forwarded-For (only behind a proxy you control)
    api_cache_ttl_s: int = Field(default=86_400, ge=0)
    api_max_question_chars: int = Field(default=500, gt=10)
    db_pool_size: int = Field(default=5, gt=0)
    # Cheaper Claude model for the public demo; "high" quality uses LLM_MODEL.
    demo_llm_model: str = "claude-haiku-5-5"

    @field_validator("anthropic_api_key", "api_cors_origin_regex", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        """``ANTHROPIC_API_KEY=`` (empty, as in .env.example) means no key."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("forms", "api_cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept comma-separated strings from env vars as lists."""
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @property
    def raw_dir(self) -> Path:
        """Directory for raw EDGAR downloads."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directory for parsed sections and chunks."""
        return self.data_dir / "processed"

    @property
    def log_dir(self) -> Path:
        """Directory for query logs (latency, citation validity)."""
        return self.data_dir / "logs"

    @property
    def cache_dir(self) -> Path:
        """Directory for the on-disk HTTP cache."""
        return self.data_dir / "cache"

    def require_edgar_identity(self) -> str:
        """Return the EDGAR User-Agent, or raise if it is not configured."""
        if not self.edgar_identity.strip():
            raise ValueError(
                "EDGAR_IDENTITY is not set. Add 'Your Name your@email.com' to .env; "
                "SEC rejects requests without a declared User-Agent."
            )
        return self.edgar_identity


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
