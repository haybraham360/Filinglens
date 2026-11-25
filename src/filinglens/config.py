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
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"
    llm_daily_spend_cap_usd: float = Field(default=5.0, ge=0)

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(default=384, gt=0)

    # API
    api_cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:8501"]

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
