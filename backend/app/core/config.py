"""Application settings, loaded from the environment with sane local defaults."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Core ---------------------------------------------------------------
    APP_NAME: str = "Nimbus SaaS Platform"
    ENVIRONMENT: Literal["development", "test", "staging", "production"] = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # -- Security -----------------------------------------------------------
    SECRET_KEY: str = "insecure-dev-secret-key-do-not-use-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14

    # -- Database -----------------------------------------------------------
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "nimbus"
    POSTGRES_PASSWORD: str = "nimbus"
    POSTGRES_DB: str = "nimbus"
    DATABASE_URL: str | None = None
    SQL_ECHO: bool = False

    # -- CORS ---------------------------------------------------------------
    BACKEND_CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )

    # -- LLM / RAG / LangGraph ----------------------------------------------
    LLM_PROVIDER: Literal["anthropic", "fake"] = "anthropic"
    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_CHAT_MODEL: str = "claude-3-5-haiku-20241022"
    EMBEDDING_PROVIDER: Literal["sentence_transformers", "fake"] = "sentence_transformers"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_DIMENSIONS: int = 384
    RAG_CHUNK_SIZE: int = 250
    RAG_CHUNK_OVERLAP: int = 40
    RAG_TOP_K: int = 5
    RAG_MIN_SCORE: float = 0.40
    DOCUMENT_RELEVANCE_THRESHOLD: float = 0.40

    # -- Online Web Search --------------------------------------------------
    WEB_SEARCH_ENABLED: bool = True
    WEB_SEARCH_PROVIDER: Literal["duckduckgo", "tavily", "fake"] = "duckduckgo"
    TAVILY_API_KEY: str | None = None
    WEB_SEARCH_MAX_RESULTS: int = 4

    # -- Hallucination Reduction / Self-RAG ---------------------------------
    ENABLE_HALLUCINATION_CHECK: bool = True
    MAX_REGENERATE_RETRIES: int = 2
    HALLUCINATION_PROVIDER: Literal["deberta", "llm", "fake"] = "deberta"
    DEBERTA_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-small"
    NLI_ENTAILMENT_THRESHOLD: float = 0.5
    NLI_CONTRADICTION_THRESHOLD: float = 0.3
    NLI_DEVICE: str = "cpu"

    # -- Conversation Memory & Sliding Window ------------------------------
    RAG_MEMORY_WINDOW_SIZE: int = 6
    RAG_ENABLE_QUERY_REWRITE: bool = True

    # -- Storage ------------------------------------------------------------
    STORAGE_DIR: str = "./storage"
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors(cls, value: Any) -> Any:
        """Accept either a JSON array or a comma-separated string."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                return json.loads(value)
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @model_validator(mode="after")
    def _assemble_database_url(self) -> Settings:
        if not self.DATABASE_URL:
            self.DATABASE_URL = (
                f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        return self

    @property
    def is_postgres(self) -> bool:
        return bool(self.DATABASE_URL and self.DATABASE_URL.startswith("postgresql"))

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed exactly once per process."""
    return Settings()


settings = get_settings()
