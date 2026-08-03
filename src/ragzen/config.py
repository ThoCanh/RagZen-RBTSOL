"""RagZen configuration system.

Supports:
- Python objects (Pydantic models)
- YAML files
- Environment variable overrides (RAGZEN_* prefix)
- Secret provider interface
- Startup validation with fail-fast

All sensitive fields use SecretStr to prevent accidental logging.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from ragzen.exceptions import ConfigurationError

logger = logging.getLogger("ragzen.config")


# --- Secret Provider ---


class SecretProvider:
    """Interface for secret providers.

    Subclass to integrate with vault systems, AWS Secrets Manager, etc.
    """

    def get_secret(self, key: str) -> str | None:
        """Retrieve a secret by key. Returns None if not found."""
        return None  # pragma: no cover


class EnvironmentSecretProvider(SecretProvider):
    """Reads secrets from environment variables."""

    def __init__(self, prefix: str = "RAGZEN_") -> None:
        self.prefix = prefix

    def get_secret(self, key: str) -> str | None:
        env_key = f"{self.prefix}{key.upper()}"
        return os.environ.get(env_key)


# --- Sub-configurations ---


class StorageConfig(BaseModel):
    """Document registry storage configuration."""

    model_config = ConfigDict(frozen=True)

    provider: str = "sqlite"
    path: str = ".ragzen/documents.db"


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""

    model_config = ConfigDict(frozen=True)

    provider: str = "sentence_transformers"
    model: str = "all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = Field(default=32, ge=1, le=4096)
    normalize: bool = True
    timeout_seconds: float = Field(default=60.0, gt=0)
    dimensions: int | None = None
    cache_enabled: bool = True


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    model_config = ConfigDict(frozen=True)

    provider: str = "memory"
    url: str = ""
    collection: str = "documents"
    timeout_seconds: float = Field(default=10.0, gt=0)
    api_key: SecretStr | None = None


class SparseIndexConfig(BaseModel):
    """Sparse index configuration."""

    model_config = ConfigDict(frozen=True)

    provider: str = "bm25"
    path: str = ".ragzen/bm25"


class RetrievalConfig(BaseModel):
    """Retrieval pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    mode: str = "hybrid"  # dense, sparse, hybrid
    top_k_dense: int = Field(default=30, ge=1)
    top_k_sparse: int = Field(default=30, ge=1)
    fusion: str = "rrf"  # rrf, weighted
    rerank_top_k: int = Field(default=12, ge=1)
    final_top_k: int = Field(default=6, ge=1)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid_modes = {"dense", "sparse", "hybrid"}
        if v not in valid_modes:
            msg = f"Invalid retrieval mode: {v}. Must be one of: {valid_modes}"
            raise ValueError(msg)
        return v

    @field_validator("fusion")
    @classmethod
    def validate_fusion(cls, v: str) -> str:
        valid_fusions = {"rrf", "weighted"}
        if v not in valid_fusions:
            msg = f"Invalid fusion strategy: {v}. Must be one of: {valid_fusions}"
            raise ValueError(msg)
        return v


class RerankerConfig(BaseModel):
    """Reranker configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    provider: str = "cross_encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_documents: int = Field(default=50, ge=1)


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    model_config = ConfigDict(frozen=True)

    provider: str = "openai_compatible"
    base_url: str = "http://localhost:11434/v1"
    api_key: SecretStr = SecretStr("")
    model: str = "qwen2.5"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    concurrency_limit: int = Field(default=10, ge=1)
    streaming: bool = True


class SecurityConfig(BaseModel):
    """Security configuration."""

    model_config = ConfigDict(frozen=True)

    require_security_context: bool = True
    fail_closed: bool = True
    prompt_injection_screening: bool = True
    max_file_size_mb: float = Field(default=100.0, gt=0)
    max_page_count: int = Field(default=10000, ge=1)
    max_chunk_count: int = Field(default=100000, ge=1)
    allowed_mime_types: list[str] = Field(
        default_factory=lambda: [
            "text/plain",
            "text/markdown",
            "text/html",
            "text/csv",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/json",
            "message/rfc822",
        ]
    )


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    model_config = ConfigDict(frozen=True)

    log_level: str = "INFO"
    log_query: bool = False
    log_context: bool = False
    log_answer: bool = False
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    redact_pii: bool = True

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            msg = f"Invalid log level: {v}. Must be one of: {valid_levels}"
            raise ValueError(msg)
        return v_upper


class CacheConfig(BaseModel):
    """Cache configuration."""

    model_config = ConfigDict(frozen=True)

    provider: str = "memory"
    ttl_seconds: int = Field(default=3600, ge=0)
    max_size: int = Field(default=10000, ge=0)
    url: str = ""


class IngestionConfig(BaseModel):
    """Ingestion pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    max_concurrent_jobs: int = Field(default=4, ge=1)
    checkpoint_interval: int = Field(default=100, ge=1)
    retry_max_attempts: int = Field(default=3, ge=0)
    retry_delay_seconds: float = Field(default=1.0, ge=0)


class ChunkingConfig(BaseModel):
    """Chunking configuration."""

    model_config = ConfigDict(frozen=True)

    strategy: str = "recursive"  # fixed, recursive, semantic, parent_child
    chunk_size: int = Field(default=512, ge=50, le=32768)
    chunk_overlap: int = Field(default=50, ge=0)
    token_based: bool = False

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = {"fixed", "recursive", "semantic", "parent_child"}
        if v not in valid:
            msg = f"Invalid chunking strategy: {v}. Must be one of: {valid}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_overlap_less_than_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            msg = (
                f"chunk_overlap ({self.chunk_overlap}) must be "
                f"less than chunk_size ({self.chunk_size})"
            )
            raise ValueError(msg)
        return self


class ServerConfig(BaseModel):
    """Server configuration."""

    model_config = ConfigDict(frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    cors_origins: list[str] = Field(default_factory=list)
    request_size_limit_mb: float = Field(default=50.0, gt=0)
    request_timeout_seconds: float = Field(default=300.0, gt=0)


# --- Main Config ---


class RagZenConfig(BaseModel):
    """Root configuration for RagZen.

    Supports loading from:
    - Python code (direct instantiation)
    - YAML file
    - Environment variable overrides

    Environment variables use RAGZEN_ prefix:
    - RAGZEN_ENVIRONMENT
    - RAGZEN_LLM_API_KEY
    - RAGZEN_LLM_BASE_URL
    - RAGZEN_LLM_MODEL
    - RAGZEN_QDRANT_URL
    """

    model_config = ConfigDict(frozen=True)

    environment: str = "development"
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    sparse_index: SparseIndexConfig = Field(default_factory=SparseIndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RagZenConfig:
        """Load configuration from a YAML file with environment variable overrides.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated RagZenConfig.

        Raises:
            ConfigurationError: If the file cannot be read or parsed.
        """
        config_path = Path(path)
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise ConfigurationError(msg)

        try:
            raw_text = config_path.read_text(encoding="utf-8")
            raw_data = yaml.safe_load(raw_text) or {}
        except yaml.YAMLError as e:
            msg = f"Invalid YAML in configuration file: {config_path}: {e}"
            raise ConfigurationError(msg) from e
        except OSError as e:
            msg = f"Cannot read configuration file: {config_path}: {e}"
            raise ConfigurationError(msg) from e

        if not isinstance(raw_data, dict):
            msg = f"Configuration file must contain a YAML mapping, got: {type(raw_data).__name__}"
            raise ConfigurationError(msg)

        # Apply environment variable overrides
        raw_data = _apply_env_overrides(raw_data)

        try:
            return cls.model_validate(raw_data)
        except Exception as e:
            msg = f"Configuration validation failed: {e}"
            raise ConfigurationError(msg) from e

    @classmethod
    def local_default(cls, storage_path: str = ".ragzen") -> RagZenConfig:
        """Create a default configuration for local development.

        Args:
            storage_path: Base path for local storage.

        Returns:
            RagZenConfig with sensible defaults for local use.
        """
        return cls(
            environment="development",
            storage=StorageConfig(path=f"{storage_path}/documents.db"),
            vector_store=VectorStoreConfig(provider="memory"),
            sparse_index=SparseIndexConfig(path=f"{storage_path}/bm25"),
            security=SecurityConfig(
                require_security_context=False,
                fail_closed=False,
            ),
            observability=ObservabilityConfig(log_level="DEBUG"),
        )


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to config data.

    Supports the following env vars:
    - RAGZEN_ENVIRONMENT -> environment
    - RAGZEN_LLM_API_KEY -> llm.api_key
    - RAGZEN_LLM_BASE_URL -> llm.base_url
    - RAGZEN_LLM_MODEL -> llm.model
    - RAGZEN_QDRANT_URL -> vector_store.url
    - RAGZEN_LOG_LEVEL -> observability.log_level
    """
    env_mappings: list[tuple[str, list[str]]] = [
        ("RAGZEN_ENVIRONMENT", ["environment"]),
        ("RAGZEN_LLM_API_KEY", ["llm", "api_key"]),
        ("RAGZEN_LLM_BASE_URL", ["llm", "base_url"]),
        ("RAGZEN_LLM_MODEL", ["llm", "model"]),
        ("RAGZEN_QDRANT_URL", ["vector_store", "url"]),
        ("RAGZEN_LOG_LEVEL", ["observability", "log_level"]),
        ("RAGZEN_STORAGE_PATH", ["storage", "path"]),
    ]

    for env_key, path in env_mappings:
        value = os.environ.get(env_key)
        if value is not None:
            _set_nested(data, path, value)
            logger.debug("Applied env override: %s", env_key)

    return data


def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
    """Set a value in a nested dict by path."""
    for key in path[:-1]:
        if key not in data:
            data[key] = {}
        data = data[key]
    data[path[-1]] = value


def validate_config(config: RagZenConfig) -> list[str]:
    """Validate configuration and return any warnings.

    This performs additional semantic validation beyond Pydantic field validators.

    Returns:
        List of warning messages (empty if no issues).
    """
    warnings: list[str] = []

    # Production environment checks
    if config.environment == "production":
        if not config.security.require_security_context:
            warnings.append(
                "SECURITY: require_security_context is disabled in production. "
                "All queries will proceed without permission checks."
            )
        if not config.security.fail_closed:
            warnings.append(
                "SECURITY: fail_closed is disabled in production. "
                "Permission check failures will allow access."
            )
        if config.observability.log_query:
            warnings.append(
                "PRIVACY: log_query is enabled in production. "
                "User queries will be written to logs."
            )
        if config.observability.log_answer:
            warnings.append(
                "PRIVACY: log_answer is enabled in production. "
                "Generated answers will be written to logs."
            )
        if config.vector_store.provider == "memory":
            warnings.append(
                "RELIABILITY: Using in-memory vector store in production. "
                "Data will be lost on restart."
            )
        cors_origins = config.server.cors_origins
        if cors_origins and "*" in cors_origins:
            warnings.append(
                "SECURITY: CORS is set to allow all origins (*) in production."
            )

    # Chunking sanity checks
    if config.chunking.chunk_overlap >= config.chunking.chunk_size // 2:
        warnings.append(
            f"PERFORMANCE: chunk_overlap ({config.chunking.chunk_overlap}) is >= 50% "
            f"of chunk_size ({config.chunking.chunk_size}). This may cause excessive overlap."
        )

    return warnings
