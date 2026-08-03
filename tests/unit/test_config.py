"""Tests for configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragzen.config import (
    ChunkingConfig,
    ObservabilityConfig,
    RagZenConfig,
    RetrievalConfig,
    SecurityConfig,
    validate_config,
)
from ragzen.exceptions import ConfigurationError


class TestRagZenConfig:
    """Tests for main config."""

    def test_default_config(self) -> None:
        config = RagZenConfig()
        assert config.environment == "development"
        assert config.storage.provider == "sqlite"
        assert config.embedding.provider == "sentence_transformers"

    def test_local_default(self, tmp_path: Path) -> None:
        config = RagZenConfig.local_default(str(tmp_path / "ragzen"))
        assert config.vector_store.provider == "memory"
        assert config.security.require_security_context is False
        assert config.observability.log_level == "DEBUG"

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
environment: production
storage:
  provider: sqlite
  path: /data/docs.db
embedding:
  model: BAAI/bge-m3
  batch_size: 64
retrieval:
  mode: hybrid
  top_k_dense: 50
security:
  require_security_context: true
  fail_closed: true
"""
        config_file = tmp_path / "ragzen.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = RagZenConfig.from_yaml(config_file)
        assert config.environment == "production"
        assert config.storage.path == "/data/docs.db"
        assert config.embedding.model == "BAAI/bge-m3"
        assert config.embedding.batch_size == 64
        assert config.retrieval.mode == "hybrid"
        assert config.retrieval.top_k_dense == 50

    def test_from_yaml_not_found(self) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            RagZenConfig.from_yaml("/nonexistent/config.yaml")

    def test_from_yaml_invalid(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid yaml", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            RagZenConfig.from_yaml(config_file)

    def test_from_yaml_non_mapping(self, tmp_path: Path) -> None:
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="YAML mapping"):
            RagZenConfig.from_yaml(config_file)

    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        yaml_content = """
environment: development
llm:
  model: default-model
"""
        config_file = tmp_path / "ragzen.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        monkeypatch.setenv("RAGZEN_ENVIRONMENT", "production")
        monkeypatch.setenv("RAGZEN_LLM_MODEL", "custom-model")

        config = RagZenConfig.from_yaml(config_file)
        assert config.environment == "production"
        assert config.llm.model == "custom-model"

    def test_config_is_frozen(self) -> None:
        from pydantic import ValidationError

        config = RagZenConfig()
        with pytest.raises((ValidationError, TypeError)):
            config.environment = "modified"  # type: ignore[misc]


class TestRetrievalConfig:
    """Tests for retrieval config validation."""

    def test_valid_modes(self) -> None:
        for mode in ("dense", "sparse", "hybrid"):
            config = RetrievalConfig(mode=mode)
            assert config.mode == mode

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="Invalid retrieval mode"):
            RetrievalConfig(mode="invalid")

    def test_invalid_fusion(self) -> None:
        with pytest.raises(ValueError, match="Invalid fusion"):
            RetrievalConfig(fusion="invalid")


class TestChunkingConfig:
    """Tests for chunking config."""

    def test_overlap_must_be_less_than_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="Invalid chunking"):
            ChunkingConfig(strategy="invalid")


class TestObservabilityConfig:
    """Tests for observability config."""

    def test_valid_log_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            config = ObservabilityConfig(log_level=level)
            assert config.log_level == level

    def test_case_insensitive_log_level(self) -> None:
        config = ObservabilityConfig(log_level="info")
        assert config.log_level == "INFO"

    def test_invalid_log_level(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            ObservabilityConfig(log_level="INVALID")


class TestValidateConfig:
    """Tests for semantic config validation."""

    def test_production_warnings(self) -> None:
        config = RagZenConfig(
            environment="production",
            security=SecurityConfig(require_security_context=False, fail_closed=False),
            observability=ObservabilityConfig(log_query=True, log_answer=True),
            vector_store__provider="memory",
        ) if False else RagZenConfig(
            environment="production",
            security=SecurityConfig(require_security_context=False, fail_closed=False),
            observability=ObservabilityConfig(log_query=True, log_answer=True),
        )
        warnings = validate_config(config)
        assert any("require_security_context" in w for w in warnings)
        assert any("fail_closed" in w for w in warnings)
        assert any("log_query" in w for w in warnings)

    def test_no_warnings_in_dev(self) -> None:
        config = RagZenConfig(environment="development")
        warnings = validate_config(config)
        assert len(warnings) == 0
