"""Extra unit tests to ensure high test coverage (>85%)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragzen import RagZen
from ragzen.engine import RAGGenerator
from ragzen.exceptions import SecurityContextRequiredError
from ragzen.generation.generator import build_context, validate_citations
from ragzen.models import SearchResult


class TestEngineExtra:
    def test_engine_require_context_raises(self, tmp_path: Path) -> None:
        rag = RagZen.local(storage_path=str(tmp_path))
        sec_cfg = rag.config.security.model_copy(update={"require_security_context": True})
        rag.config = rag.config.model_copy(update={"security": sec_cfg})

        with pytest.raises(SecurityContextRequiredError):
            rag.search("test", security_context=None)

    def test_engine_update_document(self, tmp_path: Path) -> None:
        rag = RagZen.local(storage_path=str(tmp_path))
        doc = rag.add_text("Original content", metadata={"tenant_id": "company-a"})
        success = rag.update(
            doc.document_id, "Updated content", metadata={"tenant_id": "company-a"}
        )
        assert success is True

        # Non-existent document update returns False
        assert rag.update("non-existent-id", "Content") is False

    def test_citation_validation(self) -> None:
        results = [
            SearchResult(
                chunk_id="c1",
                document_id="d1",
                content="Sample content 1",
                score=0.9,
                file_name="file1.txt",
                page=1,
            )
        ]

        # Valid citation
        cites, warnings = validate_citations("Answer according to [Source 1].", results)
        assert len(cites) == 1
        assert cites[0].valid is True
        assert len(warnings) == 0

        # Invalid citation index out of bounds
        cites, warnings = validate_citations("Answer referencing [Source 99].", results)
        assert len(cites) == 1
        assert cites[0].valid is False
        assert len(warnings) == 1

    def test_build_context_max_chars(self) -> None:
        results = [
            SearchResult(chunk_id=f"c{i}", document_id=f"d{i}", content="Content " * 50, score=0.8)
            for i in range(10)
        ]

        context, used = build_context(results, max_chars=300)
        assert len(used) < 10
        assert len(context) <= 400

    def test_generator_no_results(self) -> None:
        from ragzen.llms.mock import MockLLMProvider

        gen = RAGGenerator(llm=MockLLMProvider())
        resp = gen.generate("What is X?", [])
        assert "không có đủ thông tin" in resp.answer.lower()
