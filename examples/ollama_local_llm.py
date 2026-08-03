"""Ollama local LLM integration example for RagZen.

Shows how to run 100% offline RAG using Ollama (Llama 3 / Mistral) with OpenAI-compatible API.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ragzen import RagZen, SecurityContext
from ragzen.llms.openai_compatible import OpenAICompatibleLLM


def main() -> None:
    print("=== RagZen + Ollama Local Offline RAG Demo ===")

    # Initialize OpenAICompatibleLLM pointing to local Ollama server
    ollama_provider = OpenAICompatibleLLM(
        base_url="http://localhost:11434/v1",
        model="llama3:latest",
        api_key="ollama",
        timeout=30.0,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir) / ".ragzen"

        # Initialize RagZen with custom Ollama LLM provider
        rag = RagZen.from_components(
            storage_path=str(storage_path),
            llm=ollama_provider,
        )

        # Ingest local document
        rag.add_text(
            "RagZen is designed for local-first enterprise RAG with zero SaaS lock-in.",
            metadata={"tenant_id": "local_corp", "author": "engineering"},
        )

        sec_ctx = SecurityContext(tenant_id="local_corp", user_id="local_user")

        print("Checking Ollama reachability...")
        if ollama_provider.health_check():
            print("Ollama is online! Querying Llama 3 model...")
            response = rag.ask("Why use RagZen for local RAG?", security_context=sec_ctx)
            print("Response:", response.answer)
        else:
            print("Ollama server is offline. RagZen fallback mechanism ready.")

        rag.close()


if __name__ == "__main__":
    main()
