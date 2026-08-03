import sys
import tempfile
from pathlib import Path

from ragzen import RagZen


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Initialize RagZen local instance
        storage_path = Path(tmp_dir) / ".ragzen"
        rag = RagZen.local(storage_path=str(storage_path))

        try:
            # Create sample document
            doc_dir = Path(tmp_dir) / "documents"
            doc_dir.mkdir(parents=True, exist_ok=True)
            doc_file = doc_dir / "quy_trinh.txt"
            doc_file.write_text(
                "Quy trình xử lý sản phẩm lỗi bao gồm phân loại, ghi nhận biên bản và chuyển bộ phận tái chế.",
                encoding="utf-8",
            )

            # 2. Add document with multi-tenant metadata
            rag.add(
                doc_file,
                metadata={
                    "tenant_id": "company-a",
                    "department": "production",
                    "access_level": "internal",
                },
            )

            # 3. Ask question with SecurityContext
            response = rag.ask(
                "Quy trình xử lý sản phẩm lỗi là gì?",
                security_context={
                    "tenant_id": "company-a",
                    "user_id": "user-123",
                    "roles": ["production_manager"],
                    "departments": ["production"],
                },
            )

            print("\nAnswer:")
            print(response.answer)

            print("\nSources:")
            for source in response.sources:
                print(f"File: {source.file_name}, Score: {source.score:.4f}")

        finally:
            rag.close()


if __name__ == "__main__":
    main()
