"""Mock LLM provider for testing and offline fallback."""

from __future__ import annotations


class MockLLMProvider:
    """Mock LLM provider that generates responses based on provided context."""

    def __init__(self, model_name: str = "mock-qwen2.5") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        if "Quy trình xử lý" in prompt or "xử lý sản phẩm lỗi" in prompt:
            return (
                "Quy trình xử lý sản phẩm lỗi bao gồm phân loại, "
                "ghi nhận biên bản và chuyển bộ phận tái chế. [Source 1]"
            )
        if "báo cáo tài chính" in prompt:
            return "Báo cáo tài chính ghi nhận doanh thu tăng trưởng 15%. [Source 1]"
        return "Dựa trên tài liệu được cung cấp, quy trình đã được quy định chi tiết. [Source 1]"

    def health_check(self) -> bool:
        return True
