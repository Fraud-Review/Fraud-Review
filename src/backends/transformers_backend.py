from __future__ import annotations

from src.backends.base import LLMBackend
from src.schema import GenerationRequest


class TransformersBackend(LLMBackend):
    backend_name = "transformers"

    def generate(self, request: GenerationRequest) -> str:
        raise NotImplementedError(
            "TransformersBackend is a placeholder and intentionally does not load models yet. "
            "Implement local Hugging Face generation here when ready, or use '--backend mock'."
        )
