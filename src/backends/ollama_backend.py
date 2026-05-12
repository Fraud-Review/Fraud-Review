from __future__ import annotations

from src.backends.base import LLMBackend
from src.schema import GenerationRequest


class OllamaBackend(LLMBackend):
    backend_name = "ollama"

    def generate(self, request: GenerationRequest) -> str:
        raise NotImplementedError(
            "OllamaBackend is a placeholder and intentionally does not call Ollama yet. "
            "Implement local Ollama generation here when ready, or use '--backend mock'."
        )
