from __future__ import annotations

from abc import ABC, abstractmethod

from src.schema import DecodingConfig, GenerationRequest, LabelSchema, ModelConfig


class LLMBackend(ABC):
    """Abstract generation backend.

    Implementations should be local-only for this project unless explicitly extended.
    """

    backend_name: str = "base"

    def __init__(
        self,
        model_config: ModelConfig,
        decoding_config: DecodingConfig,
        label_schema: LabelSchema,
    ) -> None:
        self.model_config = model_config
        self.decoding_config = decoding_config
        self.label_schema = label_schema

    @abstractmethod
    def generate(self, request: GenerationRequest) -> str:
        """Return raw model text for a single prompt."""


def create_backend(
    backend_name: str,
    model_config: ModelConfig,
    decoding_config: DecodingConfig,
    label_schema: LabelSchema,
) -> LLMBackend:
    normalized = backend_name.lower().strip()
    if normalized == "mock":
        from src.backends.mock_backend import MockLLMBackend

        return MockLLMBackend(model_config, decoding_config, label_schema)
    if normalized == "transformers":
        from src.backends.transformers_backend import TransformersBackend

        return TransformersBackend(model_config, decoding_config, label_schema)
    if normalized == "ollama":
        from src.backends.ollama_backend import OllamaBackend

        return OllamaBackend(model_config, decoding_config, label_schema)
    raise ValueError(f"Unknown backend '{backend_name}'. Expected one of: mock, transformers, ollama.")
