from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    model_family: str
    parameter_size: str
    quantization: str = "none"
    enabled: bool = True
    recommended_backend: str = ""
    notes: str = ""
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecodingConfig":
        known = {
            "temperature": float(data.get("temperature", 0.0)),
            "top_p": float(data.get("top_p", 1.0)),
            "max_new_tokens": int(data.get("max_new_tokens", 512)),
            "seed": int(data.get("seed", 42)),
        }
        extra = {key: value for key, value in data.items() if key not in known}
        return cls(**known, extra=extra)


@dataclass(frozen=True)
class PipelineConfig:
    prompt_version: str
    prompt_template_path: Path
    decoding: DecodingConfig
    parse_retries: int
    output_merge_filename: Path
    summary_report_filename: Path
    models: list[ModelConfig]


@dataclass(frozen=True)
class LabelDefinition:
    name: str
    definition: str


@dataclass(frozen=True)
class LabelSchema:
    prompt_version: str
    annotation_guideline: str
    labels: list[LabelDefinition]
    output_fields: dict[str, str]

    @property
    def allowed_labels(self) -> list[str]:
        return [label.name for label in self.labels]


@dataclass(frozen=True)
class InputSample:
    sample_id: str
    text: str


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    sample: InputSample
    model_id: str
    prompt_version: str
    allowed_labels: list[str]
    attempt: int = 0


@dataclass
class ParsedPrediction:
    run_id: str
    sample_id: str
    model_id: str
    model_family: str
    parameter_size: str
    quantization: str
    backend: str
    prompt_version: str
    label: str
    confidence: float | None
    rationale: str
    raw_response: str
    parse_success: bool
    error_message: str
    created_at: str

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
            "backend": self.backend,
            "prompt_version": self.prompt_version,
            "label": self.label,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "raw_response": self.raw_response,
            "parse_success": self.parse_success,
            "error_message": self.error_message,
            "created_at": self.created_at,
        }
