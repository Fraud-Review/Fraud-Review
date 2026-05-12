from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.schema import DecodingConfig, LabelDefinition, LabelSchema, ModelConfig, PipelineConfig


def _load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"YAML file not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {source}")
    return data


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    data = _load_yaml(path)

    prompt = data.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("Config must contain a 'prompt' mapping.")

    runtime = data.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("Config field 'runtime' must be a mapping when provided.")

    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("Config must contain a non-empty 'models' list.")

    models: list[ModelConfig] = []
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise ValueError(f"Model entry at index {index} must be a mapping.")
        model_id = raw_model.get("model_id")
        if not model_id:
            raise ValueError(f"Model entry at index {index} is missing 'model_id'.")
        models.append(
            ModelConfig(
                model_id=str(model_id),
                model_family=str(raw_model.get("model_family", "unknown")),
                parameter_size=str(raw_model.get("parameter_size", "unknown")),
                quantization=str(raw_model.get("quantization", "none")),
                enabled=bool(raw_model.get("enabled", True)),
                recommended_backend=str(raw_model.get("recommended_backend", "")),
                notes=str(raw_model.get("notes", "")),
                backend_options=dict(raw_model.get("backend_options", {})),
            )
        )

    decoding = data.get("decoding", {})
    if not isinstance(decoding, dict):
        raise ValueError("Config field 'decoding' must be a mapping when provided.")

    prompt_version = prompt.get("version")
    template_path = prompt.get("template_path")
    if not prompt_version:
        raise ValueError("Config prompt.version is required.")
    if not template_path:
        raise ValueError("Config prompt.template_path is required.")

    return PipelineConfig(
        prompt_version=str(prompt_version),
        prompt_template_path=Path(template_path),
        decoding=DecodingConfig.from_dict(decoding),
        parse_retries=int(runtime.get("parse_retries", 2)),
        output_merge_filename=Path(runtime.get("output_merge_filename", "merged_predictions.csv")),
        summary_report_filename=Path(runtime.get("summary_report_filename", "summary_report.csv")),
        models=models,
    )


def load_label_schema(path: str | Path) -> LabelSchema:
    data = _load_yaml(path)

    raw_labels = data.get("labels")
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ValueError("Label schema must contain a non-empty 'labels' list.")

    labels: list[LabelDefinition] = []
    for index, raw_label in enumerate(raw_labels):
        if not isinstance(raw_label, dict):
            raise ValueError(f"Label entry at index {index} must be a mapping.")
        name = raw_label.get("name")
        definition = raw_label.get("definition")
        if not name or not definition:
            raise ValueError(f"Label entry at index {index} requires 'name' and 'definition'.")
        labels.append(LabelDefinition(name=str(name), definition=str(definition)))

    output_fields = data.get("output_fields", {})
    if not isinstance(output_fields, dict):
        raise ValueError("Label schema field 'output_fields' must be a mapping when provided.")

    prompt_version = data.get("prompt_version", "")
    annotation_guideline = data.get("annotation_guideline")
    if not annotation_guideline:
        raise ValueError("Label schema must contain 'annotation_guideline'.")

    return LabelSchema(
        prompt_version=str(prompt_version),
        annotation_guideline=str(annotation_guideline).strip(),
        labels=labels,
        output_fields={str(key): str(value) for key, value in output_fields.items()},
    )
