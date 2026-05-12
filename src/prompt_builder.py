from __future__ import annotations

import json
from pathlib import Path

from src.schema import InputSample, LabelSchema
from src.utils import read_text


def load_prompt_template(path: str | Path) -> str:
    template = read_text(path)
    required_placeholders = {
        "{{prompt_version}}",
        "{{annotation_guideline}}",
        "{{label_definitions}}",
        "{{sample_id}}",
        "{{input_text}}",
        "{{json_output_schema}}",
    }
    missing = [placeholder for placeholder in required_placeholders if placeholder not in template]
    if missing:
        raise ValueError(f"Prompt template is missing placeholder(s): {', '.join(sorted(missing))}")
    return template


def render_label_definitions(label_schema: LabelSchema) -> str:
    return "\n".join(f"- {label.name}: {label.definition}" for label in label_schema.labels)


def render_json_output_schema(label_schema: LabelSchema) -> str:
    schema = {
        "label": f"one of: {', '.join(label_schema.allowed_labels)}",
        "confidence": "number between 0.0 and 1.0",
        "rationale": "concise string grounded in the input text",
    }
    for field_name, description in label_schema.output_fields.items():
        if field_name in schema:
            schema[field_name] = description
    return json.dumps(schema, indent=2, ensure_ascii=False)


def render_prompt(
    template: str,
    label_schema: LabelSchema,
    sample: InputSample,
    prompt_version: str,
) -> str:
    replacements = {
        "{{prompt_version}}": prompt_version,
        "{{annotation_guideline}}": label_schema.annotation_guideline,
        "{{label_definitions}}": render_label_definitions(label_schema),
        "{{sample_id}}": sample.sample_id,
        "{{input_text}}": sample.text,
        "{{json_output_schema}}": render_json_output_schema(label_schema),
    }

    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def build_retry_prompt(original_prompt: str, raw_response: str, error_message: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "Your previous response could not be parsed as the required JSON object.\n"
        f"Parse error: {error_message}\n\n"
        "Previous response:\n"
        f"{raw_response}\n\n"
        "Return only a corrected JSON object with the required fields."
    )
