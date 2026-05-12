from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from src.schema import ParsedPrediction
from src.utils import utc_now_iso


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    decoder = json.JSONDecoder()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Top-level JSON value must be an object.")
    except JSONDecodeError:
        pass

    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("No valid JSON object found in model response.")


def parse_model_response(
    raw_response: str,
    run_id: str,
    sample_id: str,
    model_id: str,
    model_family: str,
    parameter_size: str,
    quantization: str,
    backend: str,
    prompt_version: str,
    allowed_labels: list[str],
) -> ParsedPrediction:
    try:
        parsed = _extract_json_object(raw_response)

        missing = [field for field in ("label", "confidence", "rationale") if field not in parsed]
        if missing:
            raise ValueError(f"JSON object is missing required field(s): {', '.join(missing)}")

        label = str(parsed["label"])
        if label not in allowed_labels:
            raise ValueError(f"Invalid label '{label}'. Expected one of: {', '.join(allowed_labels)}")

        try:
            confidence = float(parsed["confidence"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Field 'confidence' must be numeric.") from exc

        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("Field 'confidence' must be between 0.0 and 1.0.")

        rationale = str(parsed["rationale"]).strip()
        if not rationale:
            raise ValueError("Field 'rationale' must be a non-empty string.")

        return ParsedPrediction(
            run_id=run_id,
            sample_id=sample_id,
            model_id=model_id,
            model_family=model_family,
            parameter_size=parameter_size,
            quantization=quantization,
            backend=backend,
            prompt_version=prompt_version,
            label=label,
            confidence=confidence,
            rationale=rationale,
            raw_response=raw_response,
            parse_success=True,
            error_message="",
            created_at=utc_now_iso(),
        )
    except Exception as exc:
        return ParsedPrediction(
            run_id=run_id,
            sample_id=sample_id,
            model_id=model_id,
            model_family=model_family,
            parameter_size=parameter_size,
            quantization=quantization,
            backend=backend,
            prompt_version=prompt_version,
            label="",
            confidence=None,
            rationale="",
            raw_response=raw_response,
            parse_success=False,
            error_message=str(exc),
            created_at=utc_now_iso(),
        )
