from src.json_parser import parse_model_response


def test_parse_model_response_extracts_json_from_text() -> None:
    raw = 'Here is the result:\n{"label": "FRAUD", "confidence": 0.82, "rationale": "Unauthorized transfer."}'

    parsed = parse_model_response(
        raw_response=raw,
        run_id="test-run",
        sample_id="s001",
        model_id="mock/model",
        model_family="mock",
        parameter_size="0B",
        quantization="none",
        backend="mock",
        prompt_version="weak_labeling_v1",
        allowed_labels=["FRAUD", "NOT_FRAUD"],
    )

    assert parsed.parse_success is True
    assert parsed.run_id == "test-run"
    assert parsed.sample_id == "s001"
    assert parsed.label == "FRAUD"
    assert parsed.confidence == 0.82
    assert parsed.error_message == ""


def test_parse_model_response_rejects_invalid_label() -> None:
    raw = '{"label": "MAYBE", "confidence": 0.5, "rationale": "Ambiguous."}'

    parsed = parse_model_response(
        raw_response=raw,
        run_id="test-run",
        sample_id="s002",
        model_id="mock/model",
        model_family="mock",
        parameter_size="0B",
        quantization="none",
        backend="mock",
        prompt_version="weak_labeling_v1",
        allowed_labels=["FRAUD", "NOT_FRAUD"],
    )

    assert parsed.parse_success is False
    assert parsed.label == ""
    assert "Invalid label" in parsed.error_message
