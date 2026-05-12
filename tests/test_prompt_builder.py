from src.prompt_builder import render_prompt
from src.schema import InputSample, LabelDefinition, LabelSchema


def test_render_prompt_includes_guideline_labels_text_and_schema() -> None:
    template = (
        "{{prompt_version}}\n"
        "{{annotation_guideline}}\n"
        "{{label_definitions}}\n"
        "{{sample_id}}\n"
        "{{input_text}}\n"
        "{{json_output_schema}}"
    )
    schema = LabelSchema(
        prompt_version="weak_labeling_v1",
        annotation_guideline="Use only evidence in the text.",
        labels=[
            LabelDefinition(name="FRAUD", definition="Fraud evidence is present."),
            LabelDefinition(name="NOT_FRAUD", definition="Fraud evidence is absent."),
        ],
        output_fields={},
    )
    sample = InputSample(sample_id="s001", text="Unauthorized transfer reported.")

    prompt = render_prompt(template, schema, sample, "weak_labeling_v1")

    assert "weak_labeling_v1" in prompt
    assert "Use only evidence in the text." in prompt
    assert "- FRAUD: Fraud evidence is present." in prompt
    assert "s001" in prompt
    assert "Unauthorized transfer reported." in prompt
    assert "confidence" in prompt
