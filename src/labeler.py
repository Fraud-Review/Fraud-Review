from __future__ import annotations

from src.backends.base import LLMBackend
from src.json_parser import parse_model_response
from src.prompt_builder import build_retry_prompt, render_prompt
from src.schema import GenerationRequest, InputSample, LabelSchema, ParsedPrediction


class WeakLabeler:
    def __init__(self, backend: LLMBackend, run_id: str, parse_retries: int = 2) -> None:
        if parse_retries < 0:
            raise ValueError("parse_retries must be greater than or equal to 0.")
        self.backend = backend
        self.run_id = run_id
        self.parse_retries = parse_retries

    def label_sample(
        self,
        sample: InputSample,
        prompt_template: str,
        label_schema: LabelSchema,
        prompt_version: str,
    ) -> ParsedPrediction:
        original_prompt = render_prompt(prompt_template, label_schema, sample, prompt_version)
        prompt = original_prompt
        last_prediction: ParsedPrediction | None = None

        for attempt in range(self.parse_retries + 1):
            request = GenerationRequest(
                prompt=prompt,
                sample=sample,
                model_id=self.backend.model_config.model_id,
                prompt_version=prompt_version,
                allowed_labels=label_schema.allowed_labels,
                attempt=attempt,
            )
            raw_response = self.backend.generate(request)
            prediction = parse_model_response(
                raw_response=raw_response,
                run_id=self.run_id,
                sample_id=sample.sample_id,
                model_id=self.backend.model_config.model_id,
                model_family=self.backend.model_config.model_family,
                parameter_size=self.backend.model_config.parameter_size,
                quantization=self.backend.model_config.quantization,
                backend=self.backend.backend_name,
                prompt_version=prompt_version,
                allowed_labels=label_schema.allowed_labels,
            )
            last_prediction = prediction
            if prediction.parse_success:
                return prediction
            prompt = build_retry_prompt(original_prompt, raw_response, prediction.error_message)

        if last_prediction is None:
            raise RuntimeError("Labeling failed before any backend response was parsed.")
        return last_prediction

    def label_samples(
        self,
        samples: list[InputSample],
        prompt_template: str,
        label_schema: LabelSchema,
        prompt_version: str,
    ) -> list[ParsedPrediction]:
        return [
            self.label_sample(
                sample=sample,
                prompt_template=prompt_template,
                label_schema=label_schema,
                prompt_version=prompt_version,
            )
            for sample in samples
        ]
