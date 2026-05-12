from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.backends.base import create_backend
from src.config_loader import load_label_schema, load_pipeline_config
from src.data_loader import load_input_csv, write_merged_csv, write_model_jsonl, write_summary_report_csv
from src.labeler import WeakLabeler
from src.prompt_builder import load_prompt_template, render_prompt
from src.schema import InputSample, LabelSchema, ModelConfig
from src.utils import ensure_directory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weak labels with local open-weight LLM backends.")
    parser.add_argument("--config", required=True, type=Path, help="Path to model/pipeline YAML config.")
    parser.add_argument("--schema", required=True, type=Path, help="Path to label schema YAML.")
    parser.add_argument("--input", required=True, type=Path, help="Path to unlabeled input CSV.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for JSONL and merged CSV outputs.")
    parser.add_argument(
        "--backend",
        default="mock",
        choices=("mock", "transformers", "ollama"),
        help="Inference backend to use.",
    )
    parser.add_argument("--models", nargs="*", help="Optional list of model_id values to run.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact prompt for the first 3 samples and exit without loading a backend.",
    )
    return parser.parse_args(argv)


def select_models(configured_models: list[ModelConfig], requested_models: list[str] | None) -> list[ModelConfig]:
    enabled_models = [model for model in configured_models if model.enabled]
    if not requested_models:
        return enabled_models

    by_id = {model.model_id: model for model in enabled_models}
    missing = [model_id for model_id in requested_models if model_id not in by_id]
    if missing:
        available = ", ".join(sorted(by_id))
        raise ValueError(
            f"Requested model(s) not found or disabled: {', '.join(missing)}. "
            f"Available enabled models: {available}"
        )
    return [by_id[model_id] for model_id in requested_models]


def print_dry_run_prompts(
    prompt_template: str,
    label_schema: LabelSchema,
    samples: list[InputSample],
    prompt_version: str,
) -> None:
    for index, sample in enumerate(samples[:3], start=1):
        prompt = render_prompt(prompt_template, label_schema, sample, prompt_version)
        print(f"\n===== DRY RUN PROMPT {index}: sample_id={sample.sample_id} =====")
        print(prompt)


def build_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    pipeline_config = load_pipeline_config(args.config)
    label_schema = load_label_schema(args.schema)
    if label_schema.prompt_version and label_schema.prompt_version != pipeline_config.prompt_version:
        raise ValueError(
            "Prompt version mismatch: "
            f"config has '{pipeline_config.prompt_version}', schema has '{label_schema.prompt_version}'."
        )

    prompt_template = load_prompt_template(pipeline_config.prompt_template_path)
    samples = load_input_csv(args.input)
    models = select_models(pipeline_config.models, args.models)

    if not models:
        raise ValueError("No enabled models selected.")

    if args.dry_run:
        print_dry_run_prompts(prompt_template, label_schema, samples, pipeline_config.prompt_version)
        return 0

    output_dir = ensure_directory(args.output_dir)
    run_id = build_run_id()
    print(f"run_id: {run_id}")
    all_predictions = []

    for model_config in models:
        print(f"Running model '{model_config.model_id}' with backend '{args.backend}'...")
        backend = create_backend(args.backend, model_config, pipeline_config.decoding, label_schema)
        labeler = WeakLabeler(backend=backend, run_id=run_id, parse_retries=pipeline_config.parse_retries)
        predictions = labeler.label_samples(
            samples=samples,
            prompt_template=prompt_template,
            label_schema=label_schema,
            prompt_version=pipeline_config.prompt_version,
        )
        model_output_path = write_model_jsonl(output_dir, model_config.model_id, predictions)
        print(f"Wrote {len(predictions)} predictions to {model_output_path}")
        all_predictions.extend(predictions)

    merged_output_path = write_merged_csv(
        output_dir=output_dir,
        filename=pipeline_config.output_merge_filename,
        predictions=all_predictions,
    )
    print(f"Wrote merged predictions to {merged_output_path}")
    summary_output_path = write_summary_report_csv(
        output_dir=output_dir,
        filename=pipeline_config.summary_report_filename,
        predictions=all_predictions,
    )
    print(f"Wrote summary report to {summary_output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
