from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.schema import InputSample, ParsedPrediction
from src.utils import ensure_directory, sanitize_model_id


REQUIRED_INPUT_COLUMNS = ("sample_id", "text")


def load_input_csv(path: str | Path) -> list[InputSample]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input CSV not found: {source}")

    dataframe = pd.read_csv(source, dtype={"sample_id": str, "text": str}, encoding="utf-8-sig")
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {', '.join(missing)}")

    samples: list[InputSample] = []
    for row_number, row in dataframe.iterrows():
        sample_id = row["sample_id"]
        text = row["text"]
        if pd.isna(sample_id) or str(sample_id).strip() == "":
            raise ValueError(f"Input CSV row {row_number + 2} has an empty sample_id.")
        if pd.isna(text) or str(text).strip() == "":
            raise ValueError(f"Input CSV row {row_number + 2} has an empty text value.")
        samples.append(InputSample(sample_id=str(sample_id), text=str(text)))

    return samples


def write_model_jsonl(output_dir: str | Path, model_id: str, predictions: list[ParsedPrediction]) -> Path:
    directory = ensure_directory(output_dir)
    output_path = directory / f"{sanitize_model_id(model_id)}.jsonl"
    ensure_directory(output_path.parent)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_record(), ensure_ascii=False) + "\n")
    return output_path


def write_merged_csv(output_dir: str | Path, filename: str | Path, predictions: list[ParsedPrediction]) -> Path:
    directory = ensure_directory(output_dir)
    output_path = directory / Path(filename)
    ensure_directory(output_path.parent)
    records = [prediction.to_record() for prediction in predictions]
    pd.DataFrame.from_records(records).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def write_summary_report_csv(output_dir: str | Path, filename: str | Path, predictions: list[ParsedPrediction]) -> Path:
    directory = ensure_directory(output_dir)
    output_path = directory / Path(filename)
    ensure_directory(output_path.parent)
    rows = []

    model_ids = sorted({prediction.model_id for prediction in predictions})
    for model_id in model_ids:
        model_predictions = [prediction for prediction in predictions if prediction.model_id == model_id]
        success_predictions = [prediction for prediction in model_predictions if prediction.parse_success]
        confidence_values = [
            prediction.confidence for prediction in success_predictions if prediction.confidence is not None
        ]
        total_samples = len(model_predictions)
        parse_success_count = len(success_predictions)
        parse_failure_count = total_samples - parse_success_count
        rows.append(
            {
                "model_id": model_id,
                "total_samples": total_samples,
                "parse_success_count": parse_success_count,
                "parse_failure_count": parse_failure_count,
                "parse_success_rate": parse_success_count / total_samples if total_samples else 0.0,
                "average_confidence": (
                    sum(confidence_values) / len(confidence_values) if confidence_values else None
                ),
            }
        )

    pd.DataFrame.from_records(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
