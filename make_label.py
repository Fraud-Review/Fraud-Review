#!/usr/bin/env python3
"""
Steam review authenticity labeling with Qwen.

This script reads steam_top5.csv in chunks, asks the Hugging Face model to label
each review as:
  - real: likely genuine user review
  - suspicious: weak evidence or mixed signals
  - fake: likely spam, manipulation, bot-like, or non-review content

Outputs:
  - steam_top5_labeled.csv: original rows plus label, confidence, and reason
  - steam_top5_summary.csv: counts and percentages by game and overall
  - fake_review_criteria.md: human-readable criteria used for the labeling

Install example:
  pip install -U pandas tqdm transformers accelerate torch kernels

Run examples:
  python make_label.py --limit 100 --batch-size 4
  python make_label.py --input steam_top5.csv --batch-size 8 --chunk-size 500
  python make_label.py --force-download --limit 1
  python make_label.py --device cpu --limit 1
  python make_label.py --max-gpu-memory 8GiB --offload-folder model_offload --limit 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


MODEL_ID = "Qwen/Qwen3-8B"
LABELS = ("real", "suspicious", "fake")

CRITERIA_MD = """# Fake Review Labeling Criteria

The labels are model judgments, not ground truth. Use them as a screening result
that should be spot-checked before making claims.

## real

- Review contains concrete gameplay experience, mechanics, context, pros/cons, or
  a plausible personal reaction.
- Account/playtime metadata is broadly consistent with the review.
- Short reviews can still be real when metadata is strong and the text looks like
  normal human feedback.

## suspicious

- Review is extremely generic, duplicated-looking, meme-only, or too short to
  judge confidently.
- Metadata has weak or mixed signals, such as very low playtime, one total review,
  free copy, or many similar low-effort reviews.
- Text may be sincere but lacks enough evidence for a confident real/fake label.

## fake

- Review looks like spam, advertising, review farming, copy-paste manipulation,
  bot output, irrelevant text, or coordinated praise/attack.
- Strong mismatch between text and metadata, such as confident claims with almost
  no playtime, or repeated boilerplate from low-history accounts.
- Text is mostly links, scams, commands, unrelated content, or unnatural keyword
  stuffing.

The script passes both review text and metadata to the model. The final percentage
is calculated from the model's JSON labels in the output CSV.
"""


SYSTEM_PROMPT = """You are labeling Steam game reviews for authenticity screening.

Classify each review into exactly one label:
- real: likely genuine human review from a player
- suspicious: uncertain, weak evidence, low-effort, or mixed signals
- fake: likely spam, bot, review manipulation, advertisement, irrelevant content, or coordinated/farmed review

Important rules:
- Do not treat short text as automatically fake. Many Steam reviews are short.
- Use metadata as supporting evidence, not as the only reason.
- Be conservative: choose fake only when evidence is strong.
- Return only valid JSON. No markdown, no extra text.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label Steam reviews as real/suspicious/fake using Qwen."
    )
    parser.add_argument("--input", default="steam_top5.csv", help="Input CSV path.")
    parser.add_argument(
        "--output", default="steam_top5_labeled.csv", help="Labeled output CSV path."
    )
    parser.add_argument(
        "--summary", default="steam_top5_summary.csv", help="Summary output CSV path."
    )
    parser.add_argument(
        "--criteria",
        default="fake_review_criteria.md",
        help="Markdown file explaining labeling criteria.",
    )
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model id.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Rows to read from CSV at a time.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Model inference batch size. Lower this if VRAM is limited.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum generated tokens per review.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to process for testing.",
    )
    parser.add_argument(
        "--limit-per-game",
        type=int,
        default=None,
        help="Optional maximum labeled rows per game, including rows already labeled with --resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in the output CSV by recommendationid.",
    )
    parser.add_argument(
        "--text-max-chars",
        type=int,
        default=1800,
        help="Truncate very long review text to this many characters.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a simple local heuristic instead of loading the model. Good for testing I/O.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force Hugging Face to re-download model files. Use this if cached weights are broken.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="Model placement. Use cpu when your GPU is too small, but expect it to be slow.",
    )
    parser.add_argument(
        "--max-gpu-memory",
        default=None,
        help='Limit GPU memory for automatic device mapping, for example "8GiB".',
    )
    parser.add_argument(
        "--cpu-memory",
        default="64GiB",
        help='CPU RAM available for offloading, for example "64GiB".',
    )
    parser.add_argument(
        "--offload-folder",
        default="model_offload",
        help="Folder used by Transformers when model weights are offloaded to disk.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the model with bitsandbytes 4-bit quantization. Recommended for 11GB GPUs.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load the model with bitsandbytes 8-bit quantization.",
    )
    return parser.parse_args()


def load_model(
    model_id: str,
    force_download: bool = False,
    device: str = "auto",
    max_gpu_memory: str | None = None,
    cpu_memory: str = "64GiB",
    offload_folder: str = "model_offload",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    if load_in_4bit and load_in_8bit:
        raise ValueError("Use only one of --load-in-4bit or --load-in-8bit.")

    from transformers import pipeline

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_map = {"": "cpu"}
    elif load_in_4bit or load_in_8bit:
        # bitsandbytes quantized loading is happiest when the quantized model
        # stays on GPU. If accelerate dispatches layers to CPU/disk, loading can
        # fail before inference starts.
        device_map = {"": 0}
    else:
        device_map = "auto"

    model_kwargs: dict[str, Any] = {"force_download": force_download}
    if load_in_4bit or load_in_8bit:
        import torch
        from transformers import BitsAndBytesConfig

        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    if device == "auto" and max_gpu_memory and not (load_in_4bit or load_in_8bit):
        Path(offload_folder).mkdir(parents=True, exist_ok=True)
        model_kwargs.update(
            {
                "max_memory": {0: max_gpu_memory, "cpu": cpu_memory},
                "offload_folder": offload_folder,
                "offload_state_dict": True,
            }
        )

    try:
        return pipeline(
            "text-generation",
            model=model_id,
            dtype="auto",
            device_map=device_map,
            model_kwargs=model_kwargs,
        )
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
        if (
            "automatic conversion of the weights" in message
            or "CUDA out of memory" in message
            or "OutOfMemoryError" in message
            or "Some modules are dispatched on the CPU or the disk" in message
        ):
            raise RuntimeError(
                f"Failed while Transformers was loading {model_id}.\n\n"
                "Your log shows CUDA out of memory. This model can still be too large "
                "for an 11GB GPU unless you use CPU/offloading or a quantized runtime.\n\n"
                "Try these in order:\n"
                "1. Use 4-bit GPU loading without offload: python make_label.py --load-in-4bit --batch-size 1 --limit 1\n"
                "2. If that is still too large, use a smaller model with: python make_label.py --model Qwen/Qwen3-8B --load-in-4bit --batch-size 1 --limit 1\n"
                "3. Use CPU mode only for tiny tests: python make_label.py --device cpu --limit 1\n"
                "4. For practical speed on smaller GPUs, use an Ollama/GGUF/vLLM server and "
                "call it from the script instead of loading the full Transformers model locally."
            ) from exc
        raise


def safe_text(value: Any, max_chars: int) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\x00", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def minutes_to_hours(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return round(float(value) / 60.0, 2)
    except (TypeError, ValueError):
        return None


def build_messages(row: pd.Series, text_max_chars: int) -> list[dict[str, str]]:
    review = safe_text(row.get("review", ""), text_max_chars)
    metadata = {
        "game_name": row.get("game_name"),
        "language": row.get("language"),
        "voted_up": row.get("voted_up"),
        "review_text": review,
        "num_games_owned": row.get("num_games_owned"),
        "num_reviews_by_user": row.get("num_reviews"),
        "playtime_forever_hours": minutes_to_hours(row.get("playtime_forever")),
        "playtime_at_review_hours": minutes_to_hours(row.get("playtime_at_review")),
        "playtime_last_two_weeks_hours": minutes_to_hours(
            row.get("playtime_last_two_weeks")
        ),
        "votes_up": row.get("votes_up"),
        "votes_funny": row.get("votes_funny"),
        "weighted_vote_score": row.get("weighted_vote_score"),
        "comment_count": row.get("comment_count"),
        "steam_purchase": row.get("steam_purchase"),
        "received_for_free": row.get("received_for_free"),
        "written_during_early_access": row.get("written_during_early_access"),
        "created_date": row.get("created_date"),
        "updated_date": row.get("updated_date"),
    }
    user_prompt = (
        "/no_think\n"
        "Label this single Steam review. Return this exact JSON schema:\n"
        '{"label":"real|suspicious|fake","confidence":0.0,"reason":"short reason"}\n\n'
        f"Review data:\n{json.dumps(metadata, ensure_ascii=False, default=str)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_generated_text(output: Any) -> str:
    generated = output[0]["generated_text"] if isinstance(output, list) else output
    if isinstance(generated, list):
        return str(generated[-1].get("content", ""))
    return str(generated)


def parse_label(raw_text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        return {
            "label": "suspicious",
            "confidence": 0.0,
            "reason": "Model did not return parseable JSON.",
        }

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "label": "suspicious",
            "confidence": 0.0,
            "reason": "Model returned invalid JSON.",
        }

    label = str(data.get("label", "suspicious")).strip().lower()
    if label not in LABELS:
        label = "suspicious"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = safe_text(data.get("reason", ""), 300)
    return {"label": label, "confidence": confidence, "reason": reason}


def heuristic_label(row: pd.Series, text_max_chars: int) -> dict[str, Any]:
    review = safe_text(row.get("review", ""), text_max_chars).lower()
    playtime_hours = minutes_to_hours(row.get("playtime_at_review")) or 0.0
    num_reviews = row.get("num_reviews")
    try:
        num_reviews = int(num_reviews)
    except (TypeError, ValueError):
        num_reviews = 0

    spam_patterns = (
        "http://",
        "https://",
        "discord.gg",
        "free skins",
        "promo code",
        "visit my",
    )
    repeated_chars = bool(re.search(r"(.)\1{8,}", review))
    very_short = len(review.split()) <= 2

    if any(pattern in review for pattern in spam_patterns) or repeated_chars:
        return {
            "label": "fake",
            "confidence": 0.7,
            "reason": "Heuristic test mode: spam-like link/promo/repetition pattern.",
        }
    if very_short or (playtime_hours < 0.5 and num_reviews <= 1):
        return {
            "label": "suspicious",
            "confidence": 0.55,
            "reason": "Heuristic test mode: too little text or weak metadata.",
        }
    return {
        "label": "real",
        "confidence": 0.6,
        "reason": "Heuristic test mode: no obvious fake-review pattern.",
    }


def already_processed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "recommendationid" not in (reader.fieldnames or []):
            return set()
        for row in reader:
            ids.add(str(row["recommendationid"]))
    return ids


def load_existing_counts(output_path: Path) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    if not output_path.exists():
        return counts

    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"game_name", "auth_label"}
        if not required.issubset(set(reader.fieldnames or [])):
            return counts
        for row in reader:
            label = str(row.get("auth_label", "")).strip().lower()
            if label in LABELS:
                update_counts(counts, row.get("game_name", ""), label)
    return counts


def write_rows(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    append: bool,
) -> None:
    mode = "a" if append and output_path.exists() else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def update_counts(counts: dict[str, Counter], game_name: str, label: str) -> None:
    counts["__overall__"][label] += 1
    counts[str(game_name)][label] += 1


def labeled_total_for_game(counts: dict[str, Counter], game_name: str) -> int:
    return sum(counts[str(game_name)].values())


def read_game_names(input_path: Path) -> list[str]:
    games = pd.read_csv(input_path, usecols=["game_name"])["game_name"]
    return sorted(str(game) for game in games.dropna().unique())


def remaining_per_game_total(
    counts: dict[str, Counter], game_names: list[str], limit_per_game: int
) -> int:
    return sum(
        max(0, limit_per_game - labeled_total_for_game(counts, game_name))
        for game_name in game_names
    )


def all_game_limits_met(
    counts: dict[str, Counter], game_names: list[str], limit_per_game: int | None
) -> bool:
    if limit_per_game is None:
        return False
    return all(
        labeled_total_for_game(counts, game_name) >= limit_per_game
        for game_name in game_names
    )


def write_summary(summary_path: Path, counts: dict[str, Counter]) -> None:
    rows = []
    for game_name, counter in sorted(counts.items()):
        total = sum(counter.values())
        if total == 0:
            continue
        rows.append(
            {
                "game_name": "OVERALL" if game_name == "__overall__" else game_name,
                "total": total,
                "real_count": counter["real"],
                "real_percent": round(counter["real"] / total * 100, 2),
                "suspicious_count": counter["suspicious"],
                "suspicious_percent": round(counter["suspicious"] / total * 100, 2),
                "fake_count": counter["fake"],
                "fake_percent": round(counter["fake"] / total * 100, 2),
            }
        )
    pd.DataFrame(rows).to_csv(summary_path, index=False)


def classify_batch(pipe, messages_batch: list[list[dict[str, str]]], args) -> list[dict[str, Any]]:
    outputs = pipe(
        messages_batch,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return [parse_label(extract_generated_text(output)) for output in outputs]


def label_rows(pipe, rows: list[pd.Series], args) -> list[dict[str, Any]]:
    if args.mock:
        return [heuristic_label(row, args.text_max_chars) for row in rows]
    messages_batch = [build_messages(row, args.text_max_chars) for row in rows]
    return classify_batch(pipe, messages_batch, args)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    criteria_path = Path(args.criteria)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    criteria_path.write_text(CRITERIA_MD, encoding="utf-8")

    processed_ids = already_processed_ids(output_path) if args.resume else set()
    if processed_ids:
        print(f"Resume mode: skipping {len(processed_ids):,} already labeled rows.")

    pipe = (
        None
        if args.mock
        else load_model(
            args.model,
            args.force_download,
            args.device,
            args.max_gpu_memory,
            args.cpu_memory,
            args.offload_folder,
            args.load_in_4bit,
            args.load_in_8bit,
        )
    )
    counts = load_existing_counts(output_path) if args.resume else defaultdict(Counter)
    processed = 0
    output_fields: list[str] | None = None
    append_output = args.resume
    game_names = read_game_names(input_path) if args.limit_per_game is not None else []

    if args.limit_per_game is not None:
        remaining_for_game_limit = remaining_per_game_total(
            counts, game_names, args.limit_per_game
        )
        if args.limit is None:
            progress_total = remaining_for_game_limit
        else:
            progress_total = min(args.limit, remaining_for_game_limit)
        if remaining_for_game_limit == 0:
            print(f"All games already have at least {args.limit_per_game:,} labeled rows.")
            write_summary(summary_path, counts)
            return
    else:
        progress_total = args.limit

    reader = pd.read_csv(input_path, chunksize=args.chunk_size)
    progress = tqdm(total=progress_total, desc="Labeling reviews", unit="review")

    for chunk in reader:
        if args.limit is not None and processed >= args.limit:
            break
        if all_game_limits_met(counts, game_names, args.limit_per_game):
            break

        if args.resume and processed_ids:
            chunk = chunk[
                ~chunk["recommendationid"].astype(str).isin(processed_ids)
            ].copy()

        if args.limit is not None:
            remaining = args.limit - processed
            chunk = chunk.head(remaining)

        if chunk.empty:
            continue

        if output_fields is None:
            output_fields = list(chunk.columns) + [
                "auth_label",
                "auth_confidence",
                "auth_reason",
            ]

        labeled_rows: list[dict[str, Any]] = []
        source_rows: list[pd.Series] = []
        pending_by_game: Counter = Counter()

        for _, row in chunk.iterrows():
            game_name = str(row.get("game_name", ""))
            if args.limit_per_game is not None:
                current_total = labeled_total_for_game(counts, game_name)
                queued_total = pending_by_game[game_name]
                if current_total + queued_total >= args.limit_per_game:
                    continue

            source_rows.append(row)
            pending_by_game[game_name] += 1

            if len(source_rows) == args.batch_size:
                labels = label_rows(pipe, source_rows, args)
                for source_row, label_data in zip(source_rows, labels):
                    row_dict = source_row.to_dict()
                    row_dict["auth_label"] = label_data["label"]
                    row_dict["auth_confidence"] = label_data["confidence"]
                    row_dict["auth_reason"] = label_data["reason"]
                    labeled_rows.append(row_dict)
                    update_counts(
                        counts, row_dict.get("game_name", ""), label_data["label"]
                    )
                processed += len(labels)
                progress.update(len(labels))
                source_rows = []
                pending_by_game = Counter()

                if args.limit is not None and processed >= args.limit:
                    break
                if all_game_limits_met(counts, game_names, args.limit_per_game):
                    break

        if source_rows:
            labels = label_rows(pipe, source_rows, args)
            for source_row, label_data in zip(source_rows, labels):
                row_dict = source_row.to_dict()
                row_dict["auth_label"] = label_data["label"]
                row_dict["auth_confidence"] = label_data["confidence"]
                row_dict["auth_reason"] = label_data["reason"]
                labeled_rows.append(row_dict)
                update_counts(counts, row_dict.get("game_name", ""), label_data["label"])
            processed += len(labels)
            progress.update(len(labels))

        if labeled_rows:
            write_rows(
                output_path,
                labeled_rows,
                fieldnames=output_fields,
                append=append_output,
            )
            append_output = True
            write_summary(summary_path, counts)

    progress.close()
    write_summary(summary_path, counts)

    print(f"Done. Labeled rows this run: {processed:,}")
    print(f"Labeled CSV: {output_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Criteria: {criteria_path}")


if __name__ == "__main__":
    main()
