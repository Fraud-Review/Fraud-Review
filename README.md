# LLM Weak Labeling Pipeline

Runnable skeleton for generating pseudo/silver labels from unlabeled text with multiple Hugging Face open-weight LLMs under one shared annotation guideline, prompt version, decoding configuration, and JSON output schema.

This repository does not download models or run heavyweight inference by default. Use `--backend mock` or `--dry-run` today, then implement a local backend later for MLX, `transformers`, `llama.cpp`, Ollama, or vLLM on an Apple Silicon Mac Studio.

## Project Layout

```text
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- config/
|   |-- models.yaml
|   `-- label_schema.yaml
|-- data/
|   |-- input/
|   |   `-- sample_unlabeled.csv
|   `-- output/
|-- prompts/
|   `-- weak_labeling_v1.txt
|-- src/
|   |-- main.py
|   |-- config_loader.py
|   |-- data_loader.py
|   |-- prompt_builder.py
|   |-- schema.py
|   |-- json_parser.py
|   |-- backends/
|   |   |-- base.py
|   |   |-- mock_backend.py
|   |   |-- transformers_backend.py
|   |   `-- ollama_backend.py
|   |-- labeler.py
|   `-- utils.py
`-- tests/
    |-- test_prompt_builder.py
    `-- test_json_parser.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Input

The CSV input must contain:

```text
sample_id,text
```

Example: [data/input/sample_unlabeled.csv](data/input/sample_unlabeled.csv)

## Usage

Dry-run prompt inspection:

```bash
python -m src.main --config config/models.yaml --schema config/label_schema.yaml --input data/input/sample_unlabeled.csv --output-dir data/output --backend mock --dry-run
```

Mock inference:

```bash
python -m src.main --config config/models.yaml --schema config/label_schema.yaml --input data/input/sample_unlabeled.csv --output-dir data/output --backend mock
```

Run only selected models:

```bash
python -m src.main --config config/models.yaml --schema config/label_schema.yaml --input data/input/sample_unlabeled.csv --output-dir data/output --backend mock --models openai/gpt-oss-20b Qwen/Qwen3-14B
```

## Output

The pipeline creates the output directory automatically and writes:

- One append-mode UTF-8 JSONL file per model, for example `data/output/openai__gpt-oss-20b.jsonl`
- One UTF-8 with BOM merged CSV file, by default `data/output/merged_predictions.csv`
- One UTF-8 with BOM summary report CSV, by default `data/output/summary_report.csv`

Each prediction record contains:

```text
run_id
sample_id
model_id
model_family
parameter_size
quantization
backend
prompt_version
label
confidence
rationale
raw_response
parse_success
error_message
created_at
```

The summary report contains:

```text
model_id
total_samples
parse_success_count
parse_failure_count
parse_success_rate
average_confidence
```

## Config

Model selection, model metadata, and decoding live in [config/models.yaml](config/models.yaml). The supplied config includes the four model IDs explicitly requested:

- `openai/gpt-oss-20b`
- `Qwen/Qwen3-14B`
- `google/gemma-4-31B-it`
- `meta-llama/Llama-3.1-8B-Instruct`

The code supports any number of YAML-configured models, so a fifth model can be added by appending another entry under `models`.

The annotation guideline, label definitions, and output field descriptions live in [config/label_schema.yaml](config/label_schema.yaml).

## Backends

Implemented:

- `MockLLMBackend`: deterministic fake labels for testing and pipeline validation.
- `TransformersBackend`: placeholder interface for future local Hugging Face inference.
- `OllamaBackend`: placeholder interface for future local Ollama inference.

The placeholders intentionally do not import heavyweight packages, download models, or call external APIs.

## Research Methodology

Weak labeling is useful when unlabeled text needs initial pseudo/silver labels before manual review, adjudication, active learning, or downstream supervised training.

This pipeline keeps the labeling process controlled by applying:

- The same annotation guideline to every model.
- The same prompt template and prompt version.
- The same decoding configuration.
- The same required JSON output schema.
- The same parser and retry behavior.

For analysis, compare model agreement, disagreement, confidence distributions, and rationales. High-agreement examples can be candidates for silver labels; disagreement cases are often valuable for human review because they expose ambiguous guidelines, edge cases, or model-specific failure modes.

## Testing

```bash
pytest
```

The tests cover prompt rendering and JSON parsing behavior.
