from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sanitize_model_id(model_id: str) -> str:
    sanitized = model_id.replace("/", "__")
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", sanitized)
    return sanitized.strip("_")


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
