from __future__ import annotations

import hashlib
import json

from src.backends.base import LLMBackend
from src.schema import GenerationRequest


class MockLLMBackend(LLMBackend):
    backend_name = "mock"

    def generate(self, request: GenerationRequest) -> str:
        labels = request.allowed_labels
        if not labels:
            raise ValueError("Mock backend requires at least one allowed label.")

        key = f"{request.model_id}|{request.sample.sample_id}|{request.sample.text}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        value = int(digest[:12], 16)
        label = labels[value % len(labels)]
        confidence = round(0.55 + ((value // len(labels)) % 41) / 100, 2)

        response = {
            "label": label,
            "confidence": confidence,
            "rationale": "Deterministic mock label for pipeline testing; not a real model judgment.",
        }
        return json.dumps(response, ensure_ascii=False)
