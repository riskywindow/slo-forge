from __future__ import annotations

import os
from typing import Any


class Model:
    def __init__(self, **kwargs: Any) -> None:
        self._engine: Any | None = None

    def load(self) -> None:
        self._engine = "explicit-mock"

    def predict(self, model_input: dict[str, Any]) -> dict[str, Any]:
        if self._engine is None:
            raise RuntimeError("model has not completed load")
        prompt = str(model_input.get("prompt", ""))
        if not prompt:
            raise ValueError("prompt must be non-empty")
        max_tokens = max(1, min(int(model_input.get("max_tokens", 64)), 32768))
        text = f"explicit-mock:{prompt[:64]}"
        return {"choices": [{"text": text}]}
