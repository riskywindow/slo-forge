from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt: str = Field(min_length=1, max_length=1_000_000)
    max_tokens: int = Field(default=64, ge=1, le=4096)
    stream: bool = True
    temperature: float = Field(default=0.0, ge=0)


class TransformersBackend:
    """Small correctness server for the Transformers baseline.

    vLLM and SGLang remain the performance-serving paths. This implementation deliberately uses
    explicit token-by-token decoding so TTFT and token delivery are observable without relying on
    an unbounded library streamer queue.
    """

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: Literal["cpu", "cuda"],
        max_concurrency: int,
    ) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("transformers backend requested CUDA but torch cannot access it")
        self.torch = torch
        self.model_id = model_id
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype="auto",
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()
        self.capacity = threading.BoundedSemaphore(max_concurrency)

    def try_reserve(self) -> bool:
        """Reserve one bounded inference slot before response headers are committed."""
        return self.capacity.acquire(blocking=False)

    def stream_reserved(self, request: CompletionRequest) -> Iterator[bytes]:
        """Stream a request for which `try_reserve` already succeeded."""
        if not request.stream:
            self.capacity.release()
            raise ValueError("the baseline endpoint requires stream=true")
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        try:
            encoded = self.tokenizer(request.prompt, return_tensors="pt")
            input_ids = encoded.input_ids.to(self.device)
            attention_mask = encoded.attention_mask.to(self.device)
            past_key_values: Any | None = None
            emitted = 0
            for _ in range(request.max_tokens):
                model_input = input_ids if past_key_values is None else input_ids[:, -1:]
                with self.torch.inference_mode():
                    output = self.model(
                        input_ids=model_input,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                logits = output.logits[:, -1, :]
                if request.temperature > 0:
                    probabilities = self.torch.softmax(logits / request.temperature, dim=-1)
                    next_token = self.torch.multinomial(probabilities, num_samples=1)
                else:
                    next_token = self.torch.argmax(logits, dim=-1, keepdim=True)
                past_key_values = output.past_key_values
                input_ids = self.torch.cat((input_ids, next_token), dim=-1)
                attention_mask = self.torch.cat(
                    (
                        attention_mask,
                        self.torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=self.device,
                        ),
                    ),
                    dim=-1,
                )
                token_id = int(next_token.item())
                if token_id == self.tokenizer.eos_token_id:
                    break
                text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                emitted += 1
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": self.model_id,
                    "choices": [{"index": 0, "text": text, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()
            final = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": self.model_id,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                "usage": {"completion_tokens": emitted},
            }
            yield f"data: {json.dumps(final, separators=(',', ':'))}\n\ndata: [DONE]\n\n".encode()
        finally:
            self.capacity.release()


def create_app(backend: TransformersBackend) -> Any:
    from fastapi import FastAPI, HTTPException  # type: ignore[import-not-found]
    from fastapi.responses import StreamingResponse  # type: ignore[import-not-found]

    @asynccontextmanager
    async def lifespan(app: Any) -> Any:
        del app
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")  # type: ignore[untyped-decorator]
    async def health() -> dict[str, str]:
        return {"status": "ready", "model": backend.model_id}

    @app.post("/v1/completions")  # type: ignore[untyped-decorator]
    async def completions(request: CompletionRequest) -> Any:
        if not request.stream:
            raise HTTPException(
                status_code=400, detail="the baseline endpoint requires stream=true"
            )
        if not backend.try_reserve():
            raise HTTPException(
                status_code=429, detail="transformers baseline concurrency is saturated"
            )
        return StreamingResponse(backend.stream_reserved(request), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded Transformers correctness backend")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--max-concurrency", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    backend = TransformersBackend(
        model_id=arguments.model,
        revision=arguments.revision,
        device=arguments.device,
        max_concurrency=arguments.max_concurrency,
    )
    import uvicorn  # type: ignore[import-not-found]

    uvicorn.run(create_app(backend), host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
