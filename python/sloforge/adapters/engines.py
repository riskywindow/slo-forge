from __future__ import annotations

import importlib.util
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EngineProbeRequest:
    prompt: str
    max_new_tokens: int


@dataclass(frozen=True)
class EngineProbeResult:
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    e2e_ms: float
    token_timestamps_ms: tuple[float, ...]
    peak_memory_bytes: int | None


class EngineAdapter(ABC):
    """Explicit real-engine adapter; unavailable engines fail instead of falling back."""

    name: str
    package: str

    def validate_available(self) -> None:
        if importlib.util.find_spec(self.package) is None:
            raise RuntimeError(
                f"engine {self.name!r} was requested but package {self.package!r} is not installed"
            )

    @abstractmethod
    def serve_command(
        self, *, model: str, host: str, port: int, dtype: str, max_model_len: int
    ) -> list[str]:
        """Return a bounded server command; the profiler owns process lifecycle."""

    def direct_probe(
        self, *, model: str, requests: Iterable[EngineProbeRequest], device: str
    ) -> list[EngineProbeResult]:
        raise RuntimeError(f"engine {self.name!r} requires its server-mode profiler")


class TransformersAdapter(EngineAdapter):
    name = "transformers"
    package = "transformers"

    def serve_command(
        self, *, model: str, host: str, port: int, dtype: str, max_model_len: int
    ) -> list[str]:
        del model, host, port, dtype, max_model_len
        raise RuntimeError("Transformers is a direct correctness baseline, not a server adapter")

    def direct_probe(
        self, *, model: str, requests: Iterable[EngineProbeRequest], device: str
    ) -> list[EngineProbeResult]:
        self.validate_available()
        if device != "cuda":
            raise RuntimeError(
                "the GPU profiling path requires device='cuda'; no device fallback is allowed"
            )
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiling requested but torch.cuda.is_available() is false")
        tokenizer = AutoTokenizer.from_pretrained(model)
        loaded = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto").to("cuda")
        loaded.eval()
        results: list[EngineProbeResult] = []
        for request in requests:
            encoded = tokenizer(request.prompt, return_tensors="pt").to("cuda")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            with torch.inference_mode():
                generated = loaded.generate(**encoded, max_new_tokens=request.max_new_tokens)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            output_tokens = int(generated.shape[-1] - encoded.input_ids.shape[-1])
            # Non-streaming generate cannot expose true TTFT; one-token timing is measured separately.
            torch.cuda.synchronize()
            first_started = time.perf_counter_ns()
            with torch.inference_mode():
                loaded.generate(**encoded, max_new_tokens=1)
            torch.cuda.synchronize()
            ttft_ms = (time.perf_counter_ns() - first_started) / 1e6
            interval = max((elapsed_ms - ttft_ms) / max(output_tokens - 1, 1), 0.0)
            timestamps = tuple(ttft_ms + index * interval for index in range(output_tokens))
            results.append(
                EngineProbeResult(
                    prompt_tokens=int(encoded.input_ids.shape[-1]),
                    output_tokens=output_tokens,
                    ttft_ms=ttft_ms,
                    e2e_ms=elapsed_ms,
                    token_timestamps_ms=timestamps,
                    peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
                )
            )
        return results


class VllmAdapter(EngineAdapter):
    name = "vllm"
    package = "vllm"

    def serve_command(
        self, *, model: str, host: str, port: int, dtype: str, max_model_len: int
    ) -> list[str]:
        self.validate_available()
        return [
            "vllm",
            "serve",
            model,
            "--host",
            host,
            "--port",
            str(port),
            "--dtype",
            dtype,
            "--max-model-len",
            str(max_model_len),
        ]


class SglangAdapter(EngineAdapter):
    name = "sglang"
    package = "sglang"

    def serve_command(
        self, *, model: str, host: str, port: int, dtype: str, max_model_len: int
    ) -> list[str]:
        self.validate_available()
        return [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            host,
            "--port",
            str(port),
            "--dtype",
            dtype,
            "--context-length",
            str(max_model_len),
        ]


class TensorRtLlmAdapter(EngineAdapter):
    name = "tensorrt-llm"
    package = "tensorrt_llm"

    def serve_command(
        self, *, model: str, host: str, port: int, dtype: str, max_model_len: int
    ) -> list[str]:
        self.validate_available()
        return [
            "trtllm-serve",
            "--model",
            model,
            "--host",
            host,
            "--port",
            str(port),
            "--dtype",
            dtype,
            "--max_seq_len",
            str(max_model_len),
        ]


def get_engine_adapter(
    name: Literal["transformers", "vllm", "sglang", "tensorrt-llm"] | str,
) -> EngineAdapter:
    adapters: dict[str, type[EngineAdapter]] = {
        "transformers": TransformersAdapter,
        "vllm": VllmAdapter,
        "sglang": SglangAdapter,
        "tensorrt-llm": TensorRtLlmAdapter,
    }
    adapter = adapters.get(name)
    if adapter is None:
        raise ValueError(f"unsupported engine {name!r}; expected one of {sorted(adapters)}")
    return adapter()


def inter_token_latencies(timestamps_ms: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        timestamps_ms[index] - timestamps_ms[index - 1] for index in range(1, len(timestamps_ms))
    )


def median_itl(result: EngineProbeResult) -> float:
    samples = inter_token_latencies(result.token_timestamps_ms)
    return statistics.median(samples) if samples else 0.0
