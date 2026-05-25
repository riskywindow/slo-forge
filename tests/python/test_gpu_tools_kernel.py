from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _kernel() -> ModuleType:
    path = Path(__file__).parents[2] / "kernels" / "fused_logits.py"
    specification = importlib.util.spec_from_file_location("test_fused_logits", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_reference_logits_preprocess_handles_sign_and_seen_mask() -> None:
    torch = pytest.importorskip("torch")
    kernel: Any = _kernel()
    logits = torch.tensor([[-2.0, 2.0, 3.0, -4.0]])
    seen = torch.tensor([[True, True, False, False]])
    result = kernel.reference_logits_preprocess(
        logits, seen, temperature=0.5, repetition_penalty=2.0
    )
    torch.testing.assert_close(result, torch.tensor([[-8.0, 2.0, 6.0, -8.0]]))


@pytest.mark.parametrize(
    ("temperature", "penalty", "message"),
    [(0.0, 1.0, "temperature"), (1.0, 0.99, "repetition_penalty")],
)
def test_reference_rejects_invalid_parameters(
    temperature: float, penalty: float, message: str
) -> None:
    torch = pytest.importorskip("torch")
    kernel: Any = _kernel()
    logits = torch.zeros((1, 3))
    with pytest.raises(ValueError, match=message):
        kernel.reference_logits_preprocess(
            logits,
            torch.zeros_like(logits, dtype=torch.bool),
            temperature=temperature,
            repetition_penalty=penalty,
        )


def test_triton_benchmark_is_never_default_enabled() -> None:
    kernel: Any = _kernel()
    with pytest.raises(RuntimeError, match="enable_triton_experiment"):
        kernel.benchmark_fused_logits(
            batch=1,
            vocabulary=32,
            dtype_name="float32",
            temperature=1.0,
            repetition_penalty=1.0,
            seen_probability=0.1,
            warmups=3,
            samples=20,
            seed=1,
            enable_triton_experiment=False,
        )


def test_gpu_benchmark_artifact_schema_validates_without_fabricating_measurements() -> None:
    kernel: Any = _kernel()
    timing = {
        "implementation": "schema-validation-only",
        "samples_ms": [1.0] * 5,
        "median_ms": 1.0,
        "p95_ms": 1.0,
        "mad_ms": 0.0,
        "median_ci95_ms": [1.0, 1.0],
    }
    artifact = kernel.FusedLogitsBenchmark.model_validate(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "seed": 7,
            "shape": [1, 32],
            "dtype": "float16",
            "temperature": 1.0,
            "repetition_penalty": 1.1,
            "seen_probability": 0.1,
            "warmups": 3,
            "device_name": "schema-only-no-device-claim",
            "device_index": 0,
            "torch_version": "unexercised",
            "triton_version": "unexercised",
            "reference": timing,
            "triton": timing,
            "speedup": 1.0,
            "beneficial": False,
            "enablement": "experimental-only; never enabled by the SLOForge runtime",
        }
    )
    assert artifact.schema_version == "sloforge.kernel-benchmark/v1"
    assert artifact.beneficial is False
