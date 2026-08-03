from __future__ import annotations

import hashlib
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


def test_triton_benchmark_requires_environment_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLOFORGE_GENESIS_ALLOW_GPU", raising=False)
    kernel: Any = _kernel()
    with pytest.raises(RuntimeError, match="SLOFORGE_GENESIS_ALLOW_GPU"):
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
            enable_triton_experiment=True,
        )


def test_reference_rejects_nonfinite_parameters_and_undeclared_layouts() -> None:
    torch = pytest.importorskip("torch")
    kernel: Any = _kernel()
    logits = torch.zeros((2, 3)).transpose(0, 1)
    seen = torch.zeros_like(logits, dtype=torch.bool)
    with pytest.raises(ValueError, match="contiguous"):
        kernel.reference_logits_preprocess(
            logits,
            seen,
            temperature=1.0,
            repetition_penalty=1.0,
        )
    contiguous = torch.zeros((1, 3))
    with pytest.raises(ValueError, match="finite"):
        kernel.reference_logits_preprocess(
            contiguous,
            torch.zeros_like(contiguous, dtype=torch.bool),
            temperature=float("nan"),
            repetition_penalty=1.0,
        )


def test_schema_only_gpu_artifact_is_rejected_as_measurement() -> None:
    kernel: Any = _kernel()
    trials = tuple(
        kernel.InterleavedTimingTrial(
            trial_index=index,
            first="reference" if index % 2 == 0 else "triton",
            reference_ms=1.0,
            triton_ms=1.0,
        )
        for index in range(20)
    )
    timing = {
        "implementation": "pytorch-reference",
        "samples_ms": [1.0] * 20,
        "median_ms": 1.0,
        "p95_ms": 1.0,
        "mad_ms": 0.0,
        "median_ci95_ms": [1.0, 1.0],
    }
    triton_timing = {**timing, "implementation": "triton-fused"}
    hardware_manifest = (
        "device_index=0",
        "device_name=schema-only-no-device-claim",
        "compute_capability=unexercised",
        "total_memory=0",
        "multiprocessor_count=0",
        "device_count=0",
    )
    hardware_fingerprint = kernel.manifest_fingerprint(hardware_manifest)
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
            "samples": 20,
            "device_name": "schema-only-no-device-claim",
            "device_index": 0,
            "torch_version": "unexercised",
            "triton_version": "unexercised",
            "correctness": {
                "seed": 7,
                "shapes": [[1, 32]],
                "trials_per_shape": 1,
                "cases_executed": 1,
                "case_manifest_sha256": "0" * 64,
                "device_index": 0,
                "hardware_fingerprint": hardware_fingerprint,
                "relative_tolerance": 0.002,
                "absolute_tolerance": 0.002,
            },
            "raw_trials": [item.model_dump(mode="json") for item in trials],
            "raw_samples_sha256": hashlib.sha256(
                kernel.interleaved_trials_bytes(trials)
            ).hexdigest(),
            "workload_fingerprint": kernel.fused_logits_workload_fingerprint(
                shape=(1, 32),
                dtype="float16",
                temperature=1.0,
                repetition_penalty=1.1,
                seen_probability=0.1,
                warmups=3,
                samples=20,
                seed=7,
                practical_significance_percent=10.0,
            ),
            "hardware_manifest": hardware_manifest,
            "hardware_fingerprint": hardware_fingerprint,
            "software_manifest": ["torch=unexercised", "triton=unexercised"],
            "software_manifest_sha256": kernel.manifest_fingerprint(
                ("torch=unexercised", "triton=unexercised")
            ),
            "harness_source_sha256": kernel._harness_source_sha256(),
            "reference": timing,
            "triton": triton_timing,
            "speedup": 1.0,
            "paired_improvement_median_percent": 0.0,
            "paired_improvement_ci95_percent": [0.0, 0.0],
            "practical_significance_percent": 10.0,
            "beneficial": False,
            "claim": "no speedup claim; paired confidence interval did not clear the practical gate",
            "enablement": "experimental-only; never enabled by the SLOForge runtime",
        }
    )
    assert artifact.schema_version == "sloforge.kernel-benchmark/v2"
    assert artifact.beneficial is False
    with pytest.raises(ValueError, match="unexercised schema fixtures"):
        kernel.validate_fused_logits_benchmark(artifact)
