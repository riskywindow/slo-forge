import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.util import percentile, utc_now

TimingAlternative: TypeAlias = Literal["reference", "triton"]


def _gpu_opted_in() -> bool:
    return os.environ.get("SLOFORGE_GENESIS_ALLOW_GPU", "").lower() in {
        "1",
        "true",
        "yes",
    }


class KernelTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implementation: str
    samples_ms: list[float]
    median_ms: float = Field(gt=0)
    p95_ms: float = Field(gt=0)
    mad_ms: float = Field(ge=0)
    median_ci95_ms: tuple[float, float]

    @model_validator(mode="after")
    def validate_raw_samples_and_interval(self) -> Self:
        if len(self.samples_ms) < 5:
            raise ValueError("kernel timing artifacts require at least five raw samples")
        if any(not math.isfinite(sample) or sample < 0 for sample in self.samples_ms):
            raise ValueError("kernel timing samples must be finite and nonnegative")
        lower, upper = self.median_ci95_ms
        if not 0 <= lower <= self.median_ms <= upper or not math.isfinite(upper):
            raise ValueError("median_ci95_ms must be finite, ordered, and contain the median")
        return self


class RandomizedCorrectnessEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["passed"] = "passed"
    seed: int = Field(ge=0)
    shapes: tuple[tuple[int, int], ...]
    trials_per_shape: int = Field(gt=0)
    cases_executed: int = Field(gt=0)
    case_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device_index: int = Field(ge=0)
    hardware_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    relative_tolerance: float = Field(ge=0.0)
    absolute_tolerance: float = Field(ge=0.0)


class InterleavedTimingTrial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_index: int = Field(ge=0)
    first: Literal["reference", "triton"]
    reference_ms: float = Field(gt=0.0)
    triton_ms: float = Field(gt=0.0)


class FusedLogitsBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.kernel-benchmark/v2"] = "sloforge.kernel-benchmark/v2"
    generated_at: str
    seed: int
    shape: tuple[int, int]
    dtype: str
    temperature: float
    repetition_penalty: float
    seen_probability: float
    warmups: int
    samples: int
    device_name: str
    device_index: int = Field(ge=0)
    torch_version: str
    triton_version: str
    correctness: RandomizedCorrectnessEvidence
    raw_trials: tuple[InterleavedTimingTrial, ...]
    raw_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workload_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware_manifest: tuple[str, ...]
    hardware_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    software_manifest: tuple[str, ...]
    software_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    harness_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference: KernelTiming
    triton: KernelTiming
    speedup: float = Field(gt=0)
    paired_improvement_median_percent: float
    paired_improvement_ci95_percent: tuple[float, float]
    practical_significance_percent: float = Field(ge=0.0)
    beneficial: bool
    claim: str
    enablement: str

    @model_validator(mode="after")
    def validate_claim_scope(self) -> Self:
        if self.samples < 20 or len(self.raw_trials) != self.samples:
            raise ValueError("raw interleaved trial count must match the declared sample count")
        if self.correctness.device_index != self.device_index:
            raise ValueError("correctness and benchmark device indices differ")
        if self.correctness.seed != self.seed or self.shape not in self.correctness.shapes:
            raise ValueError("correctness evidence does not cover this seeded benchmark shape")
        if self.correctness.hardware_fingerprint != self.hardware_fingerprint:
            raise ValueError("correctness and benchmark hardware fingerprints differ")
        low, high = self.paired_improvement_ci95_percent
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError("paired improvement interval is invalid")
        if self.beneficial != self.claim.startswith("scoped isolated-kernel speedup"):
            raise ValueError("beneficial flag and scoped claim disagree")
        return self


def triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("the optional logits experiment requires the 'torch' package") from exc


def _require_cuda(*, device: str, device_index: int) -> Any:
    if device != "cuda":
        raise RuntimeError("the logits experiment requires device='cuda'; no fallback is allowed")
    torch = _require_torch()
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device_count = int(torch.cuda.device_count())
    if device_index < 0 or device_index >= device_count:
        raise RuntimeError(
            f"CUDA device index {device_index} is unavailable; detected {device_count} device(s)"
        )
    return torch


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _harness_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _hardware_manifest(torch: Any, device_index: int) -> tuple[str, ...]:
    properties = torch.cuda.get_device_properties(device_index)
    return (
        f"device_index={device_index}",
        f"device_name={torch.cuda.get_device_name(device_index)}",
        f"compute_capability={properties.major}.{properties.minor}",
        f"total_memory={properties.total_memory}",
        f"multiprocessor_count={properties.multi_processor_count}",
        f"device_count={torch.cuda.device_count()}",
    )


def manifest_fingerprint(manifest: Sequence[str]) -> str:
    if not manifest or len(manifest) != len(set(manifest)):
        raise ValueError("provenance manifest must be non-empty and unique")
    return hashlib.sha256("\0".join(manifest).encode()).hexdigest()


def fused_logits_workload_fingerprint(
    *,
    shape: tuple[int, int],
    dtype: str,
    temperature: float,
    repetition_penalty: float,
    seen_probability: float,
    warmups: int,
    samples: int,
    seed: int,
    practical_significance_percent: float,
) -> str:
    return _sha256_json(
        {
            "shape": shape,
            "dtype": dtype,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "seen_probability": seen_probability,
            "warmups": warmups,
            "samples": samples,
            "seed": seed,
            "practical_significance_percent": practical_significance_percent,
            "measurement": "randomized_interleaved_cuda_events",
            "harness_source_sha256": _harness_source_sha256(),
        }
    )


def interleaved_trials_bytes(trials: tuple[InterleavedTimingTrial, ...]) -> bytes:
    return b"".join(
        json.dumps(trial.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for trial in trials
    )


def _validate_inputs(
    logits: Any, seen_mask: Any, *, temperature: float, repetition_penalty: float
) -> None:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
        raise ValueError("repetition_penalty must be finite and at least one")
    if getattr(logits, "ndim", None) != 2:
        raise ValueError("logits must have shape [batch, vocabulary]")
    if getattr(seen_mask, "shape", None) != logits.shape:
        raise ValueError("seen_mask must have the same shape as logits")
    if getattr(seen_mask, "dtype", None) != _require_torch().bool:
        raise ValueError("seen_mask must have boolean dtype")
    if not bool(logits.is_floating_point()):
        raise ValueError("logits must have floating-point dtype")
    if logits.device != seen_mask.device:
        raise ValueError("logits and seen_mask must be on the same device")
    if not bool(logits.is_contiguous()) or not bool(seen_mask.is_contiguous()):
        raise ValueError("the fused logits contract requires contiguous inputs")
    torch = _require_torch()
    if logits.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("logits dtype must be float16, bfloat16, or float32")
    batch, vocabulary = (int(value) for value in logits.shape)
    if not 1 <= batch <= 65_536 or not 1 <= vocabulary <= 131_072:
        raise ValueError("logits shape is outside the bounded fused-kernel domain")


def reference_logits_preprocess(
    logits: Any,
    seen_mask: Any,
    *,
    temperature: float,
    repetition_penalty: float,
) -> Any:
    """PyTorch reference for temperature scaling plus repetition penalty."""
    _validate_inputs(
        logits,
        seen_mask,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    torch = _require_torch()
    penalized = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
    return torch.where(seen_mask, penalized, logits) / temperature


_TRITON_KERNEL: Any | None = None
tl: Any | None = None


def _load_triton_kernel() -> Any:
    global _TRITON_KERNEL, tl
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    if not triton_available():
        raise RuntimeError("the optional Triton logits experiment requires the 'triton' package")
    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")

    @triton.jit  # type: ignore[untyped-decorator]
    def fused_logits_kernel(
        logits_pointer: Any,
        seen_pointer: Any,
        output_pointer: Any,
        vocabulary: Any,
        temperature: Any,
        repetition_penalty: Any,
        BLOCK_SIZE: tl.constexpr,  # type: ignore[name-defined]
    ) -> None:
        row = tl.program_id(axis=0)
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < vocabulary
        offsets = row * vocabulary + columns
        values = tl.load(logits_pointer + offsets, mask=mask, other=0.0)
        seen = tl.load(seen_pointer + offsets, mask=mask, other=0).to(tl.int1)
        penalized = tl.where(
            values < 0,
            values * repetition_penalty,
            values / repetition_penalty,
        )
        output = tl.where(seen, penalized, values) / temperature
        tl.store(output_pointer + offsets, output, mask=mask)

    _TRITON_KERNEL = fused_logits_kernel
    return _TRITON_KERNEL


def triton_logits_preprocess(
    logits: Any,
    seen_mask: Any,
    *,
    temperature: float,
    repetition_penalty: float,
) -> Any:
    """Run the opt-in Triton experiment. It is deliberately never selected automatically."""
    if not _gpu_opted_in():
        raise RuntimeError("set SLOFORGE_GENESIS_ALLOW_GPU=1 before executing Triton code")
    _validate_inputs(
        logits,
        seen_mask,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    if logits.device.type != "cuda":
        raise RuntimeError("Triton logits preprocessing requires CUDA; no CPU fallback is provided")
    triton = importlib.import_module("triton")
    torch = importlib.import_module("torch")
    output = torch.empty_like(logits)
    vocabulary = int(logits.shape[1])
    block_size = int(triton.next_power_of_2(vocabulary))
    if block_size > 131_072:
        raise ValueError("vocabulary larger than 131072 is outside the studied Triton regime")
    kernel = _load_triton_kernel()
    kernel[(int(logits.shape[0]),)](
        logits,
        seen_mask,
        output,
        vocabulary,
        temperature,
        repetition_penalty,
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 32_768 else 4,
    )
    return output


def run_randomized_correctness(
    *,
    seed: int,
    shapes: Sequence[tuple[int, int]] = ((1, 1), (1, 3), (2, 257), (3, 4097)),
    trials_per_shape: int = 3,
    device: str = "cuda",
    device_index: int = 0,
) -> RandomizedCorrectnessEvidence:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 1 << 64:
        raise ValueError("correctness seed must be an unsigned 64-bit integer")
    if not _gpu_opted_in():
        raise RuntimeError("set SLOFORGE_GENESIS_ALLOW_GPU=1 for GPU correctness execution")
    if not 1 <= trials_per_shape <= 100:
        raise ValueError("trials_per_shape must be in [1, 100]")
    torch = _require_cuda(device=device, device_index=device_index)
    if not triton_available():
        raise RuntimeError("randomized Triton correctness requires Triton")
    shape_tuple = tuple(shapes)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    case_manifest: list[dict[str, object]] = []
    with torch.cuda.device(device_index):
        for batch, vocabulary in shape_tuple:
            if not 1 <= batch <= 65_536 or not 1 <= vocabulary <= 131_072:
                raise ValueError("correctness shape is outside the bounded kernel domain")
            for trial_index in range(trials_per_shape):
                cuda_device = f"cuda:{device_index}"
                dtype = rng.choice((torch.float16, torch.bfloat16, torch.float32))
                temperature = rng.choice((0.05, 0.5, 1.0, 2.0))
                penalty = rng.choice((1.0, 1.01, 1.2, 2.0))
                seen_probability = rng.random()
                logits = torch.randn((batch, vocabulary), device=cuda_device, dtype=dtype) * 20
                if vocabulary >= 3:
                    logits[0, :3] = torch.tensor(
                        [-float("inf"), 0.0, float("inf")],
                        device=cuda_device,
                        dtype=dtype,
                    )
                seen = torch.rand((batch, vocabulary), device=cuda_device) < seen_probability
                expected = reference_logits_preprocess(
                    logits,
                    seen,
                    temperature=temperature,
                    repetition_penalty=penalty,
                )
                actual = triton_logits_preprocess(
                    logits,
                    seen,
                    temperature=temperature,
                    repetition_penalty=penalty,
                )
                torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3, equal_nan=True)
                case_manifest.append(
                    {
                        "shape": [batch, vocabulary],
                        "trial_index": trial_index,
                        "dtype": str(dtype),
                        "temperature": temperature,
                        "repetition_penalty": penalty,
                        "seen_probability": seen_probability,
                    }
                )
    hardware = manifest_fingerprint(_hardware_manifest(torch, device_index))
    return RandomizedCorrectnessEvidence(
        seed=seed,
        shapes=shape_tuple,
        trials_per_shape=trials_per_shape,
        cases_executed=len(case_manifest),
        case_manifest_sha256=_sha256_json(case_manifest),
        device_index=device_index,
        hardware_fingerprint=hardware,
        relative_tolerance=2e-3,
        absolute_tolerance=2e-3,
    )


def _bootstrap_median_ci(samples: Sequence[float], *, seed: int) -> tuple[float, float]:
    if len(samples) < 5:
        raise ValueError("robust interval requires at least five samples")
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(samples, k=len(samples))) for _ in range(2_000)]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def _timing_from_samples(samples: Sequence[float], *, seed: int, name: str) -> KernelTiming:
    measured = list(samples)
    median = statistics.median(measured)
    deviations = [abs(value - median) for value in measured]
    return KernelTiming(
        implementation=name,
        samples_ms=measured,
        median_ms=median,
        p95_ms=percentile(measured, 0.95),
        mad_ms=statistics.median(deviations),
        median_ci95_ms=_bootstrap_median_ci(measured, seed=seed),
    )


def _time_one_cuda(
    function: Callable[[], Any], *, torch: Any, name: str, device_index: int
) -> float:
    with torch.cuda.device(device_index):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device_index))
        result = function()
        end.record(torch.cuda.current_stream(device_index))
        end.synchronize()
    if result is None:
        raise RuntimeError(f"{name} benchmark returned no result")
    return max(float(start.elapsed_time(end)), 1e-9)


def _measure_interleaved(
    reference: Callable[[], Any],
    triton: Callable[[], Any],
    *,
    warmups: int,
    samples: int,
    torch: Any,
    seed: int,
    device_index: int,
) -> tuple[InterleavedTimingTrial, ...]:
    implementations: dict[TimingAlternative, Callable[[], Any]] = {
        "reference": reference,
        "triton": triton,
    }
    generator = random.Random(seed)
    with torch.cuda.device(device_index):
        for warmup_index in range(warmups):
            order: tuple[TimingAlternative, TimingAlternative] = (
                ("triton", "reference") if (warmup_index + seed) % 2 else ("reference", "triton")
            )
            for name in order:
                result = implementations[name]()
                if result is None:
                    raise RuntimeError(f"{name} warmup returned no result")
        torch.cuda.synchronize(device_index)
    trials: list[InterleavedTimingTrial] = []
    first_positions: list[TimingAlternative] = [
        "reference" if index % 2 == 0 else "triton" for index in range(samples)
    ]
    generator.shuffle(first_positions)
    for trial_index, first in enumerate(first_positions):
        order = (first, "triton" if first == "reference" else "reference")
        durations: dict[str, float] = {}
        for name in order:
            durations[name] = _time_one_cuda(
                implementations[name],
                torch=torch,
                name=name,
                device_index=device_index,
            )
        trials.append(
            InterleavedTimingTrial(
                trial_index=trial_index,
                first=first,
                reference_ms=durations["reference"],
                triton_ms=durations["triton"],
            )
        )
    return tuple(trials)


def _paired_improvement(
    trials: Sequence[InterleavedTimingTrial], *, seed: int
) -> tuple[float, tuple[float, float]]:
    improvements = tuple(
        (trial.reference_ms - trial.triton_ms) / trial.reference_ms * 100.0 for trial in trials
    )
    generator = random.Random(seed)
    medians = [
        statistics.median(generator.choices(improvements, k=len(improvements)))
        for _ in range(2_000)
    ]
    return statistics.median(improvements), (
        percentile(medians, 0.025),
        percentile(medians, 0.975),
    )


def benchmark_fused_logits(
    *,
    batch: int,
    vocabulary: int,
    dtype_name: str,
    temperature: float,
    repetition_penalty: float,
    seen_probability: float,
    warmups: int,
    samples: int,
    seed: int,
    enable_triton_experiment: bool,
    device: str = "cuda",
    device_index: int = 0,
    practical_significance_percent: float = 10.0,
    correctness: RandomizedCorrectnessEvidence | None = None,
) -> FusedLogitsBenchmark:
    """Benchmark only after an explicit opt-in; no result controls production enablement."""
    if not enable_triton_experiment:
        raise RuntimeError("pass enable_triton_experiment=True to run this optional experiment")
    if not _gpu_opted_in():
        raise RuntimeError("set SLOFORGE_GENESIS_ALLOW_GPU=1 for the optional GPU experiment")
    if not 1 <= batch <= 65_536 or not 1 <= vocabulary <= 131_072:
        raise ValueError("batch and vocabulary are outside the bounded kernel domain")
    if not math.isfinite(seen_probability) or not 0 <= seen_probability <= 1:
        raise ValueError("seen_probability must be finite and between zero and one")
    if not 3 <= warmups <= 1_000 or not 20 <= samples <= 100_000:
        raise ValueError(
            "kernel benchmark requires warmups in [3, 1000] and samples in [20, 100000]"
        )
    if not math.isfinite(practical_significance_percent) or practical_significance_percent < 0:
        raise ValueError("practical significance must be finite and non-negative")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 1 << 64:
        raise ValueError("benchmark seed must be an unsigned 64-bit integer")
    if correctness is not None:
        raise ValueError(
            "externally supplied correctness evidence is not trusted; the GPU benchmark reruns it"
        )
    torch = _require_cuda(device=device, device_index=device_index)
    if not triton_available():
        raise RuntimeError("Triton benchmark requested but Triton is not installed")
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtypes.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported dtype {dtype_name!r}; expected one of {sorted(dtypes)}")
    with torch.cuda.device(device_index):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        hardware_manifest = _hardware_manifest(torch, device_index)
        hardware_fingerprint = manifest_fingerprint(hardware_manifest)
        correctness_evidence = run_randomized_correctness(
            seed=seed,
            shapes=((1, 1), (batch, vocabulary)),
            trials_per_shape=2,
            device=device,
            device_index=device_index,
        )
        if correctness_evidence.hardware_fingerprint != hardware_fingerprint:
            raise ValueError("correctness evidence was measured on different hardware")
        cuda_device = f"cuda:{device_index}"
        logits = torch.randn((batch, vocabulary), device=cuda_device, dtype=dtype)
        seen = torch.rand((batch, vocabulary), device=cuda_device) < seen_probability
        arguments = {
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        }
        expected = reference_logits_preprocess(logits, seen, **arguments)
        actual = triton_logits_preprocess(logits, seen, **arguments)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        trials = _measure_interleaved(
            lambda: reference_logits_preprocess(logits, seen, **arguments),
            lambda: triton_logits_preprocess(logits, seen, **arguments),
            warmups=warmups,
            samples=samples,
            torch=torch,
            seed=seed,
            device_index=device_index,
        )
    if {trial.first for trial in trials} != {"reference", "triton"}:
        raise RuntimeError("randomized measurement order did not exercise both first positions")
    reference_timing = _timing_from_samples(
        [trial.reference_ms for trial in trials],
        seed=seed + 1,
        name="pytorch-reference",
    )
    triton_timing = _timing_from_samples(
        [trial.triton_ms for trial in trials],
        seed=seed + 2,
        name="triton-fused",
    )
    speedup = reference_timing.median_ms / triton_timing.median_ms
    improvement, improvement_interval = _paired_improvement(trials, seed=seed + 3)
    beneficial = improvement_interval[0] > practical_significance_percent
    software_manifest = (
        f"python={platform.python_version()}",
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"triton={importlib.metadata.version('triton')}",
        "timer=cuda_event_elapsed_time",
        "order=randomized_interleaved_pairs",
    )
    report = FusedLogitsBenchmark(
        generated_at=utc_now(),
        seed=seed,
        shape=(batch, vocabulary),
        dtype=dtype_name,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        seen_probability=seen_probability,
        warmups=warmups,
        samples=samples,
        device_name=str(torch.cuda.get_device_name(device_index)),
        device_index=device_index,
        torch_version=str(torch.__version__),
        triton_version=importlib.metadata.version("triton"),
        correctness=correctness_evidence,
        raw_trials=trials,
        raw_samples_sha256=hashlib.sha256(interleaved_trials_bytes(trials)).hexdigest(),
        workload_fingerprint=fused_logits_workload_fingerprint(
            shape=(batch, vocabulary),
            dtype=dtype_name,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            seen_probability=seen_probability,
            warmups=warmups,
            samples=samples,
            seed=seed,
            practical_significance_percent=practical_significance_percent,
        ),
        hardware_manifest=hardware_manifest,
        hardware_fingerprint=hardware_fingerprint,
        software_manifest=software_manifest,
        software_manifest_sha256=manifest_fingerprint(software_manifest),
        harness_source_sha256=_harness_source_sha256(),
        reference=reference_timing,
        triton=triton_timing,
        speedup=speedup,
        paired_improvement_median_percent=improvement,
        paired_improvement_ci95_percent=improvement_interval,
        practical_significance_percent=practical_significance_percent,
        beneficial=beneficial,
        claim=(
            "scoped isolated-kernel speedup supported by the paired confidence gate; "
            "no end-to-end serving claim"
            if beneficial
            else "no speedup claim; paired confidence interval did not clear the practical gate"
        ),
        enablement="experimental-only; never enabled by the SLOForge runtime",
    )
    validate_fused_logits_benchmark(report)
    return report


def validate_fused_logits_benchmark(report: FusedLogitsBenchmark) -> None:
    """Recompute every timing and acceptance field from paired raw GPU trials."""

    if [trial.trial_index for trial in report.raw_trials] != list(range(report.samples)):
        raise ValueError("interleaved GPU trial indices are incomplete or reordered")
    if {trial.first for trial in report.raw_trials} != {"reference", "triton"}:
        raise ValueError("interleaved GPU trials do not include both first positions")
    digest = hashlib.sha256(interleaved_trials_bytes(report.raw_trials)).hexdigest()
    if digest != report.raw_samples_sha256:
        raise ValueError("interleaved GPU raw sample digest mismatch")
    expected_workload = fused_logits_workload_fingerprint(
        shape=report.shape,
        dtype=report.dtype,
        temperature=report.temperature,
        repetition_penalty=report.repetition_penalty,
        seen_probability=report.seen_probability,
        warmups=report.warmups,
        samples=report.samples,
        seed=report.seed,
        practical_significance_percent=report.practical_significance_percent,
    )
    if expected_workload != report.workload_fingerprint:
        raise ValueError("GPU workload fingerprint is not reproducible")
    if manifest_fingerprint(report.hardware_manifest) != report.hardware_fingerprint:
        raise ValueError("GPU hardware fingerprint is not derived from its manifest")
    if manifest_fingerprint(report.software_manifest) != report.software_manifest_sha256:
        raise ValueError("GPU software manifest digest mismatch")
    if report.harness_source_sha256 != _harness_source_sha256():
        raise ValueError("GPU benchmark harness source changed")
    if any(
        "unexercised" in item or "schema-only" in item
        for item in (
            *report.hardware_manifest,
            *report.software_manifest,
        )
    ):
        raise ValueError("unexercised schema fixtures are not measured GPU evidence")
    if (
        f"device_name={report.device_name}" not in report.hardware_manifest
        or f"torch={report.torch_version}" not in report.software_manifest
        or f"triton={report.triton_version}" not in report.software_manifest
    ):
        raise ValueError("GPU summary fields are not bound to provenance manifests")
    expected_first: list[TimingAlternative] = [
        "reference" if index % 2 == 0 else "triton" for index in range(report.samples)
    ]
    random.Random(report.seed).shuffle(expected_first)
    if [trial.first for trial in report.raw_trials] != expected_first:
        raise ValueError("interleaved GPU order is not deterministically seed-derived")
    reference = _timing_from_samples(
        [trial.reference_ms for trial in report.raw_trials],
        seed=report.seed + 1,
        name="pytorch-reference",
    )
    triton = _timing_from_samples(
        [trial.triton_ms for trial in report.raw_trials],
        seed=report.seed + 2,
        name="triton-fused",
    )
    improvement, interval = _paired_improvement(report.raw_trials, seed=report.seed + 3)
    speedup = reference.median_ms / triton.median_ms
    beneficial = interval[0] > report.practical_significance_percent
    expected_claim = (
        "scoped isolated-kernel speedup supported by the paired confidence gate; "
        "no end-to-end serving claim"
        if beneficial
        else "no speedup claim; paired confidence interval did not clear the practical gate"
    )
    if (
        reference != report.reference
        or triton != report.triton
        or speedup != report.speedup
        or improvement != report.paired_improvement_median_percent
        or interval != report.paired_improvement_ci95_percent
        or beneficial != report.beneficial
        or expected_claim != report.claim
    ):
        raise ValueError("GPU performance claim is not derived from paired raw trials")
