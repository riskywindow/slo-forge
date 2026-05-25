import importlib
import importlib.metadata
import importlib.util
import math
import random
import statistics
from collections.abc import Callable, Sequence
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.util import percentile, utc_now


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


class FusedLogitsBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "sloforge.kernel-benchmark/v1"
    generated_at: str
    seed: int
    shape: tuple[int, int]
    dtype: str
    temperature: float
    repetition_penalty: float
    seen_probability: float
    warmups: int
    device_name: str
    device_index: int = Field(ge=0)
    torch_version: str
    triton_version: str
    reference: KernelTiming
    triton: KernelTiming
    speedup: float = Field(gt=0)
    beneficial: bool
    enablement: str


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


def _validate_inputs(
    logits: Any, seen_mask: Any, *, temperature: float, repetition_penalty: float
) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if repetition_penalty < 1:
        raise ValueError("repetition_penalty must be at least one")
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
) -> None:
    if trials_per_shape < 1:
        raise ValueError("trials_per_shape must be positive")
    torch = _require_cuda(device=device, device_index=device_index)
    if not triton_available():
        raise RuntimeError("randomized Triton correctness requires Triton")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for batch, vocabulary in shapes:
        if batch < 1 or vocabulary < 1:
            raise ValueError("correctness shapes must be positive")
        for _ in range(trials_per_shape):
            cuda_device = f"cuda:{device_index}"
            dtype = rng.choice((torch.float16, torch.bfloat16, torch.float32))
            temperature = rng.choice((0.05, 0.5, 1.0, 2.0))
            penalty = rng.choice((1.0, 1.01, 1.2, 2.0))
            logits = torch.randn((batch, vocabulary), device=cuda_device, dtype=dtype) * 20
            if vocabulary >= 3:
                logits[0, :3] = torch.tensor(
                    [-float("inf"), 0.0, float("inf")], device=cuda_device, dtype=dtype
                )
            seen = torch.rand((batch, vocabulary), device=cuda_device) < rng.random()
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


def _bootstrap_median_ci(samples: Sequence[float], *, seed: int) -> tuple[float, float]:
    if len(samples) < 5:
        raise ValueError("robust interval requires at least five samples")
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(samples, k=len(samples))) for _ in range(2_000)]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def _time_cuda(
    function: Callable[[], Any], *, warmups: int, samples: int, torch: Any, seed: int, name: str
) -> KernelTiming:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    measured: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        if result is None:
            raise RuntimeError(f"{name} benchmark returned no result")
        measured.append(float(start.elapsed_time(end)))
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
) -> FusedLogitsBenchmark:
    """Benchmark only after an explicit opt-in; no result controls production enablement."""
    if not enable_triton_experiment:
        raise RuntimeError("pass enable_triton_experiment=True to run this optional experiment")
    if batch < 1 or vocabulary < 1:
        raise ValueError("batch and vocabulary must be positive")
    if not 0 <= seen_probability <= 1:
        raise ValueError("seen_probability must be between zero and one")
    if warmups < 3 or samples < 20:
        raise ValueError("kernel benchmark requires at least 3 warmups and 20 samples")
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    reference_timing = _time_cuda(
        lambda: reference_logits_preprocess(logits, seen, **arguments),
        warmups=warmups,
        samples=samples,
        torch=torch,
        seed=seed,
        name="pytorch-reference",
    )
    triton_timing = _time_cuda(
        lambda: triton_logits_preprocess(logits, seen, **arguments),
        warmups=warmups,
        samples=samples,
        torch=torch,
        seed=seed + 1,
        name="triton-fused",
    )
    speedup = reference_timing.median_ms / triton_timing.median_ms
    return FusedLogitsBenchmark(
        generated_at=utc_now(),
        seed=seed,
        shape=(batch, vocabulary),
        dtype=dtype_name,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        seen_probability=seen_probability,
        warmups=warmups,
        device_name=str(torch.cuda.get_device_name(device_index)),
        device_index=device_index,
        torch_version=str(torch.__version__),
        triton_version=importlib.metadata.version("triton"),
        reference=reference_timing,
        triton=triton_timing,
        speedup=speedup,
        # Require a margin larger than normal timing noise before calling it beneficial.
        beneficial=speedup >= 1.10,
        enablement="experimental-only; never enabled by the SLOForge runtime",
    )
