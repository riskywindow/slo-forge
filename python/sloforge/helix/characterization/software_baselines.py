"""Bounded CPU software baselines for state transform and transport operations.

The fixture shapes in this module are controlled synthetic workloads. Timings are
real host monotonic/process-time observations, but are not a production workload
distribution and contain no GPU, PCIe, NIC, or network-hardware measurements.
"""

from __future__ import annotations

import hashlib
import math
import platform
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Literal, Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.continuum.adapters import (
    CapturedState,
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
)
from sloforge.continuum.conversion import (
    KVLayout,
    KVLayoutKind,
    captured_attention_state,
    direct_convert_capture,
)
from sloforge.continuum.reference.codec import decode_captured, encode_state
from sloforge.continuum.storage import ChunkRef, MemoryContentStore
from sloforge.continuum.transport import InProcessTransport

MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_TRANSFORM_TOKENS = 512
MAX_REPETITIONS = 50
MAX_WARMUPS = 10
MAX_SAMPLES = 20_000
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
FANOUTS = (2, 4, 8, 16, 32, 64, 128)

_T = TypeVar("_T")


class SoftwareBaselineModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class BaselineFamily(StrEnum):
    TRANSFORM = "transform"
    HASH = "hash"
    IN_PROCESS_TRANSFER = "in_process_transfer"
    SOFTWARE_FANOUT = "software_fanout"


class SoftwareBaselineConfig(SoftwareBaselineModel):
    schema_version: Literal["sloforge.branchfabric.software-baseline-config/v1"] = (
        "sloforge.branchfabric.software-baseline-config/v1"
    )
    seed: int = Field(ge=0, le=2**63 - 1)
    payload_bytes: int = Field(default=256 * 1024, ge=4096, le=MAX_PAYLOAD_BYTES)
    transform_output_tokens: int = Field(default=128, ge=1, le=MAX_TRANSFORM_TOKENS)
    transform_maximum_temporary_bytes: int = Field(default=512, ge=256, le=1024 * 1024)
    chunk_sizes: tuple[int, ...] = (4096, 16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024)
    transfer_concurrency: tuple[int, ...] = (1, 2, 4, 8)
    fanouts: tuple[int, ...] = FANOUTS
    fanout_max_workers: int = Field(default=8, ge=1, le=32)
    warmup_repetitions: int = Field(default=2, ge=0, le=MAX_WARMUPS)
    measurement_repetitions: int = Field(default=7, ge=1, le=MAX_REPETITIONS)
    sample_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        le=300.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_study_bound(self) -> Self:
        if not self.chunk_sizes or len(self.chunk_sizes) != len(set(self.chunk_sizes)):
            raise ValueError("chunk sizes must be non-empty and unique")
        if tuple(sorted(self.chunk_sizes)) != self.chunk_sizes:
            raise ValueError("chunk sizes must be strictly increasing")
        if self.chunk_sizes[0] < 4096 or self.chunk_sizes[-1] > 4 * 1024 * 1024:
            raise ValueError("chunk sizes must be within 4 KiB..4 MiB")
        if (
            not self.transfer_concurrency
            or len(self.transfer_concurrency) != len(set(self.transfer_concurrency))
            or tuple(sorted(self.transfer_concurrency)) != self.transfer_concurrency
            or self.transfer_concurrency[0] != 1
            or self.transfer_concurrency[-1] > 16
        ):
            raise ValueError(
                "transfer concurrency must be unique, increasing, begin at 1, and <=16"
            )
        if self.fanouts != FANOUTS:
            raise ValueError("the required software fanout sweep is 2,4,8,16,32,64,128")
        case_count = (
            2
            + 1
            + len(self.chunk_sizes)
            + len(self.chunk_sizes) * len(self.transfer_concurrency)
            + 2 * len(self.fanouts)
        )
        sample_count = case_count * (self.warmup_repetitions + self.measurement_repetitions)
        if sample_count > MAX_SAMPLES:
            raise ValueError(f"software baseline study exceeds {MAX_SAMPLES} samples")
        return self


class EquivalenceProof(SoftwareBaselineModel):
    proof_id: str = Field(min_length=1, max_length=256)
    compared_implementations: tuple[str, ...] = Field(min_length=2, max_length=64)
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_sha256: tuple[str, ...] = Field(min_length=2, max_length=64)
    exact_byte_or_semantic_equality: Literal[True] = True
    timing_sample: Literal[False] = False
    method: str = Field(min_length=1, max_length=4096)


class SoftwareBaselineSample(SoftwareBaselineModel):
    schema_version: Literal["sloforge.branchfabric.software-baseline-sample/v1"] = (
        "sloforge.branchfabric.software-baseline-sample/v1"
    )
    sample_id: str = Field(min_length=1, max_length=512)
    case_id: str = Field(min_length=1, max_length=512)
    family: BaselineFamily
    implementation: str = Field(min_length=1, max_length=256)
    seed: int = Field(ge=0, le=2**63 - 1)
    repetition: int = Field(ge=0, le=MAX_REPETITIONS)
    warmup: bool
    randomized_order: int = Field(ge=0, le=MAX_SAMPLES)
    workload_evidence_class: Literal["SYNTHETIC"] = "SYNTHETIC"
    timing_measurement_class: Literal["HARDWARE_BACKED_REAL"] = "HARDWARE_BACKED_REAL"
    monotonic_start_ns: int = Field(ge=0)
    duration_ns: int = Field(gt=0)
    cpu_time_ns: int = Field(gt=0)
    input_bytes: int = Field(ge=0)
    logical_bytes_processed: int = Field(ge=0)
    destination_logical_bytes: int = Field(ge=0)
    source_read_bytes: int = Field(ge=0)
    chunk_size_bytes: int = Field(ge=0)
    requested_concurrency: int = Field(ge=1, le=32)
    observed_worker_tasks: int = Field(ge=1, le=4096)
    fanout: int = Field(ge=1, le=128)
    operation_count: int = Field(ge=1)
    operation_chain: tuple[str, ...] = Field(min_length=1, max_length=16)
    byte_accounting: Literal["lower_bound", "exact_logical_delivery"]
    component_temporary_bound_bytes: int | None = Field(default=None, ge=0)
    actual_allocator_peak_bytes: None = None
    allocation_count: None = None
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_equivalent: Literal[True] = True
    network_hardware_measured: Literal[False] = False
    gpu_hardware_measured: Literal[False] = False
    notes: tuple[str, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def digest_matches(self) -> Self:
        if self.expected_sha256 != self.result_sha256:
            raise ValueError("a successful software baseline sample must preserve state identity")
        return self


class SoftwareBaselineSummary(SoftwareBaselineModel):
    case_id: str
    family: BaselineFamily
    implementation: str
    warmup_count: int = Field(ge=0)
    sample_count: int = Field(ge=1)
    median_duration_ns: float = Field(gt=0.0, allow_inf_nan=False)
    p95_duration_ns: float = Field(gt=0.0, allow_inf_nan=False)
    p99_duration_ns: float = Field(gt=0.0, allow_inf_nan=False)
    median_cpu_time_ns: float = Field(gt=0.0, allow_inf_nan=False)
    median_logical_throughput_bytes_per_second: float = Field(ge=0.0, allow_inf_nan=False)
    relative_median_absolute_deviation: float = Field(ge=0.0, allow_inf_nan=False)
    samples_removed: Literal[0] = 0
    semantic_equivalence_passed: Literal[True] = True


class SelectedSoftwareBaseline(SoftwareBaselineModel):
    selection_id: str
    case_id: str
    family: BaselineFamily
    criterion: Literal["lowest observed median duration among semantically equal candidates"]
    candidate_case_ids: tuple[str, ...] = Field(min_length=1)
    scope: str = Field(min_length=1, max_length=4096)


class SoftwareBaselineReport(SoftwareBaselineModel):
    schema_version: Literal["sloforge.branchfabric.software-baselines/v1"]
    config: SoftwareBaselineConfig
    host: str
    python_version: str
    workload_evidence_class: Literal["SYNTHETIC"]
    timing_measurement_class: Literal["HARDWARE_BACKED_REAL"]
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transform_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transform_expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    equivalence_proofs: tuple[EquivalenceProof, ...] = Field(min_length=2)
    raw_samples: tuple[SoftwareBaselineSample, ...] = Field(min_length=1, max_length=MAX_SAMPLES)
    summaries: tuple[SoftwareBaselineSummary, ...] = Field(min_length=1)
    selected_baselines: tuple[SelectedSoftwareBaseline, ...] = Field(min_length=1)
    no_gpu_measurements: Literal[True]
    no_network_hardware_measurements: Literal[True]
    raw_samples_preserved: Literal[True]
    outliers_removed: Literal[False]
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def samples_are_complete(self) -> Self:
        orders = tuple(sample.randomized_order for sample in self.raw_samples)
        if orders != tuple(range(len(self.raw_samples))):
            raise ValueError("raw sample run order must be dense and preserved")
        case_count = (
            2
            + 1
            + len(self.config.chunk_sizes)
            + len(self.config.chunk_sizes) * len(self.config.transfer_concurrency)
            + 2 * len(self.config.fanouts)
        )
        expected_samples = case_count * (
            self.config.warmup_repetitions + self.config.measurement_repetitions
        )
        if len(self.raw_samples) != expected_samples:
            raise ValueError("raw sample corpus does not cover the configured sweep")
        sample_ids = tuple(sample.sample_id for sample in self.raw_samples)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("software baseline sample IDs must be unique")
        if any(sample.seed != self.config.seed for sample in self.raw_samples):
            raise ValueError("raw sample seed differs from the study configuration")
        if any(not sample.semantic_equivalent for sample in self.raw_samples):
            raise ValueError("failed semantic samples cannot enter a successful report")
        raw_cases = {sample.case_id for sample in self.raw_samples}
        summary_cases = {summary.case_id for summary in self.summaries}
        if raw_cases != summary_cases:
            raise ValueError("summary cases must exactly cover raw sample cases")
        if any(selection.case_id not in raw_cases for selection in self.selected_baselines):
            raise ValueError("selected software baseline is absent from raw samples")
        return self


@dataclass(frozen=True, slots=True)
class _Case:
    family: BaselineFamily
    implementation: str
    chunk_size: int = 0
    concurrency: int = 1
    fanout: int = 1

    @property
    def case_id(self) -> str:
        fields = [self.family.value, self.implementation]
        if self.chunk_size:
            fields.append(f"chunk_{self.chunk_size}")
        if self.concurrency != 1:
            fields.append(f"concurrency_{self.concurrency}")
        if self.fanout != 1:
            fields.append(f"fanout_{self.fanout}")
        return ".".join(fields)


@dataclass(frozen=True, slots=True)
class _Measured:
    started_ns: int
    duration_ns: int
    cpu_time_ns: int
    result: object


def _measure(function: Callable[[], _T], *, timeout_seconds: float) -> _Measured:
    started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    result = function()
    cpu_time = max(1, time.process_time_ns() - cpu_started)
    duration = max(1, time.perf_counter_ns() - started)
    if duration > timeout_seconds * 1_000_000_000:
        raise TimeoutError(f"software baseline sample exceeded {timeout_seconds} seconds")
    return _Measured(started_ns=started, duration_ns=duration, cpu_time_ns=cpu_time, result=result)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _copy_bytes(payload: bytes) -> bytes:
    copied = memoryview(payload).tobytes()
    if copied is payload:
        raise RuntimeError("software movement baseline did not allocate a destination copy")
    return copied


def _captured_semantic_hash(captured: CapturedState) -> str:
    captured.verify()
    digest = hashlib.sha256()
    digest.update(captured.logical.continuation_hash.encode("ascii"))
    digest.update(captured.runtime.runtime_name.encode("utf-8"))
    digest.update(captured.runtime.build_hash.encode("ascii"))
    digest.update(captured.layout.kind.value.encode("ascii"))
    digest.update(str(captured.layout.tensor_parallel_degree).encode("ascii"))
    digest.update(str(captured.layout.page_size_tokens).encode("ascii"))
    for segment in sorted(captured.segments, key=lambda item: item.descriptor.segment_id):
        digest.update(segment.descriptor.segment_id.encode("utf-8"))
        digest.update(segment.descriptor.checksum.encode("ascii"))
        digest.update(segment.payload)
    for page in sorted(
        captured.page_table,
        key=lambda item: (item.layer, item.shard_rank, item.physical_page_id),
    ):
        digest.update(page.logical_state_id.encode("utf-8"))
        for value in (
            page.layer,
            page.shard_rank,
            page.logical_token_start,
            page.logical_token_end,
            page.physical_page_id,
            page.page_version,
            page.dirty_epoch,
            page.owner_epoch,
            page.copy_on_write_refs,
        ):
            digest.update(str(value).encode("ascii"))
            digest.update(b"\x00")
        for segment_id in page.segment_ids:
            digest.update(segment_id.encode("utf-8"))
            digest.update(b"\x00")
    return digest.hexdigest()


def _transform_fixture(
    config: SoftwareBaselineConfig,
) -> tuple[CapturedState, ReferenceHeadMajorAdapter, str]:
    source = ReferenceTokenMajorAdapter(page_size_tokens=3)
    session_id = f"software-baseline-{config.seed}"
    source.create_session(
        session_id=session_id,
        request_id="branchfabric-software-baseline",
        tenant_id="tenant-characterization",
        input_token_ids=(2, 3, 5, 7, 11, 13, 17, 19),
        seed=config.seed,
    )
    for event in source.stream_tokens(session_id, count=config.transform_output_tokens):
        source.acknowledge_gateway(
            session_id,
            token_index=event.token_index,
            owner_epoch=event.owner_epoch,
        )
    captured = source.capture_consistent(session_id)
    source_attention = captured_attention_state(captured)
    destination = ReferenceHeadMajorAdapter(page_size_tokens=5)
    destination_layout = KVLayout(
        kind=KVLayoutKind.HEAD_MAJOR_PACKED,
        tensor_parallel_degree=destination.config.layout.tensor_parallel_degree,
        page_size_tokens=destination.config.layout.page_size_tokens,
        layer_count=source_attention.layout.layer_count,
        token_count=source_attention.layout.token_count,
        kv_head_count=source_attention.layout.kv_head_count,
        head_dim=source_attention.layout.head_dim,
        dtype="int32",
    )
    return captured, destination, destination_layout.fingerprint()


def _run_fused_transform(
    captured: CapturedState,
    destination: ReferenceHeadMajorAdapter,
    *,
    maximum_temporary_bytes: int,
) -> tuple[str, int, int]:
    converted, evidence = direct_convert_capture(
        captured,
        destination=destination,
        maximum_temporary_bytes=maximum_temporary_bytes,
    )
    if not evidence.canonical_attention_match:
        raise RuntimeError("Continuum direct conversion failed its canonical verification")
    return (
        _captured_semantic_hash(converted),
        sum(segment.descriptor.payload_bytes for segment in converted.segments),
        evidence.maximum_temporary_bytes,
    )


def _run_staged_transform(
    captured: CapturedState, destination: ReferenceHeadMajorAdapter
) -> tuple[str, int]:
    state = decode_captured(
        captured,
        destination_config=destination.config,
        destination_session_id=captured.logical.session_id,
    )
    encoded = encode_state(state, destination.config)
    converted = CapturedState(
        handle=replace(captured.handle, segment_count=len(encoded.segments)),
        runtime=destination.identity,
        layout=destination.config.layout,
        logical=encoded.logical,
        segments=encoded.segments,
        page_table=encoded.page_table,
    )
    converted.verify()
    return (
        _captured_semantic_hash(converted),
        sum(segment.descriptor.payload_bytes for segment in converted.segments),
    )


def _hash_payload(payload: bytes, chunk_size: int) -> str:
    if chunk_size == 0:
        return hashlib.sha256(payload).hexdigest()
    digest = hashlib.sha256()
    view = memoryview(payload)
    for offset in range(0, len(payload), chunk_size):
        digest.update(view[offset : offset + chunk_size])
    return digest.hexdigest()


def _partition_references(
    references: tuple[ChunkRef, ...], concurrency: int
) -> tuple[tuple[ChunkRef, ...], ...]:
    worker_count = min(concurrency, len(references))
    groups: list[list[ChunkRef]] = [[] for _ in range(worker_count)]
    for index, reference in enumerate(references):
        groups[index % worker_count].append(reference)
    return tuple(tuple(group) for group in groups)


def _prepare_transfer_source(
    payload: bytes, chunk_size: int
) -> tuple[MemoryContentStore, tuple[ChunkRef, ...]]:
    source = MemoryContentStore()
    references = tuple(
        source.put("tenant-characterization", payload[offset : offset + chunk_size])
        for offset in range(0, len(payload), chunk_size)
    )
    if len({reference.digest for reference in references}) != len(references):
        raise RuntimeError("controlled transfer fixture unexpectedly produced duplicate chunks")
    return source, references


def _transfer_in_process(
    *,
    source: MemoryContentStore,
    references: tuple[ChunkRef, ...],
    concurrency: int,
    pool: ThreadPoolExecutor | None,
    seed: int,
    timeout_seconds: float,
) -> tuple[str, int]:
    destination = MemoryContentStore()
    groups = _partition_references(references, concurrency)

    def transfer(group: tuple[ChunkRef, ...], worker: int) -> None:
        receipt = InProcessTransport().transfer(
            source=source,
            destination=destination,
            tenant_id="tenant-characterization",
            references=group,
            deadline_us=max(1, int(timeout_seconds * 1_000_000)),
            seed=seed + worker,
        )
        if receipt.acknowledged_chunks != len(group) or receipt.retransmissions:
            raise RuntimeError("in-process transfer did not acknowledge the exact input")

    if pool is None:
        transfer(groups[0], 0)
    else:
        deadline = time.monotonic() + timeout_seconds
        futures = tuple(pool.submit(transfer, group, worker) for worker, group in enumerate(groups))
        for future in futures:
            future.result(timeout=max(0.001, deadline - time.monotonic()))
    digest = hashlib.sha256()
    for reference in references:
        digest.update(destination.read("tenant-characterization", reference))
    return digest.hexdigest(), len(groups)


def _await_copies(
    pool: ThreadPoolExecutor,
    sources: Sequence[bytes],
    *,
    timeout_seconds: float,
) -> list[bytes]:
    deadline = time.monotonic() + timeout_seconds
    futures: tuple[Future[bytes], ...] = tuple(
        pool.submit(_copy_bytes, source) for source in sources
    )
    return [future.result(timeout=max(0.001, deadline - time.monotonic())) for future in futures]


def _repeated_unicast(
    payload: bytes,
    fanout: int,
    *,
    pool: ThreadPoolExecutor,
    timeout_seconds: float,
) -> tuple[list[bytes], int]:
    return _await_copies(pool, [payload] * fanout, timeout_seconds=timeout_seconds), fanout


def _tree_fanout(
    payload: bytes,
    fanout: int,
    *,
    pool: ThreadPoolExecutor,
    timeout_seconds: float,
) -> tuple[list[bytes], int]:
    destinations: list[bytes] = []
    frontier = [payload]
    maximum_tasks = 1
    while len(destinations) < fanout:
        needed = fanout - len(destinations)
        sources = [parent for parent in frontier for _ in range(2)][:needed]
        children = _await_copies(pool, sources, timeout_seconds=timeout_seconds)
        maximum_tasks = max(maximum_tasks, len(children))
        destinations.extend(children)
        frontier = children
    return destinations, maximum_tasks


def _cases(config: SoftwareBaselineConfig) -> tuple[_Case, ...]:
    cases = [
        _Case(BaselineFamily.TRANSFORM, "continuum_direct_convert_capture"),
        _Case(BaselineFamily.TRANSFORM, "trusted_canonical_staged"),
        _Case(BaselineFamily.HASH, "hashlib_sha256_whole"),
    ]
    cases.extend(
        _Case(BaselineFamily.HASH, "hashlib_sha256_chunked", chunk_size=size)
        for size in config.chunk_sizes
    )
    cases.extend(
        _Case(
            BaselineFamily.IN_PROCESS_TRANSFER,
            "continuum_in_process",
            chunk_size=size,
            concurrency=concurrency,
        )
        for size in config.chunk_sizes
        for concurrency in config.transfer_concurrency
    )
    cases.extend(
        _Case(BaselineFamily.SOFTWARE_FANOUT, implementation, fanout=fanout)
        for fanout in config.fanouts
        for implementation in ("repeated_unicast", "binary_tree")
    )
    return tuple(cases)


def _summaries(samples: tuple[SoftwareBaselineSample, ...]) -> tuple[SoftwareBaselineSummary, ...]:
    grouped: dict[str, list[SoftwareBaselineSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.case_id].append(sample)
    summaries: list[SoftwareBaselineSummary] = []
    for case_id in sorted(grouped):
        group = grouped[case_id]
        measured = [sample for sample in group if not sample.warmup]
        durations = [float(sample.duration_ns) for sample in measured]
        cpu_times = [float(sample.cpu_time_ns) for sample in measured]
        center = statistics.median(durations)
        mad = statistics.median(abs(value - center) for value in durations)
        logical_bytes = statistics.median(
            float(sample.logical_bytes_processed) for sample in measured
        )
        first = group[0]
        summaries.append(
            SoftwareBaselineSummary(
                case_id=case_id,
                family=first.family,
                implementation=first.implementation,
                warmup_count=sum(sample.warmup for sample in group),
                sample_count=len(measured),
                median_duration_ns=center,
                p95_duration_ns=_percentile(durations, 0.95),
                p99_duration_ns=_percentile(durations, 0.99),
                median_cpu_time_ns=statistics.median(cpu_times),
                median_logical_throughput_bytes_per_second=(
                    logical_bytes * 1_000_000_000.0 / center if logical_bytes else 0.0
                ),
                relative_median_absolute_deviation=mad / center,
                semantic_equivalence_passed=True,
            )
        )
    return tuple(summaries)


def _selected(
    summaries: tuple[SoftwareBaselineSummary, ...],
) -> tuple[SelectedSoftwareBaseline, ...]:
    by_family: dict[BaselineFamily, list[SoftwareBaselineSummary]] = defaultdict(list)
    for summary in summaries:
        by_family[summary.family].append(summary)
    selected: list[SelectedSoftwareBaseline] = []
    for family in (
        BaselineFamily.TRANSFORM,
        BaselineFamily.HASH,
        BaselineFamily.IN_PROCESS_TRANSFER,
    ):
        candidates = by_family[family]
        winner = min(candidates, key=lambda item: (item.median_duration_ns, item.case_id))
        scope = {
            BaselineFamily.TRANSFORM: (
                "controlled exact reference-capture conversion; current direct path includes "
                "independent canonical verification and is not an isolated kernel comparison"
            ),
            BaselineFamily.HASH: "host hashlib SHA-256 over one identical immutable byte payload",
            BaselineFamily.IN_PROCESS_TRANSFER: (
                "thread-safe host MemoryContentStore and Continuum InProcessTransport; no NIC"
            ),
        }[family]
        selected.append(
            SelectedSoftwareBaseline(
                selection_id=f"best.{family.value}",
                case_id=winner.case_id,
                family=family,
                criterion="lowest observed median duration among semantically equal candidates",
                candidate_case_ids=tuple(sorted(item.case_id for item in candidates)),
                scope=scope,
            )
        )
    fanout_candidates = by_family[BaselineFamily.SOFTWARE_FANOUT]
    for fanout in FANOUTS:
        suffix = f"fanout_{fanout}"
        candidates = [item for item in fanout_candidates if item.case_id.endswith(suffix)]
        winner = min(candidates, key=lambda item: (item.median_duration_ns, item.case_id))
        selected.append(
            SelectedSoftwareBaseline(
                selection_id=f"best.software_fanout.{fanout}",
                case_id=winner.case_id,
                family=BaselineFamily.SOFTWARE_FANOUT,
                criterion="lowest observed median duration among semantically equal candidates",
                candidate_case_ids=tuple(sorted(item.case_id for item in candidates)),
                scope=(
                    "host-memory copy scheduling only; tree root-read accounting is analytical "
                    "and no network multicast or link throughput is measured"
                ),
            )
        )
    return tuple(selected)


def run_software_baseline_study(config: SoftwareBaselineConfig) -> SoftwareBaselineReport:
    """Run bounded randomized CPU comparisons and preserve every raw sample."""

    generator = random.Random(config.seed)
    payload = generator.randbytes(config.payload_bytes)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    captured, destination, _destination_fingerprint = _transform_fixture(config)
    transform_source_sha256 = _captured_semantic_hash(captured)

    fused_preflight = _run_fused_transform(
        captured,
        destination,
        maximum_temporary_bytes=config.transform_maximum_temporary_bytes,
    )
    staged_preflight = _run_staged_transform(captured, destination)
    if fused_preflight[0] != staged_preflight[0]:
        raise RuntimeError("fused and staged Continuum conversion outputs are not equivalent")
    transform_expected_sha256 = fused_preflight[0]
    hash_observations = tuple(
        _hash_payload(payload, chunk_size) for chunk_size in (0, *config.chunk_sizes)
    )
    if any(observed != payload_sha256 for observed in hash_observations):
        raise RuntimeError("chunked hashlib baseline differs from whole-buffer SHA-256")

    equivalence_proofs = (
        EquivalenceProof(
            proof_id="continuum-transform-exact-equivalence",
            compared_implementations=(
                "direct_convert_capture",
                "decode_captured_then_encode_state",
            ),
            expected_sha256=transform_expected_sha256,
            observed_sha256=(fused_preflight[0], staged_preflight[0]),
            exact_byte_or_semantic_equality=True,
            timing_sample=False,
            method=(
                "verified complete captured-state logical identity, destination layout, ordered "
                "segment identifiers, segment checksums, payload bytes, and page-table metadata"
            ),
        ),
        EquivalenceProof(
            proof_id="hashlib-whole-chunked-byte-equivalence",
            compared_implementations=(
                "hashlib.sha256(bytes)",
                *(f"hashlib.sha256(memoryview chunks={size})" for size in config.chunk_sizes),
            ),
            expected_sha256=payload_sha256,
            observed_sha256=hash_observations,
            exact_byte_or_semantic_equality=True,
            timing_sample=False,
            method="all implementations hash the exact same immutable bytes and match SHA-256",
        ),
    )

    transfer_sources = {
        size: _prepare_transfer_source(payload, size) for size in config.chunk_sizes
    }
    transfer_pools = {
        concurrency: ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix=f"branchfabric-transfer-{concurrency}",
        )
        for concurrency in config.transfer_concurrency
        if concurrency > 1
    }
    fanout_pool = ThreadPoolExecutor(
        max_workers=config.fanout_max_workers,
        thread_name_prefix="branchfabric-fanout",
    )
    cases = _cases(config)
    samples: list[SoftwareBaselineSample] = []
    try:
        phases = tuple((True, index) for index in range(config.warmup_repetitions)) + tuple(
            (False, index) for index in range(config.measurement_repetitions)
        )
        for warmup, repetition in phases:
            ordered = list(cases)
            phase_seed = config.seed ^ ((repetition + 1) << 17) ^ (0xA5A5 if warmup else 0x5A5A)
            random.Random(phase_seed).shuffle(ordered)
            for case in ordered:
                result_sha: str
                destination_bytes: int
                component_bound: int | None
                notes: tuple[str, ...]
                input_bytes: int
                logical_bytes: int
                source_reads: int
                worker_tasks: int
                fanout: int
                concurrency: int
                chunk_size: int
                operation_count: int
                operation_chain: tuple[str, ...]
                byte_accounting: Literal["lower_bound", "exact_logical_delivery"]
                if case.family is BaselineFamily.TRANSFORM:
                    if case.implementation == "continuum_direct_convert_capture":
                        measured = _measure(
                            partial(
                                _run_fused_transform,
                                captured,
                                destination,
                                maximum_temporary_bytes=config.transform_maximum_temporary_bytes,
                            ),
                            timeout_seconds=config.sample_timeout_seconds,
                        )
                        result_sha, destination_bytes, component_bound = cast(
                            tuple[str, int, int], measured.result
                        )
                        notes = (
                            "actual Continuum direct_convert_capture path",
                            "includes trusted canonical verification",
                            "component bound is direct-converter scratch, not process peak",
                        )
                        operation_chain = (
                            "RESHARD",
                            "REPACK",
                            "CHECKSUM",
                            "CANONICAL_VERIFY",
                        )
                    else:
                        measured = _measure(
                            partial(_run_staged_transform, captured, destination),
                            timeout_seconds=config.sample_timeout_seconds,
                        )
                        result_sha, destination_bytes = cast(tuple[str, int], measured.result)
                        component_bound = None
                        notes = (
                            "trusted decode_captured then encode_state staged comparator",
                            "canonical temporary allocation peak is unavailable",
                        )
                        operation_chain = (
                            "DECODE_CANONICAL",
                            "ENCODE_PACKED",
                            "CHECKSUM",
                        )
                    input_bytes = sum(
                        segment.descriptor.payload_bytes for segment in captured.segments
                    )
                    logical_bytes = input_bytes + int(destination_bytes)
                    source_reads = input_bytes
                    worker_tasks = 1
                    fanout = 1
                    concurrency = 1
                    chunk_size = 0
                    operation_count = 3
                    byte_accounting = "lower_bound"
                elif case.family is BaselineFamily.HASH:
                    measured = _measure(
                        partial(_hash_payload, payload, case.chunk_size),
                        timeout_seconds=config.sample_timeout_seconds,
                    )
                    result_sha = cast(str, measured.result)
                    destination_bytes = 32
                    component_bound = case.chunk_size or len(payload)
                    input_bytes = len(payload)
                    logical_bytes = len(payload)
                    source_reads = len(payload)
                    worker_tasks = 1
                    fanout = 1
                    concurrency = 1
                    chunk_size = case.chunk_size
                    operation_count = max(
                        1, math.ceil(len(payload) / (case.chunk_size or len(payload)))
                    )
                    notes = (
                        "CPython hashlib backed SHA-256",
                        "chunked path passes memoryview slices without explicit payload copies",
                    )
                    operation_chain = ("CHECKSUM",)
                    byte_accounting = "exact_logical_delivery"
                elif case.family is BaselineFamily.IN_PROCESS_TRANSFER:
                    source, references = transfer_sources[case.chunk_size]
                    measured = _measure(
                        partial(
                            _transfer_in_process,
                            source=source,
                            references=references,
                            concurrency=case.concurrency,
                            pool=transfer_pools.get(case.concurrency),
                            seed=config.seed + repetition,
                            timeout_seconds=config.sample_timeout_seconds,
                        ),
                        timeout_seconds=config.sample_timeout_seconds,
                    )
                    result_sha, worker_tasks = cast(tuple[str, int], measured.result)
                    destination_bytes = len(payload)
                    component_bound = case.chunk_size
                    input_bytes = len(payload)
                    logical_bytes = len(payload)
                    source_reads = len(payload)
                    fanout = 1
                    concurrency = case.concurrency
                    chunk_size = case.chunk_size
                    operation_count = len(references)
                    notes = (
                        "Continuum MemoryContentStore and InProcessTransport",
                        "shared destination store retains its real locking and hashing",
                        "requested concurrency can exceed chunk count; observed_worker_tasks is exact",
                    )
                    operation_chain = ("READ", "HASH_VERIFY", "WRITE", "ACK")
                    byte_accounting = "exact_logical_delivery"
                else:
                    operation = (
                        _repeated_unicast
                        if case.implementation == "repeated_unicast"
                        else _tree_fanout
                    )
                    measured = _measure(
                        partial(
                            operation,
                            payload,
                            case.fanout,
                            pool=fanout_pool,
                            timeout_seconds=config.sample_timeout_seconds,
                        ),
                        timeout_seconds=config.sample_timeout_seconds,
                    )
                    destinations, worker_tasks = cast(tuple[list[bytes], int], measured.result)
                    if len(destinations) != case.fanout or any(
                        hashlib.sha256(item).hexdigest() != payload_sha256 for item in destinations
                    ):
                        raise RuntimeError("software fanout changed immutable payload bytes")
                    result_sha = payload_sha256
                    destination_bytes = len(payload) * case.fanout
                    component_bound = len(payload)
                    input_bytes = len(payload)
                    logical_bytes = destination_bytes
                    source_reads = (
                        destination_bytes
                        if case.implementation == "repeated_unicast"
                        else min(2, case.fanout) * len(payload)
                    )
                    fanout = case.fanout
                    concurrency = min(config.fanout_max_workers, case.fanout)
                    chunk_size = len(payload)
                    operation_count = case.fanout
                    notes = (
                        "host-memory copies scheduled through one reused bounded thread pool",
                        "tree source-read bytes count root reads only; aggregate host copies are unchanged",
                        "no network or multicast hardware was exercised",
                    )
                    operation_chain = ("READ", "COPY")
                    byte_accounting = "exact_logical_delivery"
                expected = (
                    transform_expected_sha256
                    if case.family is BaselineFamily.TRANSFORM
                    else payload_sha256
                )
                order = len(samples)
                samples.append(
                    SoftwareBaselineSample(
                        sample_id=f"{case.case_id}.{'warmup' if warmup else 'measurement'}.{repetition}",
                        case_id=case.case_id,
                        family=case.family,
                        implementation=case.implementation,
                        seed=config.seed,
                        repetition=repetition,
                        warmup=warmup,
                        randomized_order=order,
                        monotonic_start_ns=measured.started_ns,
                        duration_ns=measured.duration_ns,
                        cpu_time_ns=measured.cpu_time_ns,
                        input_bytes=input_bytes,
                        logical_bytes_processed=logical_bytes,
                        destination_logical_bytes=int(destination_bytes),
                        source_read_bytes=source_reads,
                        chunk_size_bytes=chunk_size,
                        requested_concurrency=concurrency,
                        observed_worker_tasks=int(worker_tasks),
                        fanout=fanout,
                        operation_count=operation_count,
                        operation_chain=operation_chain,
                        byte_accounting=byte_accounting,
                        component_temporary_bound_bytes=(
                            int(component_bound) if component_bound is not None else None
                        ),
                        expected_sha256=expected,
                        result_sha256=str(result_sha),
                        semantic_equivalent=True,
                        notes=notes,
                    )
                )
    finally:
        for pool in transfer_pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
        fanout_pool.shutdown(wait=True, cancel_futures=True)

    raw_samples = tuple(samples)
    summaries = _summaries(raw_samples)
    return SoftwareBaselineReport(
        schema_version="sloforge.branchfabric.software-baselines/v1",
        config=config,
        host=platform.node() or "unknown",
        python_version=platform.python_version(),
        workload_evidence_class="SYNTHETIC",
        timing_measurement_class="HARDWARE_BACKED_REAL",
        payload_sha256=payload_sha256,
        transform_source_sha256=transform_source_sha256,
        transform_expected_sha256=transform_expected_sha256,
        equivalence_proofs=equivalence_proofs,
        raw_samples=raw_samples,
        summaries=summaries,
        selected_baselines=_selected(summaries),
        no_gpu_measurements=True,
        no_network_hardware_measurements=True,
        raw_samples_preserved=True,
        outliers_removed=False,
        limitations=(
            "All fixture shapes and fanouts are controlled synthetic sweeps, not observed distributions.",
            "Host monotonic and process CPU timings include Python, NumPy, hashing, locks, and scheduling.",
            "The current direct capture converter includes independent canonical verification; its timing is not an isolated fused kernel.",
            "Actual allocator counts and peak bytes are unavailable and remain null; logical allocations and declared component bounds are reported.",
            "Threaded in-process transfer exercises no socket, NIC, PCIe, RDMA, or network path.",
            "Software tree fanout measures host copy scheduling only and cannot establish network multicast benefit.",
            "No GPU, HBM, copy engine, or GPU interference measurement is claimed.",
        ),
    )


def write_software_baseline_report(
    report: SoftwareBaselineReport,
    output: Path,
    *,
    replace: bool = False,
) -> str:
    """Atomically write a bounded report and return its exact SHA-256."""

    payload = (report.model_dump_json(indent=2) + "\n").encode("utf-8")
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError(f"software baseline report exceeds {MAX_OUTPUT_BYTES} bytes")
    if output.exists():
        if not output.is_file() or output.is_symlink():
            raise FileExistsError("software baseline output must be a regular file")
        if not replace:
            raise FileExistsError("software baseline output exists; pass replace=True")
        try:
            prior = SoftwareBaselineReport.model_validate_json(output.read_bytes(), strict=True)
        except ValueError as error:
            raise ValueError(
                "refusing to replace an unrecognized software baseline report"
            ) from error
        if prior.schema_version != report.schema_version:
            raise ValueError("refusing to replace a different software baseline schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "FANOUTS",
    "BaselineFamily",
    "EquivalenceProof",
    "SelectedSoftwareBaseline",
    "SoftwareBaselineConfig",
    "SoftwareBaselineReport",
    "SoftwareBaselineSample",
    "SoftwareBaselineSummary",
    "run_software_baseline_study",
    "write_software_baseline_report",
]
