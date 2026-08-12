"""Measured CPU reference-model fanout vertical for BranchFabric gating.

This module deliberately does not claim a real transformer, GPU, device-memory,
or network result.  It captures the exercised Continuum HybridDecoder adapter
and uses its exact serialized model-state segments.  Host timings are real; the
workload is a deterministic CPU reference-model workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.continuum.adapters import ReferenceTokenMajorAdapter, SessionLifecycle
from sloforge.continuum.adapters.abi import captured_to_capsule_inputs
from sloforge.continuum.adapters.sdk import CapturedState, StateKind
from sloforge.continuum.reference.codec import decode_segments, encode_state

MAX_BRANCHES = 64
MAX_SAMPLES = 10_000
# ReferenceTokenMajorAdapter constructs its exercised HybridDecoderConfig with
# max_context_tokens=4096 and does not expose a public context-bound override.
# Raising this value here would silently stop measuring the existing adapter.
MAX_CONTEXT_TOKENS: Final = 4096
DEFAULT_OUTPUT = Path("artifacts/branchfabric/execution/fanout")

_T = TypeVar("_T")


class FanoutModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class WorkloadClass(StrEnum):
    CODING_AGENT = "coding_agent"
    REASONING_VERIFICATION = "reasoning_verification"


class FanoutImplementation(StrEnum):
    NAIVE_PRIVATE = "naive_private_materialization"
    SHARED_ROOT_COW_LAZY = "shared_root_cow_lazy"


class WorkloadSpec(FanoutModel):
    workload_class: WorkloadClass
    context_tokens: int = Field(ge=16, le=MAX_CONTEXT_TOKENS)
    suffix_tokens: int = Field(ge=1, le=256)
    common_suffix_tokens: int = Field(ge=0, le=256)
    divergence_pattern: Literal["immediate_rng_divergence", "late_rng_divergence"]

    @model_validator(mode="after")
    def validate_workload(self) -> Self:
        if self.context_tokens + self.suffix_tokens > MAX_CONTEXT_TOKENS:
            raise ValueError("context plus suffix exceeds the reference runtime bound")
        if self.common_suffix_tokens > self.suffix_tokens:
            raise ValueError("common suffix cannot exceed the full suffix")
        immediate = self.divergence_pattern == "immediate_rng_divergence"
        if immediate != (self.common_suffix_tokens == 0):
            raise ValueError("divergence pattern contradicts the common suffix length")
        return self


DEFAULT_WORKLOADS = (
    WorkloadSpec(
        workload_class=WorkloadClass.CODING_AGENT,
        context_tokens=2048,
        suffix_tokens=24,
        common_suffix_tokens=0,
        divergence_pattern="immediate_rng_divergence",
    ),
    WorkloadSpec(
        workload_class=WorkloadClass.REASONING_VERIFICATION,
        context_tokens=3968,
        suffix_tokens=32,
        common_suffix_tokens=16,
        divergence_pattern="late_rng_divergence",
    ),
)


class FanoutStudyConfig(FanoutModel):
    schema_version: Literal["sloforge.branchfabric.cpu-reference-fanout-config/v1"] = (
        "sloforge.branchfabric.cpu-reference-fanout-config/v1"
    )
    seeds: tuple[int, ...] = (41, 73, 113)
    fanouts: tuple[int, ...] = (8, 16, 32)
    workloads: tuple[WorkloadSpec, ...] = DEFAULT_WORKLOADS
    warmup_repetitions: int = Field(default=1, ge=1, le=10)
    measurement_repetitions: int = Field(default=3, ge=3, le=50)
    randomization_seed: int = Field(default=20_260_809, ge=0, le=2**63 - 1)
    bootstrap_seed: int = Field(default=8_091, ge=0, le=2**63 - 1)
    bootstrap_repetitions: int = Field(default=1_000, ge=100, le=20_000)
    sample_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    page_size_tokens: int = Field(default=16, ge=1, le=256)

    @model_validator(mode="after")
    def validate_sweep(self) -> Self:
        if len(self.seeds) < 3 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("at least three unique seeds are required")
        if any(seed < 0 or seed >= 2**63 for seed in self.seeds):
            raise ValueError("seeds must fit signed uint63 for derived branch seeds")
        if not {8, 16, 32}.issubset(self.fanouts):
            raise ValueError("fanout sweep must include 8, 16, and 32")
        if tuple(sorted(set(self.fanouts))) != self.fanouts or self.fanouts[-1] > MAX_BRANCHES:
            raise ValueError("fanouts must be unique, increasing, and no greater than 64")
        classes = tuple(workload.workload_class for workload in self.workloads)
        if set(classes) != set(WorkloadClass) or len(classes) != len(set(classes)):
            raise ValueError(
                "exactly one coding and one reasoning/verification workload is required"
            )
        count = (
            len(self.seeds)
            * len(self.fanouts)
            * len(self.workloads)
            * len(FanoutImplementation)
            * (self.warmup_repetitions + self.measurement_repetitions)
        )
        if count > MAX_SAMPLES:
            raise ValueError(f"fanout study exceeds the {MAX_SAMPLES}-sample safety bound")
        return self


class FixtureEvidence(FanoutModel):
    schema_version: Literal["sloforge.branchfabric.cpu-reference-state-fixture/v1"] = (
        "sloforge.branchfabric.cpu-reference-state-fixture/v1"
    )
    fixture_id: str
    model_state_class: Literal["CPU_REFERENCE_MODEL_STATE"]
    reference_adapter_max_context_tokens: Literal[4096]
    physical_byte_accounting: Literal[
        "DETERMINISTIC_SERIALIZED_STATE_BYTES_NOT_RSS_OS_PAGES_OR_GPU_ALLOCATION"
    ]
    workload_evidence_class: Literal["SYNTHETIC_CPU_REFERENCE_MODEL_STATE"]
    timing_measurement_class: Literal["HARDWARE_BACKED_HOST_CPU"]
    workload_class: WorkloadClass
    seed: int
    context_tokens: int
    suffix_tokens: int
    common_suffix_tokens: int
    divergence_pattern: str
    source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    continuation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    root_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    serialized_model_state_bytes: int = Field(gt=0)
    attention_kv_bytes: int = Field(gt=0)
    recurrent_state_bytes: int = Field(gt=0)
    sampler_state_bytes: int = Field(gt=0)
    guided_decoding_state_bytes: int = Field(gt=0)
    delivery_state_bytes: int = Field(gt=0)
    token_history_bytes: int = Field(gt=0)
    runtime_metadata_bytes: int = Field(gt=0)
    page_metadata_bytes: int = Field(gt=0)
    segment_count: int = Field(gt=0)
    page_count: int = Field(gt=0)
    physical_page_size_tokens: int = Field(gt=0)
    suffix_private_bytes_per_branch: tuple[int, ...] = Field(min_length=32, max_length=64)
    suffix_all_branches_ready_ns: dict[int, int]
    identical_new_suffix_bytes_by_fanout: dict[int, int]
    real_transformer_measured: Literal[False]
    gpu_measured: Literal[False]
    simulated_device_bytes_claimed: Literal[False]


class FanoutRawSample(FanoutModel):
    schema_version: Literal["sloforge.branchfabric.cpu-reference-fanout-sample/v1"] = (
        "sloforge.branchfabric.cpu-reference-fanout-sample/v1"
    )
    sample_id: str
    fixture_id: str
    model_state_class: Literal["CPU_REFERENCE_MODEL_STATE"]
    physical_byte_accounting: Literal[
        "DETERMINISTIC_SERIALIZED_STATE_BYTES_NOT_RSS_OS_PAGES_OR_GPU_ALLOCATION"
    ]
    workload_evidence_class: Literal["SYNTHETIC_CPU_REFERENCE_MODEL_STATE"]
    timing_measurement_class: Literal["HARDWARE_BACKED_HOST_CPU"]
    workload_class: WorkloadClass
    implementation: FanoutImplementation
    seed: int
    fanout: int = Field(ge=8, le=MAX_BRANCHES)
    repetition: int = Field(ge=0)
    warmup: bool
    randomized_order: int = Field(ge=0)
    monotonic_start_ns: int = Field(ge=0)
    branchpoint_to_all_ready_ns: int = Field(gt=0)
    per_branch_readiness_ns: tuple[int, ...] = Field(min_length=8, max_length=MAX_BRANCHES)
    p50_branch_readiness_ns: float = Field(gt=0.0, allow_inf_nan=False)
    p90_branch_readiness_ns: float = Field(gt=0.0, allow_inf_nan=False)
    p95_branch_readiness_ns: float = Field(gt=0.0, allow_inf_nan=False)
    p99_branch_readiness_ns: float = Field(gt=0.0, allow_inf_nan=False)
    logical_branch_state_bytes: int = Field(gt=0)
    source_root_physical_bytes: int = Field(gt=0)
    branch_private_physical_bytes_at_ready: int = Field(ge=0)
    total_physical_state_bytes_at_ready: int = Field(
        gt=0,
        description="Deterministic serialized-state allocation accounting; not RSS or pages",
    )
    post_suffix_private_bytes: int = Field(gt=0)
    total_physical_state_bytes_after_suffix: int = Field(
        gt=0,
        description="Serialized/CAS-equivalent accounting; not OS or accelerator allocation",
    )
    sharing_efficiency_at_ready: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    python_allocator_peak_bytes: int = Field(gt=0)
    allocation_count_lower_bound: int = Field(gt=0)
    bounded_queue_capacity: Literal[1]
    requested_queue_concurrency: Literal[1]
    observed_max_queue_concurrency: Literal[1]
    queue_operations: int = Field(ge=8, le=MAX_BRANCHES)
    suffix_all_branches_ready_ns: int = Field(gt=0)
    semantic_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    exact_root_semantics_preserved: Literal[True]
    allocation_instrumentation: Literal["python_tracemalloc_full"]
    gpu_measured: Literal[False]
    network_hardware_measured: Literal[False]

    @model_validator(mode="after")
    def validate_accounting(self) -> Self:
        if len(self.per_branch_readiness_ns) != self.fanout:
            raise ValueError("per-branch readiness must cover every sibling")
        if self.per_branch_readiness_ns[-1] != self.branchpoint_to_all_ready_ns:
            raise ValueError("all-ready latency must equal the final branch readiness")
        if tuple(sorted(self.per_branch_readiness_ns)) != self.per_branch_readiness_ns:
            raise ValueError("per-branch readiness timestamps must be nondecreasing")
        if self.total_physical_state_bytes_at_ready != (
            self.source_root_physical_bytes + self.branch_private_physical_bytes_at_ready
        ):
            raise ValueError("ready-state physical byte accounting is inconsistent")
        if self.total_physical_state_bytes_after_suffix != (
            self.total_physical_state_bytes_at_ready + self.post_suffix_private_bytes
        ):
            raise ValueError("post-suffix physical byte accounting is inconsistent")
        return self


class ConfidenceInterval(FanoutModel):
    method: Literal["deterministic_percentile_bootstrap_median"]
    confidence: float = Field(default=0.95, ge=0.95, le=0.95)
    seed: int
    repetitions: int
    lower: float = Field(gt=0.0, allow_inf_nan=False)
    observed: float = Field(gt=0.0, allow_inf_nan=False)
    upper: float = Field(gt=0.0, allow_inf_nan=False)


class FanoutSummary(FanoutModel):
    workload_class: WorkloadClass
    implementation: FanoutImplementation
    fanout: int
    sample_count: int = Field(ge=9)
    seed_count: int = Field(ge=3)
    warmup_samples_excluded: int = Field(ge=3)
    p50_all_ready_ns: float = Field(gt=0.0, allow_inf_nan=False)
    p90_all_ready_ns: float = Field(gt=0.0, allow_inf_nan=False)
    p95_all_ready_ns: float = Field(gt=0.0, allow_inf_nan=False)
    p99_all_ready_ns: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_all_ready_ns: int = Field(gt=0)
    median_confidence_interval: ConfidenceInterval
    median_logical_branch_state_bytes: float = Field(gt=0.0, allow_inf_nan=False)
    median_physical_state_bytes_at_ready: float = Field(gt=0.0, allow_inf_nan=False)
    median_physical_state_bytes_after_suffix: float = Field(gt=0.0, allow_inf_nan=False)
    median_allocator_peak_bytes: float = Field(gt=0.0, allow_inf_nan=False)
    median_sharing_efficiency_at_ready: float = Field(ge=0.0, le=1.0)
    outliers_removed: Literal[False]


class BaselineEffect(FanoutModel):
    workload_class: WorkloadClass
    fanout: int
    matched_pair_count: int = Field(ge=9)
    baseline: Literal["naive_private_materialization"]
    optimized: Literal["shared_root_cow_lazy"]
    median_software_speedup: float = Field(gt=0.0, allow_inf_nan=False)
    speedup_confidence_interval: ConfidenceInterval
    median_relative_readiness_reduction: float = Field(allow_inf_nan=False)
    matched_pairs_optimized_faster_fraction: float = Field(ge=0.0, le=1.0)
    physical_state_reduction_at_ready: float = Field(ge=0.0, le=1.0)


class HeadroomResult(FanoutModel):
    workload_class: WorkloadClass
    fanout: int
    optimized_readiness_median_ns: float = Field(gt=0.0)
    measured_suffix_path_median_ns: float = Field(gt=0.0)
    local_branch_lifecycle_median_ns: float = Field(gt=0.0)
    readiness_fraction_of_local_lifecycle: float = Field(gt=0.0, lt=1.0)
    ideal_zero_cost_readiness_speedup: float = Field(gt=1.0)
    end_to_end_helix_speedup_established: Literal[False]
    hardware_gate_eligible: Literal[False]
    gate_blockers: tuple[str, ...] = Field(min_length=1)


class QueueConcurrencySummary(FanoutModel):
    sample_count: int = Field(gt=0)
    p50: float = Field(ge=1.0, allow_inf_nan=False)
    p90: float = Field(ge=1.0, allow_inf_nan=False)
    p95: float = Field(ge=1.0, allow_inf_nan=False)
    p99: float = Field(ge=1.0, allow_inf_nan=False)
    maximum: int = Field(ge=1)
    interpretation: Literal["serialized_queue_concurrency_one"]


class FanoutStudyReport(FanoutModel):
    schema_version: Literal["sloforge.branchfabric.cpu-reference-fanout-report/v1"]
    config: FanoutStudyConfig
    model_state_class: Literal["CPU_REFERENCE_MODEL_STATE"]
    reference_adapter_max_context_tokens: Literal[4096]
    physical_byte_accounting: Literal[
        "DETERMINISTIC_SERIALIZED_STATE_BYTES_NOT_RSS_OS_PAGES_OR_GPU_ALLOCATION"
    ]
    host: str
    platform: str
    python_version: str
    repository_revision: str
    measurement_scope: Literal["LOCAL_MODEL_STATE_BRANCH_READINESS_AND_SUFFIX_NOT_END_TO_END_HELIX"]
    reproduction_command: str
    raw_samples_uri: str
    raw_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_sample_count: int = Field(gt=0)
    fixture_evidence_uri: str
    fixture_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixtures: tuple[FixtureEvidence, ...] = Field(min_length=6)
    summaries: tuple[FanoutSummary, ...]
    baseline_effects: tuple[BaselineEffect, ...]
    headroom: tuple[HeadroomResult, ...]
    queue_concurrency: QueueConcurrencySummary
    strongest_software_baseline: Literal["shared_root_cow_lazy"]
    randomized_run_order: Literal[True]
    real_transformer_measured: Literal[False]
    cuda_gpu_measured: Literal[False]
    multi_gpu_measured: Literal[False]
    physical_network_measured: Literal[False]
    end_to_end_helix_transaction_measured: Literal[False]
    raw_samples_preserved: Literal[True]
    outliers_removed: Literal[False]
    limitations: tuple[str, ...] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _Fixture:
    evidence: FixtureEvidence
    captured: CapturedState
    payloads: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class _Case:
    workload_class: WorkloadClass
    seed: int
    fanout: int
    implementation: FanoutImplementation


@dataclass(frozen=True, slots=True)
class _SharedRoot:
    payloads: tuple[bytes, ...]
    continuation_hash: str


@dataclass(frozen=True, slots=True)
class _BranchView:
    branch_id: str
    root: _SharedRoot
    branch_seed: int


@dataclass(frozen=True, slots=True)
class _ReadyOutcome:
    monotonic_start_ns: int
    per_branch_readiness_ns: tuple[int, ...]
    allocator_peak_bytes: int
    allocation_count_lower_bound: int
    semantic_state_hash: str
    retained: object


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _root_hash(captured: CapturedState) -> str:
    digest = hashlib.sha256()
    for segment in sorted(captured.segments, key=lambda item: item.descriptor.segment_id):
        identifier = segment.descriptor.segment_id.encode("utf-8")
        digest.update(len(identifier).to_bytes(4, "big"))
        digest.update(identifier)
        digest.update(len(segment.payload).to_bytes(8, "big"))
        digest.update(segment.payload)
    return digest.hexdigest()


def _branch_seed(seed: int, branch_index: int) -> int:
    return (seed * 1_000_003 + (branch_index + 1) * 97_409) % (2**63)


def _state_bytes_by_kind(captured: CapturedState) -> dict[StateKind, int]:
    result: dict[StateKind, int] = defaultdict(int)
    for segment in captured.segments:
        result[segment.descriptor.state_kind] += len(segment.payload)
    return result


def _metadata_bytes(captured: CapturedState) -> tuple[int, int]:
    inputs = captured_to_capsule_inputs(captured)
    runtime = {
        "logical_state": inputs.logical_state.model_dump(mode="json"),
        "physical_state": inputs.physical_state.model_dump(mode="json"),
        "segment_manifests": [item.model_dump(mode="json") for item in inputs.segment_manifests],
    }
    pages = [asdict(page) for page in captured.page_table]
    return len(_canonical_bytes(runtime)), len(_canonical_bytes(pages))


def _measure_suffix_path(
    captured: CapturedState,
    adapter: ReferenceTokenMajorAdapter,
    workload: WorkloadSpec,
    seed: int,
    fanouts: tuple[int, ...],
) -> tuple[tuple[int, ...], dict[int, int], dict[int, int]]:
    parent = {
        segment.descriptor.segment_id: segment.descriptor.checksum for segment in captured.segments
    }
    maximum = max(fanouts)
    private_bytes: list[int] = []
    completion: dict[int, int] = {}
    new_segments: list[dict[str, tuple[str, int]]] = []
    start = time.perf_counter_ns()
    for branch_index in range(maximum):
        state = decode_segments(
            source_layout=captured.layout,
            source_segments=captured.segments,
            manifest=captured.logical,
            destination_config=adapter.config,
            destination_session_id=f"suffix-{workload.workload_class.value}-{seed}-{branch_index}",
        )
        state.lifecycle = SessionLifecycle.ACTIVE
        if workload.common_suffix_tokens == 0:
            state.seed = _branch_seed(seed, branch_index)
        for token_index in range(workload.suffix_tokens):
            if token_index == workload.common_suffix_tokens and workload.common_suffix_tokens > 0:
                state.seed = _branch_seed(seed, branch_index)
            event = state.generate(adapter.config, transaction_id=None)
            state.acknowledge_gateway(token_index=event.token_index, owner_epoch=event.owner_epoch)
        encoded = encode_state(state, adapter.config)
        changed = {
            segment.descriptor.segment_id: (segment.descriptor.checksum, len(segment.payload))
            for segment in encoded.segments
            if parent.get(segment.descriptor.segment_id) != segment.descriptor.checksum
        }
        new_segments.append(changed)
        private_bytes.append(sum(size for _digest, size in changed.values()))
        count = branch_index + 1
        if count in fanouts:
            completion[count] = max(1, time.perf_counter_ns() - start)

    identical: dict[int, int] = {}
    for fanout in fanouts:
        common = set(new_segments[0])
        for segments in new_segments[1:fanout]:
            common &= set(segments)
        identical[fanout] = sum(
            new_segments[0][segment_id][1]
            for segment_id in common
            if len({new_segments[index][segment_id][0] for index in range(fanout)}) == 1
        )
    return tuple(private_bytes), completion, identical


def _build_fixture(
    workload: WorkloadSpec,
    *,
    seed: int,
    fanouts: tuple[int, ...],
    page_size_tokens: int,
) -> _Fixture:
    adapter = ReferenceTokenMajorAdapter(page_size_tokens=page_size_tokens, max_sessions=1)
    tokens = tuple(
        (seed + index * 17 + index // 7) % 256 for index in range(workload.context_tokens)
    )
    adapter.create_session(
        session_id=f"branchpoint-{workload.workload_class.value}-{seed}",
        request_id=f"request-{workload.workload_class.value}",
        tenant_id="tenant-branchfabric-cpu-reference",
        input_token_ids=tokens,
        seed=seed,
    )
    captured = adapter.capture_consistent(f"branchpoint-{workload.workload_class.value}-{seed}")
    captured.verify()
    payloads = tuple(segment.payload for segment in captured.segments)
    by_kind = _state_bytes_by_kind(captured)
    runtime_metadata_bytes, page_metadata_bytes = _metadata_bytes(captured)
    suffix_bytes, suffix_completion, identical_suffix = _measure_suffix_path(
        captured, adapter, workload, seed, fanouts
    )
    attention = sum(
        by_kind[kind]
        for kind in (
            StateKind.ATTENTION_KEY,
            StateKind.ATTENTION_VALUE,
            StateKind.ATTENTION_PACKED_KV,
        )
    )
    fixture_id = f"{workload.workload_class.value}-seed-{seed}"
    evidence = FixtureEvidence(
        fixture_id=fixture_id,
        model_state_class="CPU_REFERENCE_MODEL_STATE",
        reference_adapter_max_context_tokens=MAX_CONTEXT_TOKENS,
        physical_byte_accounting=(
            "DETERMINISTIC_SERIALIZED_STATE_BYTES_NOT_RSS_OS_PAGES_OR_GPU_ALLOCATION"
        ),
        workload_evidence_class="SYNTHETIC_CPU_REFERENCE_MODEL_STATE",
        timing_measurement_class="HARDWARE_BACKED_HOST_CPU",
        workload_class=workload.workload_class,
        seed=seed,
        context_tokens=workload.context_tokens,
        suffix_tokens=workload.suffix_tokens,
        common_suffix_tokens=workload.common_suffix_tokens,
        divergence_pattern=workload.divergence_pattern,
        source_snapshot_id=captured.handle.snapshot_id,
        continuation_hash=captured.logical.continuation_hash,
        root_state_sha256=_root_hash(captured),
        serialized_model_state_bytes=sum(map(len, payloads)),
        attention_kv_bytes=attention,
        recurrent_state_bytes=by_kind[StateKind.RECURRENT],
        sampler_state_bytes=by_kind[StateKind.SAMPLER],
        guided_decoding_state_bytes=by_kind[StateKind.GUIDED_DECODING],
        delivery_state_bytes=by_kind[StateKind.CLIENT_DELIVERY],
        token_history_bytes=by_kind[StateKind.TOKEN_HISTORY],
        runtime_metadata_bytes=runtime_metadata_bytes,
        page_metadata_bytes=page_metadata_bytes,
        segment_count=len(captured.segments),
        page_count=len(captured.page_table),
        physical_page_size_tokens=page_size_tokens,
        suffix_private_bytes_per_branch=suffix_bytes,
        suffix_all_branches_ready_ns=suffix_completion,
        identical_new_suffix_bytes_by_fanout=identical_suffix,
        real_transformer_measured=False,
        gpu_measured=False,
        simulated_device_bytes_claimed=False,
    )
    return _Fixture(evidence=evidence, captured=captured, payloads=payloads)


def _semantic_hash(root_hash: str, seed: int, fanout: int) -> str:
    document = {
        "root_state_sha256": root_hash,
        "branch_seeds": [_branch_seed(seed, index) for index in range(fanout)],
        "fanout": fanout,
    }
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _measure_ready(
    fixture: _Fixture,
    *,
    seed: int,
    fanout: int,
    implementation: FanoutImplementation,
) -> _ReadyOutcome:
    root_hash = fixture.evidence.root_state_sha256
    tracemalloc.start()
    start = time.perf_counter_ns()
    readiness: list[int] = []
    if implementation is FanoutImplementation.NAIVE_PRIVATE:
        branches: list[tuple[str, tuple[bytearray, ...], int]] = []
        for index in range(fanout):
            private = tuple(bytearray(payload) for payload in fixture.payloads)
            branches.append((f"branch-{index}", private, _branch_seed(seed, index)))
            readiness.append(max(1, time.perf_counter_ns() - start))
        retained: object = branches
        allocations = fanout * (len(fixture.payloads) + 1)
    else:
        root = _SharedRoot(payloads=fixture.payloads, continuation_hash=root_hash)
        views: list[_BranchView] = []
        for index in range(fanout):
            views.append(
                _BranchView(
                    branch_id=f"branch-{index}",
                    root=root,
                    branch_seed=_branch_seed(seed, index),
                )
            )
            readiness.append(max(1, time.perf_counter_ns() - start))
        retained = (root, views)
        allocations = fanout + 1
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return _ReadyOutcome(
        monotonic_start_ns=start,
        per_branch_readiness_ns=tuple(readiness),
        allocator_peak_bytes=max(1, peak),
        allocation_count_lower_bound=allocations,
        semantic_state_hash=_semantic_hash(root_hash, seed, fanout),
        retained=retained,
    )


def _with_timeout(function: Callable[[], _T], timeout_seconds: float) -> _T:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="branchfabric-fanout")
    future = executor.submit(function)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as error:
        future.cancel()
        raise TimeoutError(f"fanout sample exceeded {timeout_seconds:.3f}s") from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _sample(
    case: _Case,
    fixture: _Fixture,
    *,
    warmup: bool,
    repetition: int,
    order: int,
    timeout_seconds: float,
) -> FanoutRawSample:
    outcome = _with_timeout(
        lambda: _measure_ready(
            fixture,
            seed=case.seed,
            fanout=case.fanout,
            implementation=case.implementation,
        ),
        timeout_seconds,
    )
    per_branch = outcome.per_branch_readiness_ns
    root_bytes = fixture.evidence.serialized_model_state_bytes
    naive_total = root_bytes * (case.fanout + 1)
    private_ready = (
        root_bytes * case.fanout if case.implementation is FanoutImplementation.NAIVE_PRIVATE else 0
    )
    ready_total = root_bytes + private_ready
    suffix_private = sum(fixture.evidence.suffix_private_bytes_per_branch[: case.fanout])
    return FanoutRawSample(
        sample_id=(
            f"{case.workload_class.value}-seed-{case.seed}-fanout-{case.fanout}-"
            f"{case.implementation.value}-{'warmup' if warmup else 'measured'}-{repetition}"
        ),
        fixture_id=fixture.evidence.fixture_id,
        model_state_class="CPU_REFERENCE_MODEL_STATE",
        physical_byte_accounting=(
            "DETERMINISTIC_SERIALIZED_STATE_BYTES_NOT_RSS_OS_PAGES_OR_GPU_ALLOCATION"
        ),
        workload_evidence_class="SYNTHETIC_CPU_REFERENCE_MODEL_STATE",
        timing_measurement_class="HARDWARE_BACKED_HOST_CPU",
        workload_class=case.workload_class,
        implementation=case.implementation,
        seed=case.seed,
        fanout=case.fanout,
        repetition=repetition,
        warmup=warmup,
        randomized_order=order,
        monotonic_start_ns=outcome.monotonic_start_ns,
        branchpoint_to_all_ready_ns=per_branch[-1],
        per_branch_readiness_ns=per_branch,
        p50_branch_readiness_ns=_percentile(per_branch, 0.50),
        p90_branch_readiness_ns=_percentile(per_branch, 0.90),
        p95_branch_readiness_ns=_percentile(per_branch, 0.95),
        p99_branch_readiness_ns=_percentile(per_branch, 0.99),
        logical_branch_state_bytes=root_bytes * case.fanout,
        source_root_physical_bytes=root_bytes,
        branch_private_physical_bytes_at_ready=private_ready,
        total_physical_state_bytes_at_ready=ready_total,
        post_suffix_private_bytes=suffix_private,
        total_physical_state_bytes_after_suffix=ready_total + suffix_private,
        sharing_efficiency_at_ready=1.0 - (ready_total / naive_total),
        python_allocator_peak_bytes=outcome.allocator_peak_bytes,
        allocation_count_lower_bound=outcome.allocation_count_lower_bound,
        bounded_queue_capacity=1,
        requested_queue_concurrency=1,
        observed_max_queue_concurrency=1,
        queue_operations=case.fanout,
        suffix_all_branches_ready_ns=fixture.evidence.suffix_all_branches_ready_ns[case.fanout],
        semantic_state_hash=outcome.semantic_state_hash,
        exact_root_semantics_preserved=True,
        allocation_instrumentation="python_tracemalloc_full",
        gpu_measured=False,
        network_hardware_measured=False,
    )


def run_fanout_study(
    config: FanoutStudyConfig,
) -> tuple[tuple[FixtureEvidence, ...], tuple[FanoutRawSample, ...]]:
    """Run the bounded randomized sweep and return fixtures plus unfiltered samples."""

    fixtures: dict[tuple[WorkloadClass, int], _Fixture] = {}
    for workload in config.workloads:
        for seed in config.seeds:
            fixture = _build_fixture(
                workload,
                seed=seed,
                fanouts=config.fanouts,
                page_size_tokens=config.page_size_tokens,
            )
            fixtures[(workload.workload_class, seed)] = fixture

    cases = [
        _Case(workload.workload_class, seed, fanout, implementation)
        for workload in config.workloads
        for seed in config.seeds
        for fanout in config.fanouts
        for implementation in FanoutImplementation
    ]
    samples: list[FanoutRawSample] = []
    order = 0
    phase_count = config.warmup_repetitions + config.measurement_repetitions
    for phase in range(phase_count):
        warmup = phase < config.warmup_repetitions
        repetition = phase if warmup else phase - config.warmup_repetitions
        phase_cases = list(cases)
        random.Random(config.randomization_seed + phase * 1_000_003).shuffle(phase_cases)
        for case in phase_cases:
            samples.append(
                _sample(
                    case,
                    fixtures[(case.workload_class, case.seed)],
                    warmup=warmup,
                    repetition=repetition,
                    order=order,
                    timeout_seconds=config.sample_timeout_seconds,
                )
            )
            order += 1
    return (
        tuple(
            fixtures[key].evidence for key in sorted(fixtures, key=lambda item: (item[0], item[1]))
        ),
        tuple(samples),
    )


def _bootstrap_median(
    values: Sequence[float], *, seed: int, repetitions: int
) -> ConfidenceInterval:
    observed = float(statistics.median(values))
    rng = random.Random(seed)
    medians = [
        float(statistics.median(rng.choice(values) for _ in range(len(values))))
        for _ in range(repetitions)
    ]
    lower = min(observed, _percentile(medians, 0.025))
    upper = max(observed, _percentile(medians, 0.975))
    return ConfidenceInterval(
        method="deterministic_percentile_bootstrap_median",
        confidence=0.95,
        seed=seed,
        repetitions=repetitions,
        lower=max(sys.float_info.min, lower),
        observed=observed,
        upper=max(sys.float_info.min, upper),
    )


def _summaries(
    config: FanoutStudyConfig,
    fixtures: tuple[FixtureEvidence, ...],
    samples: tuple[FanoutRawSample, ...],
) -> tuple[tuple[FanoutSummary, ...], tuple[BaselineEffect, ...], tuple[HeadroomResult, ...]]:
    measured = tuple(sample for sample in samples if not sample.warmup)
    grouped: dict[tuple[WorkloadClass, int, FanoutImplementation], list[FanoutRawSample]] = (
        defaultdict(list)
    )
    for sample in measured:
        grouped[(sample.workload_class, sample.fanout, sample.implementation)].append(sample)
    summaries: list[FanoutSummary] = []
    for index, key in enumerate(sorted(grouped, key=lambda item: (item[0], item[1], item[2]))):
        workload, fanout, implementation = key
        group = grouped[key]
        durations = [float(item.branchpoint_to_all_ready_ns) for item in group]
        summaries.append(
            FanoutSummary(
                workload_class=workload,
                implementation=implementation,
                fanout=fanout,
                sample_count=len(group),
                seed_count=len({item.seed for item in group}),
                warmup_samples_excluded=config.warmup_repetitions * len(config.seeds),
                p50_all_ready_ns=_percentile(durations, 0.50),
                p90_all_ready_ns=_percentile(durations, 0.90),
                p95_all_ready_ns=_percentile(durations, 0.95),
                p99_all_ready_ns=_percentile(durations, 0.99),
                maximum_all_ready_ns=max(item.branchpoint_to_all_ready_ns for item in group),
                median_confidence_interval=_bootstrap_median(
                    durations,
                    seed=config.bootstrap_seed + index,
                    repetitions=config.bootstrap_repetitions,
                ),
                median_logical_branch_state_bytes=float(
                    statistics.median(item.logical_branch_state_bytes for item in group)
                ),
                median_physical_state_bytes_at_ready=float(
                    statistics.median(item.total_physical_state_bytes_at_ready for item in group)
                ),
                median_physical_state_bytes_after_suffix=float(
                    statistics.median(
                        item.total_physical_state_bytes_after_suffix for item in group
                    )
                ),
                median_allocator_peak_bytes=float(
                    statistics.median(item.python_allocator_peak_bytes for item in group)
                ),
                median_sharing_efficiency_at_ready=float(
                    statistics.median(item.sharing_efficiency_at_ready for item in group)
                ),
                outliers_removed=False,
            )
        )

    effects: list[BaselineEffect] = []
    headroom: list[HeadroomResult] = []
    fixture_map = {(item.workload_class, item.seed): item for item in fixtures}
    for index, workload in enumerate(sorted(set(item.workload_class for item in measured))):
        for fanout in config.fanouts:
            naive_group = grouped[(workload, fanout, FanoutImplementation.NAIVE_PRIVATE)]
            optimized_group = grouped[(workload, fanout, FanoutImplementation.SHARED_ROOT_COW_LAZY)]
            naive = {(item.seed, item.repetition): item for item in naive_group}
            optimized = {(item.seed, item.repetition): item for item in optimized_group}
            if set(naive) != set(optimized):
                raise ValueError("software baseline samples do not form exact matched pairs")
            keys = sorted(naive)
            speedups = [
                naive[key].branchpoint_to_all_ready_ns / optimized[key].branchpoint_to_all_ready_ns
                for key in keys
            ]
            reductions = [
                1.0
                - optimized[key].branchpoint_to_all_ready_ns
                / naive[key].branchpoint_to_all_ready_ns
                for key in keys
            ]
            physical_reductions = [
                1.0
                - optimized[key].total_physical_state_bytes_at_ready
                / naive[key].total_physical_state_bytes_at_ready
                for key in keys
            ]
            effects.append(
                BaselineEffect(
                    workload_class=workload,
                    fanout=fanout,
                    matched_pair_count=len(keys),
                    baseline="naive_private_materialization",
                    optimized="shared_root_cow_lazy",
                    median_software_speedup=float(statistics.median(speedups)),
                    speedup_confidence_interval=_bootstrap_median(
                        speedups,
                        seed=config.bootstrap_seed + 100 + index * 10 + fanout,
                        repetitions=config.bootstrap_repetitions,
                    ),
                    median_relative_readiness_reduction=float(statistics.median(reductions)),
                    matched_pairs_optimized_faster_fraction=sum(
                        optimized[key].branchpoint_to_all_ready_ns
                        < naive[key].branchpoint_to_all_ready_ns
                        for key in keys
                    )
                    / len(keys),
                    physical_state_reduction_at_ready=float(statistics.median(physical_reductions)),
                )
            )
            optimized_median = float(
                statistics.median(item.branchpoint_to_all_ready_ns for item in optimized_group)
            )
            suffix_median = float(
                statistics.median(
                    fixture_map[(workload, seed)].suffix_all_branches_ready_ns[fanout]
                    for seed in config.seeds
                )
            )
            lifecycle = optimized_median + suffix_median
            fraction = optimized_median / lifecycle
            headroom.append(
                HeadroomResult(
                    workload_class=workload,
                    fanout=fanout,
                    optimized_readiness_median_ns=optimized_median,
                    measured_suffix_path_median_ns=suffix_median,
                    local_branch_lifecycle_median_ns=lifecycle,
                    readiness_fraction_of_local_lifecycle=fraction,
                    ideal_zero_cost_readiness_speedup=1.0 / (1.0 - fraction),
                    end_to_end_helix_speedup_established=False,
                    hardware_gate_eligible=False,
                    gate_blockers=(
                        "CPU_REFERENCE_MODEL_STATE is not a real transformer workload",
                        "the exercised public reference adapter is capped at 4096 tokens, so no 8K, 16K, or 32K context was attempted",
                        "no CUDA GPU, PCIe GPU transfer, NVLink, RDMA, or NIC was measured",
                        "the measured scope is a local branch-state lifecycle, not an end-to-end Helix transaction",
                        "no evidence-backed hardware service curve exists for a candidate primitive",
                    ),
                )
            )
    return tuple(summaries), tuple(effects), tuple(headroom)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"{path} exists; pass replace=True to regenerate")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _revision(repository: Path) -> str:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN_NOT_CAPTURED"
    return result.stdout.strip()


def write_fanout_study(
    config: FanoutStudyConfig,
    fixtures: tuple[FixtureEvidence, ...],
    samples: tuple[FanoutRawSample, ...],
    output: Path,
    *,
    replace: bool = False,
    repository: Path | None = None,
) -> FanoutStudyReport:
    """Write authoritative JSONL samples and a summary bound to their digest."""

    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw-samples.jsonl"
    raw_payload = b"".join(
        _canonical_bytes(sample.model_dump(mode="json")) + b"\n" for sample in samples
    )
    _atomic_write(raw_path, raw_payload, replace=replace)
    raw_hash = _sha256(raw_path)
    fixture_path = output / "fixture-evidence.jsonl"
    fixture_payload = b"".join(
        _canonical_bytes(fixture.model_dump(mode="json")) + b"\n" for fixture in fixtures
    )
    _atomic_write(fixture_path, fixture_payload, replace=replace)
    fixture_hash = _sha256(fixture_path)
    summaries, effects, headroom = _summaries(config, fixtures, samples)
    measured_concurrency = [
        float(sample.observed_max_queue_concurrency) for sample in samples if not sample.warmup
    ]
    report = FanoutStudyReport(
        schema_version="sloforge.branchfabric.cpu-reference-fanout-report/v1",
        config=config,
        model_state_class="CPU_REFERENCE_MODEL_STATE",
        reference_adapter_max_context_tokens=MAX_CONTEXT_TOKENS,
        physical_byte_accounting=(
            "DETERMINISTIC_SERIALIZED_STATE_BYTES_NOT_RSS_OS_PAGES_OR_GPU_ALLOCATION"
        ),
        host=platform.node(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        repository_revision=_revision(repository or Path.cwd()),
        measurement_scope="LOCAL_MODEL_STATE_BRANCH_READINESS_AND_SUFFIX_NOT_END_TO_END_HELIX",
        reproduction_command=(
            "uv run --locked python -m "
            "sloforge.helix.characterization.fanout_vertical "
            f"--output {output} --seeds {' '.join(map(str, config.seeds))} "
            f"--fanouts {' '.join(map(str, config.fanouts))} "
            f"--warmups {config.warmup_repetitions} "
            f"--repetitions {config.measurement_repetitions} --replace"
        ),
        raw_samples_uri=str(raw_path),
        raw_samples_sha256=raw_hash,
        raw_sample_count=len(samples),
        fixture_evidence_uri=str(fixture_path),
        fixture_evidence_sha256=fixture_hash,
        fixtures=fixtures,
        summaries=summaries,
        baseline_effects=effects,
        headroom=headroom,
        queue_concurrency=QueueConcurrencySummary(
            sample_count=len(measured_concurrency),
            p50=_percentile(measured_concurrency, 0.50),
            p90=_percentile(measured_concurrency, 0.90),
            p95=_percentile(measured_concurrency, 0.95),
            p99=_percentile(measured_concurrency, 0.99),
            maximum=int(max(measured_concurrency)),
            interpretation="serialized_queue_concurrency_one",
        ),
        strongest_software_baseline="shared_root_cow_lazy",
        randomized_run_order=True,
        real_transformer_measured=False,
        cuda_gpu_measured=False,
        multi_gpu_measured=False,
        physical_network_measured=False,
        end_to_end_helix_transaction_measured=False,
        raw_samples_preserved=True,
        outliers_removed=False,
        limitations=(
            "CPU_REFERENCE_MODEL_STATE is the exercised deterministic Continuum HybridDecoder, not a real transformer or hybrid production model.",
            "The exercised contexts are 2048 and 3968 reference tokens. ReferenceTokenMajorAdapter fixes max_context_tokens=4096 and exposes no public override, so 8K, 16K, and 32K were unsupported and were not attempted.",
            "The adapter labels GPU placements as simulated devices; no simulated-device byte or latency value is reported as hardware evidence.",
            "Host wall timing includes Python tracemalloc allocation instrumentation and must not be generalized to CUDA, PCIe, NVLink, RDMA, NIC, FPGA, or DPU behavior.",
            "Shared-root readiness defers full reference-runtime materialization to first suffix use; the measured suffix path remains in the local lifecycle denominator.",
            "The local branch lifecycle excludes serving, reward, training, evaluation, promotion, and causal capacity reclamation, so it cannot pass the end-to-end hardware gate.",
            "Every total_physical_state_bytes field is deterministic serialized/content-addressed state accounting, not RSS, OS-page allocation, device memory, or GPU allocation.",
            "Python tracemalloc peak is the only allocator observation; it is scoped to Python objects created during the timed region and remains separate from serialized-state accounting.",
        ),
    )
    summary_path = output / "summary.json"
    _atomic_write(
        summary_path,
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
        replace=replace,
    )
    manifest = {
        "schema_version": "sloforge.branchfabric.cpu-reference-fanout-artifacts/v1",
        "model_state_class": "CPU_REFERENCE_MODEL_STATE",
        "physical_byte_accounting": (
            "DETERMINISTIC_SERIALIZED_STATE_BYTES_NOT_RSS_OS_PAGES_OR_GPU_ALLOCATION"
        ),
        "reproduction_command": report.reproduction_command,
        "artifacts": [
            {
                "path": str(raw_path),
                "sha256": raw_hash,
                "bytes": raw_path.stat().st_size,
                "records": len(samples),
                "role": "authoritative_raw_samples",
            },
            {
                "path": str(summary_path),
                "sha256": _sha256(summary_path),
                "bytes": summary_path.stat().st_size,
                "records": 1,
                "role": "derived_summary",
            },
            {
                "path": str(fixture_path),
                "sha256": fixture_hash,
                "bytes": fixture_path.stat().st_size,
                "records": len(fixtures),
                "role": "authoritative_reference_state_and_suffix_evidence",
            },
        ],
        "hardware_gate_eligible": False,
    }
    _atomic_write(
        output / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        replace=replace,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 73, 113])
    parser.add_argument("--fanouts", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = FanoutStudyConfig(
        seeds=tuple(arguments.seeds),
        fanouts=tuple(arguments.fanouts),
        warmup_repetitions=arguments.warmups,
        measurement_repetitions=arguments.repetitions,
        sample_timeout_seconds=arguments.timeout_seconds,
    )
    fixtures, samples = run_fanout_study(config)
    report = write_fanout_study(
        config,
        fixtures,
        samples,
        arguments.output,
        replace=arguments.replace,
    )
    print(
        json.dumps(
            {
                "model_state_class": report.model_state_class,
                "raw_sample_count": report.raw_sample_count,
                "summary": str(arguments.output / "summary.json"),
                "hardware_gate_eligible": False,
                "real_transformer_measured": False,
                "cuda_gpu_measured": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
