"""Fail-closed capacity calibration for BranchFabric Experiment 004.

This module is deliberately independent of Modal and CUDA.  A GPU worker records
the raw request timestamps from a probe; these functions validate the global
arrival clock and routing, derive the empirical capacity verdict, and choose the
next bounded probe.  In particular, a configured request rate is never promoted
to capacity evidence without complete raw request accounting.
"""

from __future__ import annotations

import math
import statistics
from enum import StrEnum
from itertools import pairwise
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NS_PER_SECOND = 1_000_000_000
DEFAULT_RATE_GRID_RPS = (10.0, 15.0, 17.25, 20.0, 23.5, 25.0, 30.0, 35.0)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ProbeTopology(StrEnum):
    GPU0_ONLY = "gpu0-only"
    TWO_GPU_ROUND_ROBIN = "two-gpu-round-robin"


class ProbeVerdict(StrEnum):
    SUSTAINABLE = "sustainable"
    UNSUSTAINABLE = "unsustainable"
    INCONCLUSIVE = "inconclusive"


class RequestTerminalState(StrEnum):
    COMPLETED = "completed"
    ABORTED = "aborted"
    NOT_ADMITTED = "not-admitted"


class CapacityCalibrationBounds(_StrictModel):
    """Immutable resource and measurement bounds for the one-container search."""

    seed: int = Field(ge=0, lt=1 << 63)
    gpu_count: Literal[2] = 2
    output_tokens: Literal[64] = 64
    warmup_seconds: float = Field(default=1.0, ge=1.0, le=5.0, allow_inf_nan=False)
    minimum_measurement_seconds: float = Field(default=10.0, ge=10.0, le=20.0, allow_inf_nan=False)
    maximum_measurement_seconds: float = Field(default=10.0, ge=10.0, le=20.0, allow_inf_nan=False)
    maximum_probes: int = Field(default=8, ge=3, le=8)
    maximum_allocation_wall_seconds: float = Field(
        default=212.0, gt=0.0, le=212.0, allow_inf_nan=False
    )
    remaining_gpu_seconds_before: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_calibration_fraction: float = Field(
        default=0.2625, gt=0.0, le=0.2625, allow_inf_nan=False
    )
    rate_grid_rps: tuple[float, ...] = DEFAULT_RATE_GRID_RPS

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.maximum_measurement_seconds < self.minimum_measurement_seconds:
            raise ValueError("maximum measurement duration precedes its minimum")
        if (
            len(self.rate_grid_rps) < 3
            or any(not math.isfinite(value) or value <= 0.0 for value in self.rate_grid_rps)
            or tuple(sorted(set(self.rate_grid_rps))) != self.rate_grid_rps
        ):
            raise ValueError("capacity rate grid must be finite, positive, unique, and sorted")
        if self.maximum_gpu_seconds > (
            self.remaining_gpu_seconds_before * self.maximum_calibration_fraction + 1e-9
        ):
            raise ValueError("calibration reservation exceeds its fraction of remaining budget")
        return self

    @property
    def maximum_gpu_seconds(self) -> float:
        return self.gpu_count * self.maximum_allocation_wall_seconds


class GlobalArrival(_StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    global_sequence: int = Field(ge=0)
    scheduled_arrival_ns: int = Field(ge=0)
    assigned_device: Literal["gpu0", "gpu1"]


class CapacityProbePlan(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.capacity-probe-plan/v1"] = (
        "sloforge.branchfabric.capacity-probe-plan/v1"
    )
    probe_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{3,95}$")
    seed: int = Field(ge=0, lt=1 << 63)
    topology: ProbeTopology
    configured_rate_rps: float = Field(gt=0.0, allow_inf_nan=False)
    warmup_start_ns: int = Field(ge=0)
    measurement_start_ns: int = Field(ge=0)
    measurement_end_ns: int = Field(gt=0)
    output_tokens: Literal[64] = 64
    arrivals: tuple[GlobalArrival, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_global_stream(self) -> Self:
        if not self.warmup_start_ns < self.measurement_start_ns < self.measurement_end_ns:
            raise ValueError("probe warmup and measurement bounds must be ordered")
        sequences = tuple(item.global_sequence for item in self.arrivals)
        if sequences != tuple(range(len(self.arrivals))):
            raise ValueError("global arrival sequences must be contiguous from zero")
        if len({item.request_id for item in self.arrivals}) != len(self.arrivals):
            raise ValueError("global arrival request IDs must be unique")
        timestamps = tuple(item.scheduled_arrival_ns for item in self.arrivals)
        if timestamps != tuple(sorted(timestamps)):
            raise ValueError("global arrival timestamps must be monotonic")
        if any(timestamp < self.warmup_start_ns for timestamp in timestamps):
            raise ValueError("global arrival precedes probe warmup")
        for item in self.arrivals:
            expected = (
                "gpu0"
                if self.topology == ProbeTopology.GPU0_ONLY or item.global_sequence % 2 == 0
                else "gpu1"
            )
            if item.assigned_device != expected:
                raise ValueError("arrival routing differs from the single global stream policy")
        return self

    @property
    def measurement_duration_seconds(self) -> float:
        return (self.measurement_end_ns - self.measurement_start_ns) / NS_PER_SECOND


class CapacityRequestObservation(_StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    global_sequence: int = Field(ge=0)
    device: Literal["gpu0", "gpu1"]
    scheduled_arrival_ns: int = Field(ge=0)
    offered_ns: int = Field(ge=0)
    enqueued_ns: int | None = Field(default=None, ge=0)
    admitted_ns: int | None = Field(default=None, ge=0)
    first_token_ns: int | None = Field(default=None, ge=0)
    completed_ns: int | None = Field(default=None, ge=0)
    terminated_ns: int = Field(ge=0)
    requested_output_tokens: Literal[64] = 64
    emitted_tokens: int = Field(ge=0, le=64)
    terminal_state: RequestTerminalState

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        if self.offered_ns < self.scheduled_arrival_ns:
            raise ValueError("request was offered before its scheduled arrival")
        if self.enqueued_ns is not None and self.enqueued_ns < self.offered_ns:
            raise ValueError("request was enqueued before it was offered")
        if self.admitted_ns is not None and (
            self.enqueued_ns is None or self.admitted_ns < self.enqueued_ns
        ):
            raise ValueError("request admission requires a preceding enqueue")
        if self.first_token_ns is not None and (
            self.admitted_ns is None or self.first_token_ns < self.admitted_ns
        ):
            raise ValueError("first token precedes admission")
        if self.completed_ns is not None:
            if self.admitted_ns is None:
                raise ValueError("request completion requires admission")
            lower = self.first_token_ns if self.first_token_ns is not None else self.admitted_ns
            if self.completed_ns < lower:
                raise ValueError("completion precedes request progress")
        latest = max(
            timestamp
            for timestamp in (
                self.offered_ns,
                self.enqueued_ns,
                self.admitted_ns,
                self.first_token_ns,
                self.completed_ns,
            )
            if timestamp is not None
        )
        if self.terminated_ns < latest:
            raise ValueError("terminal timestamp precedes request progress")
        if self.terminal_state == RequestTerminalState.COMPLETED:
            if self.completed_ns is None or self.emitted_tokens != self.requested_output_tokens:
                raise ValueError("completed requests must emit the exact sampling target")
        elif self.completed_ns is not None:
            raise ValueError("unfinished requests cannot carry completion timestamps")
        if self.terminal_state == RequestTerminalState.ABORTED and self.admitted_ns is None:
            raise ValueError("only admitted requests may be aborted")
        if (
            self.terminal_state == RequestTerminalState.NOT_ADMITTED
            and self.admitted_ns is not None
        ):
            raise ValueError("not-admitted requests cannot carry admission timestamps")
        return self


class CapacityProbeRaw(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.capacity-probe-raw/v1"] = (
        "sloforge.branchfabric.capacity-probe-raw/v1"
    )
    plan: CapacityProbePlan
    maximum_plan: CapacityProbePlan | None = None
    observations: tuple[CapacityRequestObservation, ...] = Field(max_length=10_000)
    tail_drain_end_ns: int = Field(ge=0)
    probe_end_ns: int = Field(ge=0)
    tail_drain_seconds: float = Field(gt=0.0, le=5.0, allow_inf_nan=False)
    stopped_early: bool = False
    stop_reason: str | None = None

    @model_validator(mode="after")
    def validate_stop_reason(self) -> Self:
        expected_tail_end = self.plan.measurement_end_ns + round(
            self.tail_drain_seconds * NS_PER_SECOND
        )
        if self.tail_drain_end_ns != expected_tail_end:
            raise ValueError("probe tail-drain timestamp differs from its fixed bound")
        if (
            not self.tail_drain_end_ns
            <= self.probe_end_ns
            <= self.tail_drain_end_ns + NS_PER_SECOND
        ):
            raise ValueError("probe end exceeds the one-second abort-finalization allowance")
        if any(item.terminated_ns > self.probe_end_ns for item in self.observations):
            raise ValueError("request terminated after the recorded probe end")
        if self.stopped_early != (self.stop_reason is not None):
            raise ValueError("early-stop status and reason must be present together")
        if self.stopped_early != (self.maximum_plan is not None):
            raise ValueError("an early stop must preserve its original maximum plan")
        if self.maximum_plan is not None:
            maximum = self.maximum_plan
            comparable = (
                "probe_id",
                "seed",
                "topology",
                "configured_rate_rps",
                "warmup_start_ns",
                "measurement_start_ns",
                "output_tokens",
            )
            if any(getattr(self.plan, field) != getattr(maximum, field) for field in comparable):
                raise ValueError("executed early-stop plan differs from its original plan")
            expected_prefix = tuple(
                item
                for item in maximum.arrivals
                if item.scheduled_arrival_ns < self.plan.measurement_end_ns
            )
            if (
                self.plan.measurement_end_ns >= maximum.measurement_end_ns
                or self.plan.arrivals != expected_prefix
            ):
                raise ValueError("executed early-stop plan is not an exact original-plan prefix")
        return self


class CapacityProbeResult(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.capacity-probe-result/v1"] = (
        "sloforge.branchfabric.capacity-probe-result/v1"
    )
    probe_id: str
    topology: ProbeTopology
    configured_rate_rps: float = Field(gt=0.0, allow_inf_nan=False)
    observed_offered_rate_rps: float = Field(ge=0.0, allow_inf_nan=False)
    completed_rate_rps: float = Field(ge=0.0, allow_inf_nan=False)
    completed_rate_estimator: Literal["measurement-interval-completion-events"] = (
        "measurement-interval-completion-events"
    )
    completed_rate_sample_count: int = Field(default=0, ge=0)
    completed_rate_span_seconds: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    interval_offered_requests: int = Field(default=0, ge=0)
    interval_completed_requests: int = Field(default=0, ge=0)
    queue_flow_start_depth: int = Field(default=0, ge=0)
    queue_flow_expected_end_depth: int = Field(default=0, ge=0)
    queue_flow_observed_end_depth: int = Field(default=0, ge=0)
    queue_flow_conservation_error_requests: int = 0
    queue_flow_conservation_pass: bool = False
    p95_ttft_seconds: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    queue_depth_start: int = Field(ge=0)
    queue_depth_end: int = Field(ge=0)
    maximum_queue_depth: int = Field(ge=0)
    queue_slope_requests_per_second: float = Field(allow_inf_nan=False)
    queue_persistent_positive_drift: bool
    configured_arrival_rate_verified: bool
    offer_cadence_verified: bool
    admission_cadence_verified: bool
    admission_lag_pass: bool
    maximum_offer_lateness_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    p95_offer_lateness_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    p95_offer_interarrival_error_fraction: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_offer_burst_50ms: int = Field(ge=0)
    offer_burst_limit_50ms: int = Field(ge=1)
    maximum_admission_lateness_seconds: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    p95_admission_interarrival_error_fraction: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    maximum_admission_burst_50ms: int = Field(ge=0)
    admission_burst_limit_50ms: int = Field(ge=1)
    routing_verified: bool
    monotonic_timestamps_verified: bool
    output_target_verified: bool
    complete_request_accounting: bool
    planned_requests: int = Field(ge=1)
    observed_requests: int = Field(ge=0)
    measurement_requests: int = Field(ge=1)
    all_measurement_requests_completed: bool
    completed_requests: int = Field(ge=0)
    aborted_requests: int = Field(ge=0)
    not_admitted_requests: int = Field(ge=0)
    ttft_slo_pass: bool
    queue_stability_pass: bool
    throughput_pass: bool
    verdict: ProbeVerdict
    reasons: tuple[str, ...]


class SpikeSelection(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.capacity-selection/v1"] = (
        "sloforge.branchfabric.capacity-selection/v1"
    )
    lambda_1_rps: float = Field(gt=0.0, allow_inf_nan=False)
    lambda_2_rps: float = Field(gt=0.0, allow_inf_nan=False)
    lambda_spike_rps: float = Field(gt=0.0, allow_inf_nan=False)
    spike_over_lambda_1: float = Field(gt=1.0, allow_inf_nan=False)
    spike_over_lambda_2: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    one_gpu_unsustainable_probe_id: str
    one_gpu_sustainable_probe_id: str
    two_gpu_sustainable_probe_id: str
    reviewer_gate_required: Literal[True] = True


class NextProbe(_StrictModel):
    status: Literal["probe", "ready", "blocked"]
    topology: ProbeTopology | None = None
    rate_rps: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    reason: str
    selection: SpikeSelection | None = None

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        has_probe = self.topology is not None or self.rate_rps is not None
        if (self.status == "probe") != has_probe:
            raise ValueError("only a probe action may carry topology and rate")
        if self.status == "probe" and (self.topology is None or self.rate_rps is None):
            raise ValueError("probe action requires topology and rate")
        if (self.status == "ready") != (self.selection is not None):
            raise ValueError("only a ready action may carry a spike selection")
        return self


def build_probe_plan(
    *,
    probe_id: str,
    seed: int,
    topology: ProbeTopology,
    configured_rate_rps: float,
    start_ns: int,
    warmup_seconds: float = 1.0,
    measurement_seconds: float = 10.0,
) -> CapacityProbePlan:
    """Build one exact constant-cadence global stream and deterministic routing map."""

    numeric = (configured_rate_rps, warmup_seconds, measurement_seconds)
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
        raise ValueError("probe rate and durations must be finite and positive")
    if start_ns < 0:
        raise ValueError("probe start must be non-negative")
    measurement_start_ns = start_ns + round(warmup_seconds * NS_PER_SECOND)
    measurement_end_ns = measurement_start_ns + round(measurement_seconds * NS_PER_SECOND)
    interval_ns = NS_PER_SECOND / configured_rate_rps
    arrivals: list[GlobalArrival] = []
    sequence = 0
    while True:
        timestamp_ns = start_ns + round(sequence * interval_ns)
        if timestamp_ns >= measurement_end_ns:
            break
        device: Literal["gpu0", "gpu1"] = (
            "gpu0" if topology == ProbeTopology.GPU0_ONLY or sequence % 2 == 0 else "gpu1"
        )
        arrivals.append(
            GlobalArrival(
                request_id=f"{probe_id}-{sequence:05d}",
                global_sequence=sequence,
                scheduled_arrival_ns=timestamp_ns,
                assigned_device=device,
            )
        )
        sequence += 1
    return CapacityProbePlan(
        probe_id=probe_id,
        seed=seed,
        topology=topology,
        configured_rate_rps=configured_rate_rps,
        warmup_start_ns=start_ns,
        measurement_start_ns=measurement_start_ns,
        measurement_end_ns=measurement_end_ns,
        arrivals=tuple(arrivals),
    )


def _percentile(values: tuple[float, ...], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _queue_depth(
    plan: CapacityProbePlan, observations: dict[str, CapacityRequestObservation], at_ns: int
) -> int:
    depth = 0
    for arrival in plan.arrivals:
        observation = observations.get(arrival.request_id)
        # Queue membership begins at the actual client offer, not at the ideal
        # schedule. Counting a delayed, not-yet-offered request invents backlog.
        offered_ns = arrival.scheduled_arrival_ns if observation is None else observation.offered_ns
        if offered_ns > at_ns:
            continue
        if (
            observation is None
            or observation.completed_ns is None
            or observation.completed_ns > at_ns
        ):
            depth += 1
    return depth


def _linear_slope(points: tuple[tuple[float, int], ...]) -> float:
    if len(points) < 2:
        return 0.0
    center_x = statistics.fmean(item[0] for item in points)
    center_y = statistics.fmean(item[1] for item in points)
    denominator = sum((x - center_x) ** 2 for x, _ in points)
    if denominator == 0.0:
        return 0.0
    return sum((x - center_x) * (y - center_y) for x, y in points) / denominator


def _maximum_burst(timestamps: tuple[int, ...], *, window_ns: int) -> int:
    ordered = sorted(timestamps)
    left = 0
    maximum = 0
    for right, timestamp in enumerate(ordered):
        while timestamp - ordered[left] >= window_ns:
            left += 1
        maximum = max(maximum, right - left + 1)
    return maximum


def evaluate_probe(raw: CapacityProbeRaw, *, slo_ttft_seconds: float = 2.0) -> CapacityProbeResult:
    """Validate raw evidence and classify the empirical operating point."""

    plan = raw.plan
    observations = {item.request_id: item for item in raw.observations}
    duplicate_observations = len(observations) != len(raw.observations)
    planned = {item.request_id: item for item in plan.arrivals}
    identity_match = set(observations) == set(planned) and not duplicate_observations
    routing_verified = identity_match and all(
        observations[item.request_id].global_sequence == item.global_sequence
        and observations[item.request_id].device == item.assigned_device
        and observations[item.request_id].scheduled_arrival_ns == item.scheduled_arrival_ns
        for item in plan.arrivals
    )
    device_observations = tuple(
        tuple(
            sorted(
                (item for item in raw.observations if item.device == device),
                key=lambda item: item.global_sequence,
            )
        )
        for device in ("gpu0", "gpu1")
    )
    monotonic = (
        identity_match
        and all(
            observation.offered_ns >= planned[observation.request_id].scheduled_arrival_ns
            and (
                observation.admitted_ns is None or observation.admitted_ns >= observation.offered_ns
            )
            for observation in raw.observations
        )
        and all(
            left.offered_ns <= right.offered_ns
            for device_rows in device_observations
            for left, right in pairwise(device_rows)
        )
        and all(
            left.admitted_ns <= right.admitted_ns
            for device_rows in device_observations
            for left, right in pairwise(
                tuple(item for item in device_rows if item.admitted_ns is not None)
            )
            if left.admitted_ns is not None and right.admitted_ns is not None
        )
    )
    output_verified = identity_match and all(
        observation.requested_output_tokens == plan.output_tokens
        and (
            observation.terminal_state != RequestTerminalState.COMPLETED
            or observation.emitted_tokens == plan.output_tokens
        )
        for observation in raw.observations
    )
    measurement_arrivals = tuple(
        item
        for item in plan.arrivals
        if plan.measurement_start_ns <= item.scheduled_arrival_ns < plan.measurement_end_ns
    )
    duration_s = plan.measurement_duration_seconds
    measurement_observations = tuple(
        observations[item.request_id]
        for item in measurement_arrivals
        if item.request_id in observations
    )
    offered_in_window = tuple(
        item
        for item in measurement_observations
        if plan.measurement_start_ns <= item.offered_ns < plan.measurement_end_ns
    )
    offered_rate = len(offered_in_window) / duration_s
    configured_count_verified = (
        abs(len(measurement_arrivals) / duration_s - plan.configured_rate_rps)
        <= 1.0 / duration_s + 1e-12
    )
    offer_lateness = tuple(
        (item.offered_ns - item.scheduled_arrival_ns) / NS_PER_SECOND
        for item in measurement_observations
    )
    maximum_offer_lateness = max(offer_lateness, default=0.0)
    p95_offer_lateness = _percentile(offer_lateness, 0.95)
    expected_interval_s = 1.0 / plan.configured_rate_rps
    offered_timestamps = tuple(sorted(item.offered_ns for item in measurement_observations))
    interarrival_errors = tuple(
        abs((right - left) / NS_PER_SECOND - expected_interval_s) / expected_interval_s
        for left, right in pairwise(offered_timestamps)
    )
    p95_interarrival_error = _percentile(interarrival_errors, 0.95)
    burst_limit = math.ceil(plan.configured_rate_rps * 0.05) + 2
    maximum_offer_burst = _maximum_burst(offered_timestamps, window_ns=50_000_000)
    offer_cadence_verified = (
        len(measurement_observations) == len(measurement_arrivals)
        and len(offered_in_window) == len(measurement_arrivals)
        and maximum_offer_lateness <= 0.150
        and p95_offer_lateness is not None
        and p95_offer_lateness <= 0.050
        and p95_interarrival_error is not None
        and p95_interarrival_error <= 0.75
        and maximum_offer_burst <= burst_limit
    )
    arrival_rate_verified = configured_count_verified and offer_cadence_verified
    admission_lateness = tuple(
        (item.admitted_ns - item.scheduled_arrival_ns) / NS_PER_SECOND
        for item in measurement_observations
        if item.admitted_ns is not None
    )
    maximum_admission_lateness = max(admission_lateness) if admission_lateness else None
    admitted_timestamps = tuple(
        item.admitted_ns for item in measurement_observations if item.admitted_ns is not None
    )
    maximum_admission_burst = _maximum_burst(admitted_timestamps, window_ns=50_000_000)
    admission_interarrival_errors = tuple(
        abs(
            (right.admitted_ns - left.admitted_ns) / NS_PER_SECOND
            - (1.0 if plan.topology == ProbeTopology.GPU0_ONLY else 2.0) * expected_interval_s
        )
        / ((1.0 if plan.topology == ProbeTopology.GPU0_ONLY else 2.0) * expected_interval_s)
        for device_rows in device_observations
        for left, right in pairwise(
            tuple(
                item
                for item in device_rows
                if item.admitted_ns is not None
                and plan.measurement_start_ns <= item.scheduled_arrival_ns < plan.measurement_end_ns
            )
        )
        if left.admitted_ns is not None and right.admitted_ns is not None
    )
    p95_admission_interarrival_error = _percentile(admission_interarrival_errors, 0.95)
    # Admission lag is a capacity signal: a smooth, growing lag must remain eligible
    # for an UNSUSTAINABLE verdict.  Bursty admission, however, invalidates the probe
    # because it changes the request shape delivered to vLLM.
    admission_cadence_verified = monotonic and maximum_admission_burst <= burst_limit
    admission_lag_pass = (
        maximum_admission_lateness is not None and maximum_admission_lateness <= 0.250
    )
    # Capacity is the count of real completion events during the half-open
    # measurement interval. TTFT and output correctness remain attached to the
    # scheduled measurement cohort. Queue-flow conservation proves warmup
    # carry-ins and cohort carry-outs balance rather than inflating capacity:
    # Q_end = Q_start + interval_offers - interval_completions.
    interval_offered_requests = sum(
        plan.measurement_start_ns <= item.offered_ns < plan.measurement_end_ns
        for item in raw.observations
    )
    interval_completed_requests = sum(
        item.completed_ns is not None
        and plan.measurement_start_ns <= item.completed_ns < plan.measurement_end_ns
        for item in raw.observations
    )
    completed_rate = interval_completed_requests / duration_s
    ttft_values: list[float] = []
    for item in measurement_arrivals:
        observation = observations.get(item.request_id)
        if observation is not None and observation.first_token_ns is not None:
            ttft_values.append(
                (observation.first_token_ns - item.scheduled_arrival_ns) / NS_PER_SECOND
            )
    ttfts = tuple(ttft_values)
    p95_ttft = _percentile(ttfts, 0.95)
    sample_count = math.floor(duration_s) + 1
    queue_points_list: list[tuple[float, int]] = []
    for index in range(sample_count):
        offset = duration_s * index / (sample_count - 1)
        if index == 0:
            sample_ns = plan.measurement_start_ns - 1
        elif index == sample_count - 1:
            sample_ns = plan.measurement_end_ns - 1
        else:
            sample_ns = plan.measurement_start_ns + round(offset * 1e9)
        queue_points_list.append((offset, _queue_depth(plan, observations, sample_ns)))
    queue_points = tuple(queue_points_list)
    queue_flow_start_depth = queue_points[0][1]
    queue_flow_observed_end_depth = queue_points[-1][1]
    queue_flow_expected_end_depth = (
        queue_flow_start_depth + interval_offered_requests - interval_completed_requests
    )
    queue_flow_conservation_error = queue_flow_observed_end_depth - queue_flow_expected_end_depth
    queue_flow_conservation_pass = (
        queue_flow_expected_end_depth >= 0 and queue_flow_conservation_error == 0
    )
    slope = _linear_slope(queue_points)
    first_quarter = tuple(depth for _, depth in queue_points[: max(2, len(queue_points) // 4)])
    last_quarter = tuple(depth for _, depth in queue_points[-max(2, len(queue_points) // 4) :])
    persistent_positive_drift = (
        slope > 0.10 and statistics.median(last_quarter) > statistics.median(first_quarter) + 2
    )
    complete = identity_match and output_verified
    all_measurement_completed = len(measurement_observations) == len(measurement_arrivals) and all(
        item.terminal_state == RequestTerminalState.COMPLETED for item in measurement_observations
    )
    completed_requests = sum(
        item.terminal_state == RequestTerminalState.COMPLETED for item in raw.observations
    )
    aborted_requests = sum(
        item.terminal_state == RequestTerminalState.ABORTED for item in raw.observations
    )
    not_admitted_requests = sum(
        item.terminal_state == RequestTerminalState.NOT_ADMITTED for item in raw.observations
    )
    ttft_pass = p95_ttft is not None and p95_ttft <= slo_ttft_seconds
    queue_pass = not persistent_positive_drift
    # Apply the declared five-percent operating-point tolerance while exact
    # cohort completion, flow conservation, queue drift, and TTFT remain
    # independent sustainability gates.
    throughput_pass = all_measurement_completed and completed_rate + 1e-12 >= offered_rate * 0.95
    validity = (
        arrival_rate_verified
        and admission_cadence_verified
        and routing_verified
        and monotonic
        and output_verified
        and complete
        and queue_flow_conservation_pass
    )
    obvious_overload = (
        max(depth for _, depth in queue_points) >= 64
        or (p95_ttft is not None and p95_ttft >= 2.0 * slo_ttft_seconds)
        or (duration_s >= 5.0 and persistent_positive_drift and not throughput_pass)
        or (duration_s >= 10.0 and not all_measurement_completed and not throughput_pass)
    )
    sufficient_window = duration_s >= 10.0 or (raw.stopped_early and obvious_overload)
    if not validity or not sufficient_window:
        verdict = ProbeVerdict.INCONCLUSIVE
    elif (
        duration_s >= 10.0
        and all_measurement_completed
        and admission_lag_pass
        and ttft_pass
        and queue_pass
        and throughput_pass
    ):
        verdict = ProbeVerdict.SUSTAINABLE
    else:
        verdict = ProbeVerdict.UNSUSTAINABLE
    reasons: list[str] = []
    checks = (
        (arrival_rate_verified, "configured arrival rate was not reproduced"),
        (offer_cadence_verified, "producer offer cadence/lateness/burst audit failed"),
        (admission_cadence_verified, "admission timestamp monotonicity/burst audit failed"),
        (routing_verified, "global-stream routing was not reproduced"),
        (monotonic, "admission timestamps precede scheduled arrivals"),
        (output_verified, "64-token completion target was not reproduced"),
        (complete, "terminal request accounting is incomplete"),
        (queue_flow_conservation_pass, "measurement queue flow does not conserve requests"),
        (
            sufficient_window,
            "measurement is shorter than 10 seconds without an objective early overload gate",
        ),
        (ttft_pass, "p95 TTFT exceeds the 2-second SLO"),
        (admission_lag_pass, "maximum admission lag exceeds the sustainable bound"),
        (queue_pass, "queue depth has persistent positive drift"),
        (throughput_pass, "completed throughput is below offered load"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    return CapacityProbeResult(
        probe_id=plan.probe_id,
        topology=plan.topology,
        configured_rate_rps=plan.configured_rate_rps,
        observed_offered_rate_rps=offered_rate,
        completed_rate_rps=completed_rate,
        completed_rate_sample_count=interval_completed_requests,
        completed_rate_span_seconds=duration_s,
        interval_offered_requests=interval_offered_requests,
        interval_completed_requests=interval_completed_requests,
        queue_flow_start_depth=queue_flow_start_depth,
        queue_flow_expected_end_depth=queue_flow_expected_end_depth,
        queue_flow_observed_end_depth=queue_flow_observed_end_depth,
        queue_flow_conservation_error_requests=queue_flow_conservation_error,
        queue_flow_conservation_pass=queue_flow_conservation_pass,
        p95_ttft_seconds=p95_ttft,
        queue_depth_start=queue_points[0][1],
        queue_depth_end=queue_points[-1][1],
        maximum_queue_depth=max(depth for _, depth in queue_points),
        queue_slope_requests_per_second=slope,
        queue_persistent_positive_drift=persistent_positive_drift,
        configured_arrival_rate_verified=arrival_rate_verified,
        offer_cadence_verified=offer_cadence_verified,
        admission_cadence_verified=admission_cadence_verified,
        admission_lag_pass=admission_lag_pass,
        maximum_offer_lateness_seconds=maximum_offer_lateness,
        p95_offer_lateness_seconds=p95_offer_lateness or 0.0,
        p95_offer_interarrival_error_fraction=p95_interarrival_error or 0.0,
        maximum_offer_burst_50ms=maximum_offer_burst,
        offer_burst_limit_50ms=burst_limit,
        maximum_admission_lateness_seconds=maximum_admission_lateness,
        p95_admission_interarrival_error_fraction=p95_admission_interarrival_error,
        maximum_admission_burst_50ms=maximum_admission_burst,
        admission_burst_limit_50ms=burst_limit,
        routing_verified=routing_verified,
        monotonic_timestamps_verified=monotonic,
        output_target_verified=output_verified,
        complete_request_accounting=complete,
        planned_requests=len(plan.arrivals),
        observed_requests=len(raw.observations),
        measurement_requests=len(measurement_arrivals),
        all_measurement_requests_completed=all_measurement_completed,
        completed_requests=completed_requests,
        aborted_requests=aborted_requests,
        not_admitted_requests=not_admitted_requests,
        ttft_slo_pass=ttft_pass,
        queue_stability_pass=queue_pass,
        throughput_pass=throughput_pass,
        verdict=verdict,
        reasons=tuple(reasons),
    )


def choose_spike(results: tuple[CapacityProbeResult, ...]) -> SpikeSelection | None:
    """Select a measured GPU0 overload with the preferred two-GPU headroom.

    The baseline path deliberately does not implement the 0.80--0.90 ADR
    exception.  A spike above 80% of the measured two-GPU operating point must
    therefore fail closed instead of being silently promoted to v10.
    """

    one_stable = sorted(
        (
            item
            for item in results
            if item.topology == ProbeTopology.GPU0_ONLY and item.verdict == ProbeVerdict.SUSTAINABLE
        ),
        key=lambda item: item.configured_rate_rps,
    )
    one_unstable = tuple(
        item
        for item in results
        if item.topology == ProbeTopology.GPU0_ONLY and item.verdict == ProbeVerdict.UNSUSTAINABLE
    )
    two_stable = sorted(
        (
            item
            for item in results
            if item.topology == ProbeTopology.TWO_GPU_ROUND_ROBIN
            and item.verdict == ProbeVerdict.SUSTAINABLE
        ),
        key=lambda item: item.configured_rate_rps,
    )
    if not one_stable or not one_unstable or not two_stable:
        return None
    lambda_1 = one_stable[-1]
    lambda_2 = two_stable[-1]
    candidates = tuple(
        item
        for item in one_unstable
        if 1.15 <= item.configured_rate_rps / lambda_1.configured_rate_rps <= 1.30
        and item.configured_rate_rps / lambda_2.configured_rate_rps <= 0.80
    )
    if not candidates:
        return None
    spike = min(
        candidates,
        key=lambda item: (
            abs(item.configured_rate_rps / lambda_2.configured_rate_rps - 0.75),
            item.configured_rate_rps,
        ),
    )
    return SpikeSelection(
        lambda_1_rps=lambda_1.configured_rate_rps,
        lambda_2_rps=lambda_2.configured_rate_rps,
        lambda_spike_rps=spike.configured_rate_rps,
        spike_over_lambda_1=spike.configured_rate_rps / lambda_1.configured_rate_rps,
        spike_over_lambda_2=spike.configured_rate_rps / lambda_2.configured_rate_rps,
        one_gpu_unsustainable_probe_id=spike.probe_id,
        one_gpu_sustainable_probe_id=lambda_1.probe_id,
        two_gpu_sustainable_probe_id=lambda_2.probe_id,
    )


def recommend_next_probe(
    results: tuple[CapacityProbeResult, ...],
    *,
    maximum_probes: int = 8,
    rate_grid_rps: tuple[float, ...] = DEFAULT_RATE_GRID_RPS,
) -> NextProbe:
    """Advance a bounded adaptive search using only conclusive raw probe results."""

    if any(item.verdict == ProbeVerdict.INCONCLUSIVE for item in results):
        return NextProbe(status="blocked", reason="a capacity probe is inconclusive")
    selection = choose_spike(results)
    if selection is not None:
        return NextProbe(
            status="ready",
            reason="measured spike satisfies one-GPU overload and two-GPU headroom bounds",
            selection=selection,
        )
    if len(results) >= maximum_probes:
        return NextProbe(
            status="blocked",
            reason="bounded search ended without a measured safe spike interval",
        )
    tested = {(item.topology, item.configured_rate_rps) for item in results}
    one_results = tuple(item for item in results if item.topology == ProbeTopology.GPU0_ONLY)
    one_stable = sorted(
        item.configured_rate_rps
        for item in results
        if item.topology == ProbeTopology.GPU0_ONLY and item.verdict == ProbeVerdict.SUSTAINABLE
    )
    one_unstable = sorted(
        item.configured_rate_rps
        for item in results
        if item.topology == ProbeTopology.GPU0_ONLY and item.verdict == ProbeVerdict.UNSUSTAINABLE
    )
    if not one_stable:
        if not one_results:
            rate = 20.0
        else:
            lower = tuple(
                value
                for value in reversed(rate_grid_rps)
                if value < min(one_unstable) and (ProbeTopology.GPU0_ONLY, value) not in tested
            )
            if not lower:
                return NextProbe(
                    status="blocked",
                    reason="rate grid has no lower untested GPU0 operating point",
                )
            # Work downward from the lowest measured overload. This avoids an
            # exhaustive sweep while allowing more than one bad prior point.
            rate = lower[0]
        return NextProbe(
            status="probe",
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=rate,
            reason="locate one clearly sustainable GPU0-only operating point",
        )
    if not one_unstable:
        stable_peak = max(one_stable)
        preferred = round(1.25 * stable_peak, 3)
        higher = tuple(
            value
            for value in rate_grid_rps
            if value > stable_peak and (ProbeTopology.GPU0_ONLY, value) not in tested
        )
        if not higher:
            # The v9 diagnostic establishes 96 rps as an extreme invalid load,
            # but it is not itself a valid capacity point.  When the configured
            # low-rate grid remains entirely sustainable, continue the real
            # measurement geometrically instead of wasting the remaining
            # bounded probes or promoting the diagnostic result.  The 96-rps
            # cap prevents an unbounded offered-load escalation.
            extrapolated = round(min(96.0, preferred), 3)
            if extrapolated <= stable_peak or (ProbeTopology.GPU0_ONLY, extrapolated) in tested:
                return NextProbe(
                    status="blocked",
                    reason="bounded 96-rps prior cap has no untested GPU0 overload point",
                )
            return NextProbe(
                status="probe",
                topology=ProbeTopology.GPU0_ONLY,
                rate_rps=extrapolated,
                reason="continue above the sustainable grid peak within the 96-rps prior cap",
            )
        return NextProbe(
            status="probe",
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=preferred if preferred in higher else higher[0],
            reason="locate one clearly unsustainable GPU0-only operating point",
        )
    lambda_1 = max(one_stable)
    eligible_spikes = tuple(value for value in one_unstable if 1.15 <= value / lambda_1 <= 1.30)
    if not eligible_spikes:
        between = tuple(
            value
            for value in rate_grid_rps
            if 1.15 <= value / lambda_1 <= 1.30 and (ProbeTopology.GPU0_ONLY, value) not in tested
        )
        if between:
            return NextProbe(
                status="probe",
                topology=ProbeTopology.GPU0_ONLY,
                rate_rps=between[0],
                reason="test overload at a permitted v10 spike multiplier",
            )
        fractional = round((1.20 * lambda_1) * 2.0) / 2.0
        fractional_ratio = fractional / lambda_1
        if 1.15 <= fractional_ratio <= 1.30 and (ProbeTopology.GPU0_ONLY, fractional) not in tested:
            return NextProbe(
                status="probe",
                topology=ProbeTopology.GPU0_ONLY,
                rate_rps=fractional,
                reason="test fractional GPU0 overload inside the permitted spike interval",
            )
        return NextProbe(
            status="blocked",
            reason="one-GPU bracket has no measured overload in the 1.15x-1.30x interval",
        )
    candidate_spike = eligible_spikes[0]
    two_rates = tuple(
        value
        for value in rate_grid_rps
        if value > candidate_spike
        and candidate_spike / value <= 0.80
        and (ProbeTopology.TWO_GPU_ROUND_ROBIN, value) not in tested
    )
    if not two_rates:
        fractional = round((candidate_spike / 0.75) * 2.0) / 2.0
        if (
            fractional > candidate_spike
            and candidate_spike / fractional <= 0.80
            and (ProbeTopology.TWO_GPU_ROUND_ROBIN, fractional) not in tested
        ):
            return NextProbe(
                status="probe",
                topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
                rate_rps=fractional,
                reason="test fractional two-GPU rate with at least 20% reserve headroom",
            )
        return NextProbe(
            status="blocked",
            reason="no untested two-GPU point provides the preferred 20% reserve headroom",
        )
    target = candidate_spike / 0.75
    rate = min(two_rates, key=lambda value: (abs(value - target), value))
    return NextProbe(
        status="probe",
        topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
        rate_rps=rate,
        reason="verify sustainable two-GPU excess capacity with preferred reserve headroom",
    )


def early_stop_reason(result: CapacityProbeResult, *, elapsed_seconds: float) -> str | None:
    """Return a conservative early-stop reason after the ten-second evidence floor."""

    if result.verdict == ProbeVerdict.INCONCLUSIVE:
        return None
    if elapsed_seconds >= 10.0 and result.verdict == ProbeVerdict.SUSTAINABLE:
        return "all empirical sustainability gates passed after the minimum window"
    if result.maximum_queue_depth >= 64:
        return "queue explosion threshold reached"
    if result.p95_ttft_seconds is not None and result.p95_ttft_seconds >= 4.0:
        return "p95 TTFT exceeded twice the serving SLO"
    if (
        elapsed_seconds >= 5.0
        and result.queue_persistent_positive_drift
        and not result.throughput_pass
    ):
        return "queue drift and throughput deficit jointly establish overload"
    return None


__all__ = [
    "DEFAULT_RATE_GRID_RPS",
    "CapacityCalibrationBounds",
    "CapacityProbePlan",
    "CapacityProbeRaw",
    "CapacityProbeResult",
    "CapacityRequestObservation",
    "GlobalArrival",
    "NextProbe",
    "ProbeTopology",
    "ProbeVerdict",
    "SpikeSelection",
    "build_probe_plan",
    "choose_spike",
    "early_stop_reason",
    "evaluate_probe",
    "recommend_next_probe",
]
