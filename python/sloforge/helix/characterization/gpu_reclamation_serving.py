"""Deterministic serving-spike generation and SLO measurement for Experiment 004.

The generator in this module creates a request *plan*.  It does not sleep, send
requests, or claim that locally generated timestamps are hardware evidence.  A
GPU experiment client executes the plan and supplies :class:`ServingObservation`
records captured from monotonic clocks.

Intervals use half-open ``[start_ns, end_ns)`` bounds.  TTFT and waiting-time
statistics are attributed by request arrival.  Token and request throughput are
attributed by the time at which tokens and completions actually occur.  This
distinction is important at reclamation phase boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NS_PER_SECOND = 1_000_000_000
MIN_P99_SAMPLE_COUNT = 100
MAX_SERVING_REQUESTS = 1_000_000

Identifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$"),
]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class ServingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class WeightedTokenDistribution(ServingModel):
    """A small, explicit discrete distribution with integer weights."""

    values: tuple[int, ...] = Field(min_length=1, max_length=1024)
    weights: tuple[int, ...] = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        if len(self.values) != len(self.weights):
            raise ValueError("distribution values and weights must have equal length")
        if any(value <= 0 for value in self.values):
            raise ValueError("token counts must be positive")
        if len(set(self.values)) != len(self.values):
            raise ValueError("distribution token counts must be unique")
        if any(weight <= 0 for weight in self.weights):
            raise ValueError("distribution weights must be positive")
        return self

    def select(self, sample: int) -> int:
        target = sample % sum(self.weights)
        cumulative = 0
        for value, weight in zip(self.values, self.weights, strict=True):
            cumulative += weight
            if target < cumulative:
                return value
        raise AssertionError("validated distribution failed to select a value")


class ArrivalPhase(ServingModel):
    """A deterministic constant-cadence arrival phase relative to workload start."""

    name: Identifier
    start_offset_ns: int = Field(ge=0)
    end_offset_ns: int = Field(gt=0)
    interarrival_ns: int = Field(gt=0)
    first_arrival_offset_ns: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.end_offset_ns <= self.start_offset_ns:
            raise ValueError("arrival phase end must be after its start")
        if self.first_arrival_offset_ns >= self.end_offset_ns - self.start_offset_ns:
            raise ValueError("first arrival must occur within the arrival phase")
        return self


def _default_prompt_distribution() -> WeightedTokenDistribution:
    return WeightedTokenDistribution(values=(128, 256, 512), weights=(2, 5, 1))


def _default_output_distribution() -> WeightedTokenDistribution:
    return WeightedTokenDistribution(values=(16, 32, 64), weights=(2, 5, 1))


class ServingSpikeConfig(ServingModel):
    schema_version: Literal["sloforge.branchfabric.serving-spike-config/v1"] = (
        "sloforge.branchfabric.serving-spike-config/v1"
    )
    seed: int = Field(ge=0, le=2**64 - 1)
    control_phase: Identifier
    spike_phase: Identifier
    phases: tuple[ArrivalPhase, ...] = Field(min_length=2, max_length=128)
    prompt_tokens: WeightedTokenDistribution = Field(default_factory=_default_prompt_distribution)
    output_tokens: WeightedTokenDistribution = Field(default_factory=_default_output_distribution)
    request_id_prefix: Identifier = "exp004-serving"

    @model_validator(mode="after")
    def validate_phases(self) -> Self:
        names = tuple(phase.name for phase in self.phases)
        if len(set(names)) != len(names):
            raise ValueError("arrival phase names must be unique")
        if self.control_phase not in names or self.spike_phase not in names:
            raise ValueError("control_phase and spike_phase must name declared phases")
        if self.control_phase == self.spike_phase:
            raise ValueError("control and spike phases must differ")
        previous_end = -1
        for phase in self.phases:
            if phase.start_offset_ns < previous_end:
                raise ValueError("arrival phases must be ordered and non-overlapping")
            previous_end = phase.end_offset_ns
        return self


class ServingRequest(ServingModel):
    sequence: int = Field(ge=0, lt=MAX_SERVING_REQUESTS)
    request_id: Identifier
    phase: Identifier
    arrival_ns: int = Field(ge=0)
    prompt_tokens: int = Field(gt=0)
    requested_output_tokens: int = Field(gt=0)
    request_class: Literal["production-like-serving"] = "production-like-serving"


class ServingWorkload(ServingModel):
    schema_version: Literal["sloforge.branchfabric.serving-workload/v1"] = (
        "sloforge.branchfabric.serving-workload/v1"
    )
    workload_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_ns: int = Field(ge=0)
    end_ns: int = Field(gt=0)
    config: ServingSpikeConfig
    requests: tuple[ServingRequest, ...] = Field(max_length=MAX_SERVING_REQUESTS)

    @model_validator(mode="after")
    def validate_requests(self) -> Self:
        if self.end_ns <= self.start_ns:
            raise ValueError("workload end must be after its start")
        if len({request.request_id for request in self.requests}) != len(self.requests):
            raise ValueError("serving request IDs must be unique")
        arrivals = tuple(request.arrival_ns for request in self.requests)
        if arrivals != tuple(sorted(arrivals)):
            raise ValueError("serving requests must be ordered by arrival")
        if any(not self.start_ns <= arrival < self.end_ns for arrival in arrivals):
            raise ValueError("every request arrival must fall within the workload")
        return self


def _seeded_word(seed: int, sequence: int, dimension: str) -> int:
    payload = f"sloforge-exp004:{seed}:{sequence}:{dimension}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _workload_digest(config: ServingSpikeConfig, start_ns: int) -> str:
    canonical = json.dumps(
        {"config": config.model_dump(mode="json"), "start_ns": start_ns},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def generate_serving_spike(config: ServingSpikeConfig, *, start_ns: int = 0) -> ServingWorkload:
    """Generate a replayable request plan without consulting wall-clock state."""

    if start_ns < 0:
        raise ValueError("workload start_ns must be non-negative")
    workload_id = _workload_digest(config, start_ns)
    requests: list[ServingRequest] = []
    sequence = 0
    for phase in config.phases:
        relative_ns = phase.start_offset_ns + phase.first_arrival_offset_ns
        while relative_ns < phase.end_offset_ns:
            if sequence >= MAX_SERVING_REQUESTS:
                raise ValueError(f"serving workload exceeds {MAX_SERVING_REQUESTS} requests")
            requests.append(
                ServingRequest(
                    sequence=sequence,
                    request_id=f"{config.request_id_prefix}.{config.seed}.{sequence:06d}",
                    phase=phase.name,
                    arrival_ns=start_ns + relative_ns,
                    prompt_tokens=config.prompt_tokens.select(
                        _seeded_word(config.seed, sequence, "prompt")
                    ),
                    requested_output_tokens=config.output_tokens.select(
                        _seeded_word(config.seed, sequence, "output")
                    ),
                )
            )
            sequence += 1
            relative_ns += phase.interarrival_ns
    return ServingWorkload(
        workload_id=workload_id,
        start_ns=start_ns,
        end_ns=start_ns + max(phase.end_offset_ns for phase in config.phases),
        config=config,
        requests=tuple(requests),
    )


def workload_phase_intervals(workload: ServingWorkload) -> tuple[PhaseInterval, ...]:
    """Return absolute measurement intervals for the workload's arrival phases."""

    return tuple(
        PhaseInterval(
            name=phase.name,
            start_ns=workload.start_ns + phase.start_offset_ns,
            end_ns=workload.start_ns + phase.end_offset_ns,
        )
        for phase in workload.config.phases
    )


class ObservationOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServingObservation(ServingModel):
    """Monotonic timestamps observed while executing one planned request."""

    request_id: Identifier
    arrival_ns: int = Field(ge=0)
    service_start_ns: int | None = Field(default=None, ge=0)
    token_timestamps_ns: tuple[int, ...] = Field(default=(), max_length=1_000_000)
    completion_ns: int = Field(ge=0)
    outcome: ObservationOutcome
    device: Identifier

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        if self.completion_ns < self.arrival_ns:
            raise ValueError("request completion cannot precede arrival")
        if self.service_start_ns is not None:
            if self.service_start_ns < self.arrival_ns:
                raise ValueError("request service cannot begin before arrival")
            if self.completion_ns < self.service_start_ns:
                raise ValueError("request completion cannot precede service start")
        if any(timestamp < self.arrival_ns for timestamp in self.token_timestamps_ns):
            raise ValueError("token timestamps cannot precede request arrival")
        if any(
            right < left
            for left, right in zip(
                self.token_timestamps_ns, self.token_timestamps_ns[1:], strict=False
            )
        ):
            raise ValueError("token timestamps must be monotonic")
        if self.token_timestamps_ns:
            if self.service_start_ns is None:
                raise ValueError("token observations require a service start")
            if self.token_timestamps_ns[0] < self.service_start_ns:
                raise ValueError("tokens cannot precede service start")
            if self.token_timestamps_ns[-1] > self.completion_ns:
                raise ValueError("tokens cannot follow request completion")
        if self.outcome is ObservationOutcome.COMPLETED and not self.token_timestamps_ns:
            raise ValueError("a completed request must produce at least one token")
        return self


class PhaseInterval(ServingModel):
    name: Identifier
    start_ns: int = Field(ge=0)
    end_ns: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end_ns <= self.start_ns:
            raise ValueError("phase interval end must be after its start")
        return self


class ServingMeasurementPlan(ServingModel):
    schema_version: Literal["sloforge.branchfabric.serving-measurement-plan/v1"] = (
        "sloforge.branchfabric.serving-measurement-plan/v1"
    )
    intervals: tuple[PhaseInterval, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_intervals(self) -> Self:
        names = tuple(interval.name for interval in self.intervals)
        if len(set(names)) != len(names):
            raise ValueError("measurement interval names must be unique")
        previous_end = -1
        for interval in self.intervals:
            if interval.start_ns < previous_end:
                raise ValueError("measurement intervals must be ordered and non-overlapping")
            previous_end = interval.end_ns
        return self


class DistributionSummary(ServingModel):
    unit: Literal["nanoseconds"]
    sample_count: int = Field(ge=0)
    minimum: FiniteFloat | None = None
    mean: FiniteFloat | None = None
    median: FiniteFloat | None = None
    p95: FiniteFloat | None = None
    p99: FiniteFloat | None = None
    maximum: FiniteFloat | None = None
    percentile_method: Literal["Hyndman-Fan type 7"]
    p99_minimum_sample_count: Literal[100] = 100
    p99_status: Literal["reported", "insufficient-samples", "no-samples"]

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        common = (self.minimum, self.mean, self.median, self.p95, self.maximum)
        if self.sample_count == 0:
            if any(value is not None for value in (*common, self.p99)):
                raise ValueError("empty distributions cannot report statistics")
            if self.p99_status != "no-samples":
                raise ValueError("empty distributions require no-samples p99 status")
        else:
            if any(value is None for value in common):
                raise ValueError("non-empty distributions require non-p99 statistics")
            expected = (
                "reported" if self.sample_count >= MIN_P99_SAMPLE_COUNT else "insufficient-samples"
            )
            if self.p99_status != expected:
                raise ValueError("p99 status does not match distribution sample count")
            if (self.p99 is not None) != (expected == "reported"):
                raise ValueError("p99 value does not match its availability status")
        return self


def _quantile(values: tuple[int, ...], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_latency(values: tuple[int, ...]) -> DistributionSummary:
    """Summarize latency samples while suppressing unsupported small-N p99s."""

    if not values:
        return DistributionSummary(
            unit="nanoseconds",
            sample_count=0,
            percentile_method="Hyndman-Fan type 7",
            p99_status="no-samples",
        )
    if any(value < 0 for value in values):
        raise ValueError("latency samples must be non-negative")
    return DistributionSummary(
        unit="nanoseconds",
        sample_count=len(values),
        minimum=float(min(values)),
        mean=statistics.fmean(values),
        median=float(statistics.median(values)),
        p95=_quantile(values, 0.95),
        p99=_quantile(values, 0.99) if len(values) >= MIN_P99_SAMPLE_COUNT else None,
        maximum=float(max(values)),
        percentile_method="Hyndman-Fan type 7",
        p99_status=("reported" if len(values) >= MIN_P99_SAMPLE_COUNT else "insufficient-samples"),
    )


class IntervalMetrics(ServingModel):
    interval: PhaseInterval
    offered_requests: int = Field(ge=0)
    observed_requests: int = Field(ge=0)
    completed_requests: int = Field(ge=0)
    failed_requests: int = Field(ge=0)
    cancelled_requests: int = Field(ge=0)
    emitted_tokens: int = Field(ge=0)
    completions_in_interval: int = Field(ge=0)
    completion_fraction: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    mean_queue_depth: FiniteFloat = Field(ge=0.0)
    max_queue_depth: int = Field(ge=0)
    output_token_throughput_per_second: FiniteFloat = Field(ge=0.0)
    request_throughput_per_second: FiniteFloat = Field(ge=0.0)
    waiting_time: DistributionSummary
    ttft: DistributionSummary
    inter_token_latency: DistributionSummary


def _queue_metrics(
    requests: tuple[ServingRequest, ...],
    observations: dict[str, ServingObservation],
    interval: PhaseInterval,
) -> tuple[float, int]:
    changes: dict[int, int] = {}
    for request in requests:
        observation = observations[request.request_id]
        changes[request.arrival_ns] = changes.get(request.arrival_ns, 0) + 1
        leaves_queue_ns = (
            observation.service_start_ns
            if observation.service_start_ns is not None
            else observation.completion_ns
        )
        changes[leaves_queue_ns] = changes.get(leaves_queue_ns, 0) - 1

    depth = sum(delta for timestamp, delta in changes.items() if timestamp < interval.start_ns)
    if depth < 0:
        raise ValueError("queue event accounting produced a negative initial depth")
    maximum = depth
    area_ns = 0
    cursor_ns = interval.start_ns
    for timestamp in sorted(
        timestamp for timestamp in changes if interval.start_ns <= timestamp < interval.end_ns
    ):
        area_ns += depth * (timestamp - cursor_ns)
        depth += changes[timestamp]
        if depth < 0:
            raise ValueError("queue event accounting produced a negative depth")
        maximum = max(maximum, depth)
        cursor_ns = timestamp
    area_ns += depth * (interval.end_ns - cursor_ns)
    return area_ns / (interval.end_ns - interval.start_ns), maximum


def _validate_observations(
    workload: ServingWorkload,
    observations: tuple[ServingObservation, ...],
    *,
    require_complete_accounting: bool,
) -> dict[str, ServingObservation]:
    by_request = {observation.request_id: observation for observation in observations}
    if len(by_request) != len(observations):
        raise ValueError("serving observations must have unique request IDs")
    planned = {request.request_id: request for request in workload.requests}
    unknown = set(by_request) - set(planned)
    if unknown:
        raise ValueError(f"observations contain unknown request IDs: {sorted(unknown)!r}")
    missing = set(planned) - set(by_request)
    if require_complete_accounting and missing:
        raise ValueError(f"observations are missing planned request IDs: {sorted(missing)!r}")
    if missing:
        # Queue depth cannot be measured without knowing when an unobserved request
        # left the queue, so partial accounting is not accepted by this analyzer.
        raise ValueError("queue measurement requires an observation for every planned request")
    for request_id, observation in by_request.items():
        request = planned[request_id]
        if observation.arrival_ns != request.arrival_ns:
            raise ValueError(f"observation arrival does not match plan for {request_id}")
        if len(observation.token_timestamps_ns) > request.requested_output_tokens:
            raise ValueError(f"observation emitted too many tokens for {request_id}")
    return by_request


def _measure_interval(
    workload: ServingWorkload,
    observations: dict[str, ServingObservation],
    interval: PhaseInterval,
) -> IntervalMetrics:
    cohort = tuple(
        request
        for request in workload.requests
        if interval.start_ns <= request.arrival_ns < interval.end_ns
    )
    cohort_observations = tuple(observations[request.request_id] for request in cohort)
    waiting = tuple(
        observation.service_start_ns - observation.arrival_ns
        for observation in cohort_observations
        if observation.service_start_ns is not None
    )
    ttft = tuple(
        observation.token_timestamps_ns[0] - observation.arrival_ns
        for observation in cohort_observations
        if observation.token_timestamps_ns
    )
    inter_token: list[int] = []
    emitted_tokens = 0
    completions = 0
    for observation in observations.values():
        emitted_tokens += sum(
            interval.start_ns <= timestamp < interval.end_ns
            for timestamp in observation.token_timestamps_ns
        )
        completions += interval.start_ns <= observation.completion_ns < interval.end_ns
        for previous, current in zip(
            observation.token_timestamps_ns, observation.token_timestamps_ns[1:], strict=False
        ):
            if interval.start_ns <= current < interval.end_ns:
                inter_token.append(current - previous)

    completed = sum(
        observation.outcome is ObservationOutcome.COMPLETED for observation in cohort_observations
    )
    failed = sum(
        observation.outcome is ObservationOutcome.FAILED for observation in cohort_observations
    )
    cancelled = sum(
        observation.outcome is ObservationOutcome.CANCELLED for observation in cohort_observations
    )
    mean_queue, max_queue = _queue_metrics(workload.requests, observations, interval)
    duration_seconds = (interval.end_ns - interval.start_ns) / NS_PER_SECOND
    return IntervalMetrics(
        interval=interval,
        offered_requests=len(cohort),
        observed_requests=len(cohort_observations),
        completed_requests=completed,
        failed_requests=failed,
        cancelled_requests=cancelled,
        emitted_tokens=emitted_tokens,
        completions_in_interval=completions,
        completion_fraction=completed / len(cohort) if cohort else None,
        mean_queue_depth=mean_queue,
        max_queue_depth=max_queue,
        output_token_throughput_per_second=emitted_tokens / duration_seconds,
        request_throughput_per_second=completions / duration_seconds,
        waiting_time=summarize_latency(waiting),
        ttft=summarize_latency(ttft),
        inter_token_latency=summarize_latency(tuple(inter_token)),
    )


class ServingMeasurement(ServingModel):
    schema_version: Literal["sloforge.branchfabric.serving-measurement/v1"] = (
        "sloforge.branchfabric.serving-measurement/v1"
    )
    workload_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_count: int = Field(ge=0)
    complete_request_accounting: bool
    intervals: tuple[IntervalMetrics, ...] = Field(min_length=1)


def measure_serving_intervals(
    workload: ServingWorkload,
    observations: tuple[ServingObservation, ...],
    plan: ServingMeasurementPlan,
    *,
    require_complete_accounting: bool = True,
) -> ServingMeasurement:
    """Measure queue, latency, throughput, and outcomes for strict intervals."""

    by_request = _validate_observations(
        workload,
        observations,
        require_complete_accounting=require_complete_accounting,
    )
    metrics = tuple(
        _measure_interval(workload, by_request, interval) for interval in plan.intervals
    )
    return ServingMeasurement(
        workload_id=workload.workload_id,
        observation_count=len(observations),
        complete_request_accounting=len(observations) == len(workload.requests),
        intervals=metrics,
    )


class ServingSLO(ServingModel):
    schema_version: Literal["sloforge.branchfabric.serving-slo/v1"] = (
        "sloforge.branchfabric.serving-slo/v1"
    )
    maximum_p95_ttft_ns: int | None = Field(default=None, gt=0)
    maximum_p95_inter_token_latency_ns: int | None = Field(default=None, gt=0)
    maximum_mean_queue_depth: FiniteFloat | None = Field(default=None, ge=0.0)
    maximum_queue_depth: int | None = Field(default=None, ge=0)
    minimum_output_token_throughput_per_second: FiniteFloat | None = Field(default=None, ge=0.0)
    minimum_completion_fraction: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def has_constraint(self) -> Self:
        if all(
            value is None
            for value in (
                self.maximum_p95_ttft_ns,
                self.maximum_p95_inter_token_latency_ns,
                self.maximum_mean_queue_depth,
                self.maximum_queue_depth,
                self.minimum_output_token_throughput_per_second,
                self.minimum_completion_fraction,
            )
        ):
            raise ValueError("serving SLO requires at least one constraint")
        return self


class SLOCheck(ServingModel):
    metric: Identifier
    comparator: Literal["<=", ">="]
    threshold: FiniteFloat
    observed: FiniteFloat | None = None
    passed: bool
    availability: Literal["available", "no-samples"]


class IntervalSLOEvaluation(ServingModel):
    interval: PhaseInterval
    satisfied: bool
    checks: tuple[SLOCheck, ...] = Field(min_length=1)


def _slo_check(
    *, metric: str, comparator: Literal["<=", ">="], threshold: float, observed: float | None
) -> SLOCheck:
    available = observed is not None
    passed = False
    if observed is not None:
        passed = observed <= threshold if comparator == "<=" else observed >= threshold
    return SLOCheck(
        metric=metric,
        comparator=comparator,
        threshold=threshold,
        observed=observed,
        passed=passed,
        availability="available" if available else "no-samples",
    )


def evaluate_serving_slo(metrics: IntervalMetrics, slo: ServingSLO) -> IntervalSLOEvaluation:
    """Evaluate declared SLO constraints; unavailable samples fail closed."""

    checks: list[SLOCheck] = []
    if slo.maximum_p95_ttft_ns is not None:
        checks.append(
            _slo_check(
                metric="p95_ttft_ns",
                comparator="<=",
                threshold=float(slo.maximum_p95_ttft_ns),
                observed=metrics.ttft.p95,
            )
        )
    if slo.maximum_p95_inter_token_latency_ns is not None:
        checks.append(
            _slo_check(
                metric="p95_inter_token_latency_ns",
                comparator="<=",
                threshold=float(slo.maximum_p95_inter_token_latency_ns),
                observed=metrics.inter_token_latency.p95,
            )
        )
    if slo.maximum_mean_queue_depth is not None:
        checks.append(
            _slo_check(
                metric="mean_queue_depth",
                comparator="<=",
                threshold=slo.maximum_mean_queue_depth,
                observed=metrics.mean_queue_depth,
            )
        )
    if slo.maximum_queue_depth is not None:
        checks.append(
            _slo_check(
                metric="max_queue_depth",
                comparator="<=",
                threshold=float(slo.maximum_queue_depth),
                observed=float(metrics.max_queue_depth),
            )
        )
    if slo.minimum_output_token_throughput_per_second is not None:
        checks.append(
            _slo_check(
                metric="output_token_throughput_per_second",
                comparator=">=",
                threshold=slo.minimum_output_token_throughput_per_second,
                observed=metrics.output_token_throughput_per_second,
            )
        )
    if slo.minimum_completion_fraction is not None:
        checks.append(
            _slo_check(
                metric="completion_fraction",
                comparator=">=",
                threshold=slo.minimum_completion_fraction,
                observed=metrics.completion_fraction,
            )
        )
    return IntervalSLOEvaluation(
        interval=metrics.interval,
        satisfied=all(check.passed for check in checks),
        checks=tuple(checks),
    )


class SLOStabilityConfig(ServingModel):
    evaluation_window_ns: int = Field(gt=0)
    stability_window_ns: int = Field(gt=0)

    @model_validator(mode="after")
    def exact_window_count(self) -> Self:
        if self.stability_window_ns % self.evaluation_window_ns:
            raise ValueError("stability window must be a multiple of evaluation window")
        return self


class SLORestoration(ServingModel):
    schema_version: Literal["sloforge.branchfabric.slo-restoration/v1"] = (
        "sloforge.branchfabric.slo-restoration/v1"
    )
    trigger_ns: int = Field(ge=0)
    measurement_end_ns: int = Field(gt=0)
    evaluation_window_ns: int = Field(gt=0)
    stability_window_ns: int = Field(gt=0)
    evaluations: tuple[IntervalSLOEvaluation, ...]
    stable_window_confirmed: bool
    slo_reentered_at_ns: int | None = Field(default=None, ge=0)
    restored_at_ns: int | None = Field(default=None, ge=0)
    restoration_latency_ns: int | None = Field(default=None, ge=0)
    status: Literal["restored", "not-restored"]


def find_serving_slo_restoration(
    workload: ServingWorkload,
    observations: tuple[ServingObservation, ...],
    *,
    trigger_ns: int,
    measurement_end_ns: int,
    slo: ServingSLO,
    stability: SLOStabilityConfig,
) -> SLORestoration:
    """Find the first confirmed consecutive sequence of passing SLO windows."""

    if measurement_end_ns <= trigger_ns:
        raise ValueError("SLO measurement end must be after its trigger")
    intervals: list[PhaseInterval] = []
    cursor_ns = trigger_ns
    index = 0
    while cursor_ns + stability.evaluation_window_ns <= measurement_end_ns:
        intervals.append(
            PhaseInterval(
                name=f"slo-window-{index:06d}",
                start_ns=cursor_ns,
                end_ns=cursor_ns + stability.evaluation_window_ns,
            )
        )
        cursor_ns += stability.evaluation_window_ns
        index += 1
    if not intervals:
        raise ValueError("measurement range contains no complete SLO evaluation window")
    measurement = measure_serving_intervals(
        workload,
        observations,
        ServingMeasurementPlan(intervals=tuple(intervals)),
    )
    evaluations = tuple(evaluate_serving_slo(metrics, slo) for metrics in measurement.intervals)
    required_windows = stability.stability_window_ns // stability.evaluation_window_ns
    consecutive = 0
    for index, evaluation in enumerate(evaluations):
        consecutive = consecutive + 1 if evaluation.satisfied else 0
        if consecutive >= required_windows:
            first = evaluations[index - required_windows + 1].interval.start_ns
            confirmed = evaluation.interval.end_ns
            return SLORestoration(
                trigger_ns=trigger_ns,
                measurement_end_ns=measurement_end_ns,
                evaluation_window_ns=stability.evaluation_window_ns,
                stability_window_ns=stability.stability_window_ns,
                evaluations=evaluations,
                stable_window_confirmed=True,
                slo_reentered_at_ns=first,
                restored_at_ns=confirmed,
                restoration_latency_ns=confirmed - trigger_ns,
                status="restored",
            )
    return SLORestoration(
        trigger_ns=trigger_ns,
        measurement_end_ns=measurement_end_ns,
        evaluation_window_ns=stability.evaluation_window_ns,
        stability_window_ns=stability.stability_window_ns,
        evaluations=evaluations,
        stable_window_confirmed=False,
        status="not-restored",
    )


class MetricDirection(StrEnum):
    HIGHER_IS_WORSE = "higher-is-worse"
    LOWER_IS_WORSE = "lower-is-worse"


class InterferenceMetric(ServingModel):
    metric: Identifier
    direction: MetricDirection
    control_value: FiniteFloat | None = None
    affected_value: FiniteFloat | None = None
    absolute_degradation: FiniteFloat | None = None
    interference_fraction: FiniteFloat | None = None
    availability: Literal["available", "missing-metric", "zero-control"]


def _interference_metric(
    metric: str,
    direction: MetricDirection,
    control: float | None,
    affected: float | None,
) -> InterferenceMetric:
    if control is None or affected is None:
        return InterferenceMetric(
            metric=metric,
            direction=direction,
            control_value=control,
            affected_value=affected,
            availability="missing-metric",
        )
    degradation = affected - control
    if direction is MetricDirection.LOWER_IS_WORSE:
        degradation = control - affected
    if control == 0.0:
        return InterferenceMetric(
            metric=metric,
            direction=direction,
            control_value=control,
            affected_value=affected,
            absolute_degradation=degradation,
            availability="zero-control",
        )
    return InterferenceMetric(
        metric=metric,
        direction=direction,
        control_value=control,
        affected_value=affected,
        absolute_degradation=degradation,
        interference_fraction=degradation / abs(control),
        availability="available",
    )


class ServingInterference(ServingModel):
    schema_version: Literal["sloforge.branchfabric.serving-interference/v1"] = (
        "sloforge.branchfabric.serving-interference/v1"
    )
    control_interval: Identifier
    affected_interval: Identifier
    metrics: tuple[InterferenceMetric, ...] = Field(min_length=1)
    maximum_positive_interference_fraction: FiniteFloat = Field(ge=0.0)
    maximum_positive_metric: Identifier | None = None


def calculate_serving_interference(
    control: IntervalMetrics, affected: IntervalMetrics
) -> ServingInterference:
    """Calculate metric-specific degradation relative to a control interval.

    Positive fractions are degradations and negative fractions are improvements.
    A zero control value is reported explicitly rather than dividing by zero.
    """

    metrics = (
        _interference_metric(
            "p95_ttft_ns",
            MetricDirection.HIGHER_IS_WORSE,
            control.ttft.p95,
            affected.ttft.p95,
        ),
        _interference_metric(
            "p95_inter_token_latency_ns",
            MetricDirection.HIGHER_IS_WORSE,
            control.inter_token_latency.p95,
            affected.inter_token_latency.p95,
        ),
        _interference_metric(
            "mean_queue_depth",
            MetricDirection.HIGHER_IS_WORSE,
            control.mean_queue_depth,
            affected.mean_queue_depth,
        ),
        _interference_metric(
            "max_queue_depth",
            MetricDirection.HIGHER_IS_WORSE,
            float(control.max_queue_depth),
            float(affected.max_queue_depth),
        ),
        _interference_metric(
            "output_token_throughput_per_second",
            MetricDirection.LOWER_IS_WORSE,
            control.output_token_throughput_per_second,
            affected.output_token_throughput_per_second,
        ),
        _interference_metric(
            "request_throughput_per_second",
            MetricDirection.LOWER_IS_WORSE,
            control.request_throughput_per_second,
            affected.request_throughput_per_second,
        ),
    )
    positive = tuple(
        metric
        for metric in metrics
        if metric.interference_fraction is not None and metric.interference_fraction > 0.0
    )
    worst = max(positive, key=lambda item: item.interference_fraction or 0.0, default=None)
    return ServingInterference(
        control_interval=control.interval.name,
        affected_interval=affected.interval.name,
        metrics=metrics,
        maximum_positive_interference_fraction=(
            worst.interference_fraction
            if worst and worst.interference_fraction is not None
            else 0.0
        ),
        maximum_positive_metric=worst.metric if worst else None,
    )


def experiment_004_serving_spike_config(
    *,
    seed: int,
    control_duration_ns: int,
    spike_duration_ns: int,
    recovery_duration_ns: int,
    control_interarrival_ns: int,
    spike_interarrival_ns: int,
) -> ServingSpikeConfig:
    """Build the standard control/spike/recovery Experiment 004 request plan."""

    if min(control_duration_ns, spike_duration_ns, recovery_duration_ns) <= 0:
        raise ValueError("all Experiment 004 serving phase durations must be positive")
    control_end = control_duration_ns
    spike_end = control_end + spike_duration_ns
    return ServingSpikeConfig(
        seed=seed,
        control_phase="control",
        spike_phase="spike",
        phases=(
            ArrivalPhase(
                name="control",
                start_offset_ns=0,
                end_offset_ns=control_end,
                interarrival_ns=control_interarrival_ns,
            ),
            ArrivalPhase(
                name="spike",
                start_offset_ns=control_end,
                end_offset_ns=spike_end,
                interarrival_ns=spike_interarrival_ns,
            ),
            ArrivalPhase(
                name="recovery",
                start_offset_ns=spike_end,
                end_offset_ns=spike_end + recovery_duration_ns,
                interarrival_ns=control_interarrival_ns,
            ),
        ),
    )


__all__ = [
    "MIN_P99_SAMPLE_COUNT",
    "NS_PER_SECOND",
    "ArrivalPhase",
    "DistributionSummary",
    "InterferenceMetric",
    "IntervalMetrics",
    "IntervalSLOEvaluation",
    "MetricDirection",
    "ObservationOutcome",
    "PhaseInterval",
    "SLOCheck",
    "SLORestoration",
    "SLOStabilityConfig",
    "ServingInterference",
    "ServingMeasurement",
    "ServingMeasurementPlan",
    "ServingObservation",
    "ServingRequest",
    "ServingSLO",
    "ServingSpikeConfig",
    "ServingWorkload",
    "WeightedTokenDistribution",
    "calculate_serving_interference",
    "evaluate_serving_slo",
    "experiment_004_serving_spike_config",
    "find_serving_slo_restoration",
    "generate_serving_spike",
    "measure_serving_intervals",
    "summarize_latency",
    "workload_phase_intervals",
]
