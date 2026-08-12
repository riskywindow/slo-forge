"""Fail-closed serving-plan primitives for Experiment 004 v10.

The v9 pilot used one independent arrival stream per worker.  That made the
configured aggregate rate increase when GPU1 joined and assigned work to GPU1
before it was available.  This module instead describes one global arrival
clock and changes only its routing policy at explicit causal boundaries.

The functions are CPU-only.  Runtime workers may execute their assigned shard,
but neither request timestamps nor validity claims are synthesized here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NS_PER_SECOND = 1_000_000_000
GPU0: Literal["gpu0"] = "gpu0"
GPU1: Literal["gpu1"] = "gpu1"
MAX_V10_REQUESTS = 20_000


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class RoutingEpoch(_StrictModel):
    """One end-exclusive interval on the global offered-load clock."""

    name: Literal["control", "gpu0-overload", "two-gpu-recovery", "restore-interference"]
    start_ns: int = Field(ge=0)
    end_ns: int = Field(gt=0)
    rate_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    route_cycle: tuple[Literal["gpu0", "gpu1"], ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def valid_epoch(self) -> Self:
        if self.end_ns <= self.start_ns:
            raise ValueError("routing epoch end must be after its start")
        return self


class RoutedServingRequest(_StrictModel):
    sequence: int = Field(ge=0)
    request_id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]{7,127}$")
    phase: Literal["control", "gpu0-overload", "two-gpu-recovery", "restore-interference"]
    scheduled_arrival_ns: int = Field(ge=0)
    planned_device: Literal["gpu0", "gpu1"]
    prompt_tokens: int = Field(gt=0)
    requested_output_tokens: Literal[64] = 64


class GlobalServingPlan(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-v10-serving-plan/v1"] = (
        "sloforge.branchfabric.experiment-004-v10-serving-plan/v1"
    )
    attempt_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{7,95}$")
    seed: int = Field(ge=0, lt=1 << 63)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    epochs: tuple[RoutingEpoch, RoutingEpoch, RoutingEpoch, RoutingEpoch]
    requests: tuple[RoutedServingRequest, ...] = Field(min_length=1, max_length=MAX_V10_REQUESTS)

    @model_validator(mode="after")
    def valid_plan(self) -> Self:
        expected_names = (
            "control",
            "gpu0-overload",
            "two-gpu-recovery",
            "restore-interference",
        )
        if tuple(epoch.name for epoch in self.epochs) != expected_names:
            raise ValueError("v10 routing epochs do not have the required order")
        if any(
            left.end_ns != right.start_ns
            for left, right in zip(self.epochs, self.epochs[1:], strict=False)
        ):
            raise ValueError("v10 routing epochs must be contiguous")
        if self.epochs[0].route_cycle != (GPU0,) or self.epochs[1].route_cycle != (GPU0,):
            raise ValueError("control and overload must route exclusively to GPU0")
        if set(self.epochs[2].route_cycle) != {GPU0, GPU1}:
            raise ValueError("the recovery epoch must route the one global stream to both GPUs")
        if self.epochs[3].route_cycle != (GPU0,):
            raise ValueError("restore interference must route exclusively to GPU0")
        if self.epochs[1].rate_per_second != self.epochs[2].rate_per_second:
            raise ValueError("GPU1 enable must not change the global spike offered rate")
        sequences = tuple(request.sequence for request in self.requests)
        if sequences != tuple(range(len(self.requests))):
            raise ValueError("global request sequences must be contiguous")
        ids = tuple(request.request_id for request in self.requests)
        if len(set(ids)) != len(ids):
            raise ValueError("global request IDs must be unique")
        arrivals = tuple(request.scheduled_arrival_ns for request in self.requests)
        if arrivals != tuple(sorted(arrivals)):
            raise ValueError("global request arrivals must be monotonic")
        control_arrivals = tuple(
            request.scheduled_arrival_ns for request in self.requests if request.phase == "control"
        )
        spike_arrivals = tuple(
            request.scheduled_arrival_ns
            for request in self.requests
            if request.phase in {"gpu0-overload", "two-gpu-recovery"}
        )
        restore_arrivals = tuple(
            request.scheduled_arrival_ns
            for request in self.requests
            if request.phase == "restore-interference"
        )
        if control_arrivals != _epoch_arrivals(self.epochs[0]):
            raise ValueError("control arrivals differ from the declared global clock")
        if spike_arrivals != _constant_rate_arrivals(
            start_ns=self.epochs[1].start_ns,
            end_ns=self.epochs[2].end_ns,
            rate_per_second=self.epochs[1].rate_per_second,
        ):
            raise ValueError("GPU1 enable changed the global spike clock")
        if restore_arrivals != _epoch_arrivals(self.epochs[3]):
            raise ValueError("restore arrivals differ from the declared global clock")
        for epoch in self.epochs:
            cohort = tuple(request for request in self.requests if request.phase == epoch.name)
            if not cohort:
                raise ValueError(f"routing epoch {epoch.name!r} has no requests")
            if any(
                not epoch.start_ns <= request.scheduled_arrival_ns < epoch.end_ns
                for request in cohort
            ):
                raise ValueError("request arrival falls outside its routing epoch")
            if any(
                request.planned_device != epoch.route_cycle[index % len(epoch.route_cycle)]
                for index, request in enumerate(cohort)
            ):
                raise ValueError("request route differs from its deterministic routing cycle")
        expected_digest = _plan_digest(
            attempt_id=self.attempt_id,
            seed=self.seed,
            epochs=self.epochs,
            requests=self.requests,
        )
        if self.plan_sha256 != expected_digest:
            raise ValueError("global serving plan digest is invalid")
        return self


class ExecutedServingRequest(_StrictModel):
    """Raw per-request evidence emitted by one serving worker."""

    request_id: str
    sequence: int = Field(ge=0)
    phase: Literal["control", "gpu0-overload", "two-gpu-recovery", "restore-interference"]
    device: Literal["gpu0", "gpu1"]
    scheduled_arrival_ns: int = Field(ge=0)
    admitted_ns: int = Field(ge=0)
    service_start_ns: int = Field(ge=0)
    first_token_ns: int = Field(ge=0)
    completed_ns: int = Field(ge=0)
    token_timestamps_ns: tuple[int, ...] = Field(min_length=64, max_length=64)
    output_token_ids: tuple[int, ...] = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def valid_timeline(self) -> Self:
        if not (
            self.scheduled_arrival_ns
            <= self.admitted_ns
            <= self.service_start_ns
            <= self.first_token_ns
            <= self.completed_ns
        ):
            raise ValueError("executed serving request timestamps are not causal")
        if self.token_timestamps_ns[0] != self.first_token_ns:
            raise ValueError("first-token scalar differs from token timestamps")
        if any(
            right < left
            for left, right in zip(
                self.token_timestamps_ns,
                self.token_timestamps_ns[1:],
                strict=False,
            )
        ):
            raise ValueError("token timestamps must be monotonic")
        if self.token_timestamps_ns[-1] > self.completed_ns:
            raise ValueError("tokens cannot be observed after completion")
        return self


class ServingExecutionAudit(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-v10-serving-audit/v1"] = (
        "sloforge.branchfabric.experiment-004-v10-serving-audit/v1"
    )
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_count: int = Field(gt=0)
    arrivals_monotonic: bool
    exact_output_length: bool
    routing_matches_plan: bool
    gpu0_only_before_enable: bool
    both_gpus_after_enable: bool
    gpu1_idle_during_restore: bool
    gpu0_restore_completions: int = Field(ge=0)
    gpu0_restore_emitted_tokens: int = Field(ge=0)
    gpu0_restore_ttft_samples: int = Field(ge=0)
    gpu0_active_during_restore: bool
    passed: bool

    @model_validator(mode="after")
    def fail_closed(self) -> Self:
        required = (
            self.arrivals_monotonic,
            self.exact_output_length,
            self.routing_matches_plan,
            self.gpu0_only_before_enable,
            self.both_gpus_after_enable,
            self.gpu1_idle_during_restore,
            self.gpu0_active_during_restore,
        )
        if self.passed != all(required):
            raise ValueError("serving execution audit pass flag is not fail-closed")
        return self


class ReclamationTriggerEvidence(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.reclamation-trigger-evidence/v1"] = (
        "sloforge.branchfabric.reclamation-trigger-evidence/v1"
    )
    window_start_ns: int = Field(ge=0)
    window_end_ns: int = Field(gt=0)
    offered_rate_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    completed_rate_per_second: float = Field(ge=0.0, allow_inf_nan=False)
    queue_depth_start: int = Field(ge=0)
    queue_depth_end: int = Field(ge=0)
    queue_depth_slope_per_second: float = Field(allow_inf_nan=False)
    p95_ttft_ns: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    trigger_rule: Literal[
        "fixed-overload-window-with-measured-overload",
        "ttft-and-queue-threshold",
    ]
    trigger_reason: tuple[str, ...] = ()
    overload_confirmed: bool

    @model_validator(mode="after")
    def fail_closed(self) -> Self:
        if self.window_end_ns <= self.window_start_ns:
            raise ValueError("trigger evidence window is empty")
        available = {
            "p95_ttft_above_slo": self.p95_ttft_ns is not None
            and self.p95_ttft_ns > 2 * NS_PER_SECOND,
            "queue_positive_drift": self.queue_depth_slope_per_second > 0.0,
            "offered_above_completed": (
                self.offered_rate_per_second > self.completed_rate_per_second
            ),
        }
        if any(reason not in available or not available[reason] for reason in self.trigger_reason):
            raise ValueError("trigger reason is not supported by the measured window")
        if self.overload_confirmed != bool(self.trigger_reason):
            raise ValueError("overload confirmation does not match trigger reasons")
        return self


class RecoveryStabilityWindow(_StrictModel):
    start_ns: int = Field(ge=0)
    end_ns: int = Field(gt=0)
    ttft_sample_count: int = Field(ge=0)
    p95_ttft_ns: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    queue_depth_end: int = Field(ge=0)
    passed: bool

    @model_validator(mode="after")
    def valid_window(self) -> Self:
        if self.end_ns <= self.start_ns:
            raise ValueError("recovery stability window is empty")
        return self


class ServingRecoveryEvidence(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.serving-recovery-evidence/v1"] = (
        "sloforge.branchfabric.serving-recovery-evidence/v1"
    )
    gpu1_first_useful_ns: int = Field(ge=0)
    restore_eligible_ns: int = Field(gt=0)
    offered_rate_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    completed_rate_per_second: float = Field(ge=0.0, allow_inf_nan=False)
    queue_depth_at_gpu1_first_useful: int = Field(ge=0)
    queue_depth_at_stability_start: int = Field(ge=0)
    queue_depth_at_restore_eligible: int = Field(ge=0)
    queue_depth_slope_per_second: float = Field(allow_inf_nan=False)
    recovery_queue_threshold: int = Field(ge=0)
    stability_window_ns: int = Field(gt=0)
    evaluation_window_ns: int = Field(gt=0)
    stability_windows: tuple[RecoveryStabilityWindow, ...] = Field(min_length=1)
    two_gpu_excess_capacity_pass: bool
    queue_drain_pass: bool
    slo_restoration_pass: bool
    pre_restore_stability_pass: bool
    restore_eligible: bool

    @model_validator(mode="after")
    def fail_closed(self) -> Self:
        if self.restore_eligible_ns <= self.gpu1_first_useful_ns:
            raise ValueError("restore eligibility must follow useful GPU1 capacity")
        if self.stability_window_ns % self.evaluation_window_ns:
            raise ValueError("stability duration must contain whole evaluation windows")
        required_window_count = self.stability_window_ns // self.evaluation_window_ns
        if len(self.stability_windows) != required_window_count:
            raise ValueError("recovery evidence has the wrong stability-window count")
        if self.two_gpu_excess_capacity_pass != (
            self.completed_rate_per_second > self.offered_rate_per_second
        ):
            raise ValueError("two-GPU excess-capacity flag disagrees with measured rates")
        if self.queue_drain_pass != (
            self.queue_depth_slope_per_second < 0.0
            and self.queue_depth_at_stability_start < self.queue_depth_at_gpu1_first_useful
        ):
            raise ValueError("queue-drain flag disagrees with measured outstanding depth")
        if self.slo_restoration_pass != all(
            window.p95_ttft_ns is not None and window.p95_ttft_ns <= 2 * NS_PER_SECOND
            for window in self.stability_windows
        ):
            raise ValueError("SLO-restoration flag disagrees with measured TTFT")
        if self.pre_restore_stability_pass != (
            self.queue_depth_at_stability_start <= self.recovery_queue_threshold
            and all(window.passed for window in self.stability_windows)
        ):
            raise ValueError("pre-restore stability flag disagrees with window evidence")
        required = (
            self.two_gpu_excess_capacity_pass,
            self.queue_drain_pass,
            self.slo_restoration_pass,
            self.pre_restore_stability_pass,
            self.queue_depth_at_restore_eligible <= self.recovery_queue_threshold,
        )
        if self.restore_eligible != all(required):
            raise ValueError("restore eligibility is not fail-closed")
        return self


def _as_rate_fraction(rate_per_second: float) -> Fraction:
    if not math.isfinite(rate_per_second) or rate_per_second <= 0:
        raise ValueError("serving rate must be finite and positive")
    return Fraction(Decimal(str(rate_per_second)))


def _epoch_arrivals(epoch: RoutingEpoch) -> tuple[int, ...]:
    return _constant_rate_arrivals(
        start_ns=epoch.start_ns,
        end_ns=epoch.end_ns,
        rate_per_second=epoch.rate_per_second,
    )


def _constant_rate_arrivals(
    *, start_ns: int, end_ns: int, rate_per_second: float
) -> tuple[int, ...]:
    rate = _as_rate_fraction(rate_per_second)
    duration_ns = end_ns - start_ns
    if duration_ns <= 0:
        raise ValueError("serving arrival interval is empty")
    result: list[int] = []
    index = 0
    while True:
        offset = (index * NS_PER_SECOND * rate.denominator) // rate.numerator
        if offset >= duration_ns:
            return tuple(result)
        result.append(start_ns + offset)
        if len(result) > MAX_V10_REQUESTS:
            raise ValueError("v10 global serving plan exceeds its request bound")
        index += 1


def _plan_digest(
    *,
    attempt_id: str,
    seed: int,
    epochs: tuple[RoutingEpoch, ...],
    requests: tuple[RoutedServingRequest, ...],
) -> str:
    payload = {
        "attempt_id": attempt_id,
        "seed": seed,
        "epochs": [epoch.model_dump(mode="json") for epoch in epochs],
        "requests": [request.model_dump(mode="json") for request in requests],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_global_serving_plan(
    *,
    attempt_id: str,
    seed: int,
    control_start_ns: int,
    spike_start_ns: int,
    gpu1_route_start_ns: int,
    restore_start_ns: int,
    end_ns: int,
    control_rate_per_second: float,
    spike_rate_per_second: float,
    restore_rate_per_second: float,
    prompt_tokens: int = 256,
) -> GlobalServingPlan:
    """Build one global stream with a causal GPU1 routing epoch.

    ``gpu1_route_start_ns`` must be the acknowledged routing-boundary event,
    not merely the time GPU1 began reclamation.  The spike rate is identical
    on both sides of this boundary; only request ownership changes.
    """

    if not 0 <= control_start_ns < spike_start_ns < gpu1_route_start_ns < restore_start_ns < end_ns:
        raise ValueError("v10 serving boundaries are not strictly ordered")
    if prompt_tokens <= 0:
        raise ValueError("serving prompt length must be positive")
    epochs = (
        RoutingEpoch(
            name="control",
            start_ns=control_start_ns,
            end_ns=spike_start_ns,
            rate_per_second=control_rate_per_second,
            route_cycle=(GPU0,),
        ),
        RoutingEpoch(
            name="gpu0-overload",
            start_ns=spike_start_ns,
            end_ns=gpu1_route_start_ns,
            rate_per_second=spike_rate_per_second,
            route_cycle=(GPU0,),
        ),
        RoutingEpoch(
            name="two-gpu-recovery",
            start_ns=gpu1_route_start_ns,
            end_ns=restore_start_ns,
            rate_per_second=spike_rate_per_second,
            route_cycle=(GPU0, GPU1),
        ),
        RoutingEpoch(
            name="restore-interference",
            start_ns=restore_start_ns,
            end_ns=end_ns,
            rate_per_second=restore_rate_per_second,
            route_cycle=(GPU0,),
        ),
    )
    requests: list[RoutedServingRequest] = []

    def append_request(
        *,
        epoch: RoutingEpoch,
        local_index: int,
        arrival_ns: int,
    ) -> None:
        sequence = len(requests)
        requests.append(
            RoutedServingRequest(
                sequence=sequence,
                request_id=f"{attempt_id}.{seed}.{sequence:06d}",
                phase=epoch.name,
                scheduled_arrival_ns=arrival_ns,
                planned_device=epoch.route_cycle[local_index % len(epoch.route_cycle)],
                prompt_tokens=prompt_tokens,
            )
        )

    for local_index, arrival_ns in enumerate(_epoch_arrivals(epochs[0])):
        append_request(epoch=epochs[0], local_index=local_index, arrival_ns=arrival_ns)
    recovery_index = 0
    for arrival_ns in _constant_rate_arrivals(
        start_ns=spike_start_ns,
        end_ns=restore_start_ns,
        rate_per_second=spike_rate_per_second,
    ):
        if arrival_ns < gpu1_route_start_ns:
            append_request(epoch=epochs[1], local_index=0, arrival_ns=arrival_ns)
        else:
            append_request(
                epoch=epochs[2],
                local_index=recovery_index,
                arrival_ns=arrival_ns,
            )
            recovery_index += 1
    for local_index, arrival_ns in enumerate(_epoch_arrivals(epochs[3])):
        append_request(epoch=epochs[3], local_index=local_index, arrival_ns=arrival_ns)
    request_tuple = tuple(requests)
    return GlobalServingPlan(
        attempt_id=attempt_id,
        seed=seed,
        plan_sha256=_plan_digest(
            attempt_id=attempt_id,
            seed=seed,
            epochs=epochs,
            requests=request_tuple,
        ),
        epochs=epochs,
        requests=request_tuple,
    )


def audit_serving_execution(
    plan: GlobalServingPlan,
    observations: tuple[ExecutedServingRequest, ...],
) -> ServingExecutionAudit:
    """Reconcile executed requests with the global plan and restore gate."""

    planned = {request.request_id: request for request in plan.requests}
    observed = {request.request_id: request for request in observations}
    routing_matches = (
        len(observed) == len(observations)
        and set(observed) == set(planned)
        and all(
            row.sequence == planned[request_id].sequence
            and row.phase == planned[request_id].phase
            and row.device == planned[request_id].planned_device
            and row.scheduled_arrival_ns == planned[request_id].scheduled_arrival_ns
            for request_id, row in observed.items()
        )
    )
    arrivals = tuple(
        row.scheduled_arrival_ns for row in sorted(observations, key=lambda item: item.sequence)
    )
    enable_ns = plan.epochs[2].start_ns
    restore_ns = plan.epochs[3].start_ns
    gpu0_only = all(
        row.device == GPU0 for row in observations if row.scheduled_arrival_ns < enable_ns
    )
    recovery_devices = {
        row.device for row in observations if enable_ns <= row.scheduled_arrival_ns < restore_ns
    }
    restore_rows = tuple(row for row in observations if row.phase == "restore-interference")
    restore_end_ns = plan.epochs[3].end_ns
    gpu0_completions = sum(
        restore_ns <= row.completed_ns < restore_end_ns
        for row in restore_rows
        if row.device == GPU0
    )
    gpu0_tokens = sum(
        restore_ns <= timestamp < restore_end_ns
        for row in restore_rows
        if row.device == GPU0
        for timestamp in row.token_timestamps_ns
    )
    gpu0_ttft = sum(
        restore_ns <= row.first_token_ns < restore_end_ns
        for row in restore_rows
        if row.device == GPU0
    )
    required = (
        arrivals == tuple(sorted(arrivals)),
        all(
            len(row.output_token_ids) == len(row.token_timestamps_ns) == 64 for row in observations
        ),
        routing_matches,
        gpu0_only,
        recovery_devices == {GPU0, GPU1},
        all(row.device == GPU0 for row in restore_rows),
        gpu0_completions > 0 and gpu0_tokens > 0 and gpu0_ttft > 0,
    )
    return ServingExecutionAudit(
        plan_sha256=plan.plan_sha256,
        request_count=len(observations),
        arrivals_monotonic=required[0],
        exact_output_length=required[1],
        routing_matches_plan=required[2],
        gpu0_only_before_enable=required[3],
        both_gpus_after_enable=required[4],
        gpu1_idle_during_restore=required[5],
        gpu0_restore_completions=gpu0_completions,
        gpu0_restore_emitted_tokens=gpu0_tokens,
        gpu0_restore_ttft_samples=gpu0_ttft,
        gpu0_active_during_restore=required[6],
        passed=all(required),
    )


def outstanding_depth(
    observations: tuple[ExecutedServingRequest, ...], *, timestamp_ns: int
) -> int:
    """Count arrived, incomplete work; unlike v9 this includes engine-owned work."""

    return sum(row.scheduled_arrival_ns <= timestamp_ns < row.completed_ns for row in observations)


def _p95(values: list[int]) -> float | None:
    if not values:
        return None
    values.sort()
    position = (len(values) - 1) * 0.95
    low = math.floor(position)
    high = math.ceil(position)
    return (
        float(values[low])
        if low == high
        else (values[low] * (high - position) + values[high] * (position - low))
    )


def build_reclamation_trigger_evidence(
    observations: tuple[ExecutedServingRequest, ...],
    *,
    window_start_ns: int,
    window_end_ns: int,
    offered_rate_per_second: float,
    trigger_rule: Literal[
        "fixed-overload-window-with-measured-overload",
        "ttft-and-queue-threshold",
    ] = "fixed-overload-window-with-measured-overload",
) -> ReclamationTriggerEvidence:
    """Derive overload evidence from an actual bounded pre-reclaim window."""

    if window_end_ns <= window_start_ns:
        raise ValueError("trigger evidence window is empty")
    duration_seconds = (window_end_ns - window_start_ns) / NS_PER_SECOND
    completions = sum(window_start_ns <= row.completed_ns < window_end_ns for row in observations)
    ttft = [
        row.first_token_ns - row.scheduled_arrival_ns
        for row in observations
        if window_start_ns <= row.scheduled_arrival_ns < window_end_ns
    ]
    p95 = _p95(ttft)
    queue_start = outstanding_depth(observations, timestamp_ns=window_start_ns)
    queue_end = outstanding_depth(observations, timestamp_ns=window_end_ns)
    completed_rate = completions / duration_seconds
    slope = (queue_end - queue_start) / duration_seconds
    reasons: list[str] = []
    if p95 is not None and p95 > 2 * NS_PER_SECOND:
        reasons.append("p95_ttft_above_slo")
    if slope > 0.0:
        reasons.append("queue_positive_drift")
    if offered_rate_per_second > completed_rate:
        reasons.append("offered_above_completed")
    return ReclamationTriggerEvidence(
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        offered_rate_per_second=offered_rate_per_second,
        completed_rate_per_second=completed_rate,
        queue_depth_start=queue_start,
        queue_depth_end=queue_end,
        queue_depth_slope_per_second=slope,
        p95_ttft_ns=p95,
        trigger_rule=trigger_rule,
        trigger_reason=tuple(reasons),
        overload_confirmed=bool(reasons),
    )


def assess_serving_recovery(
    plan: GlobalServingPlan,
    observations: tuple[ExecutedServingRequest, ...],
    *,
    gpu1_first_useful_ns: int,
    recovery_queue_threshold: int,
    stability_window_ns: int = 5 * NS_PER_SECOND,
    evaluation_window_ns: int = NS_PER_SECOND,
) -> ServingRecoveryEvidence:
    """Evaluate two-GPU queue drain and the complete pre-restore SLO gate."""

    restore_eligible_ns = plan.epochs[3].start_ns
    if not plan.epochs[2].start_ns <= gpu1_first_useful_ns < restore_eligible_ns:
        raise ValueError("first useful GPU1 capacity is outside the recovery epoch")
    if recovery_queue_threshold < 0:
        raise ValueError("recovery queue threshold cannot be negative")
    if (
        stability_window_ns <= 0
        or evaluation_window_ns <= 0
        or stability_window_ns % evaluation_window_ns
    ):
        raise ValueError("recovery stability bounds are invalid")
    stability_start_ns = restore_eligible_ns - stability_window_ns
    if stability_start_ns <= gpu1_first_useful_ns:
        raise ValueError("recovery interval is too short for the stability gate")
    duration_seconds = (restore_eligible_ns - gpu1_first_useful_ns) / NS_PER_SECOND
    completions = sum(
        gpu1_first_useful_ns <= row.completed_ns < restore_eligible_ns for row in observations
    )
    completed_rate = completions / duration_seconds
    queue_at_first = outstanding_depth(observations, timestamp_ns=gpu1_first_useful_ns)
    queue_at_stability = outstanding_depth(observations, timestamp_ns=stability_start_ns)
    queue_at_restore = outstanding_depth(observations, timestamp_ns=restore_eligible_ns)
    drain_duration_seconds = (stability_start_ns - gpu1_first_useful_ns) / NS_PER_SECOND
    queue_slope = (queue_at_stability - queue_at_first) / drain_duration_seconds
    windows: list[RecoveryStabilityWindow] = []
    cursor = stability_start_ns
    while cursor < restore_eligible_ns:
        end = cursor + evaluation_window_ns
        ttft = [
            row.first_token_ns - row.scheduled_arrival_ns
            for row in observations
            if cursor <= row.scheduled_arrival_ns < end
        ]
        p95 = _p95(ttft)
        depth = outstanding_depth(observations, timestamp_ns=end)
        passed = p95 is not None and p95 <= 2 * NS_PER_SECOND and depth <= recovery_queue_threshold
        windows.append(
            RecoveryStabilityWindow(
                start_ns=cursor,
                end_ns=end,
                ttft_sample_count=len(ttft),
                p95_ttft_ns=p95,
                queue_depth_end=depth,
                passed=passed,
            )
        )
        cursor = end
    excess = completed_rate > plan.epochs[2].rate_per_second
    queue_drain = queue_slope < 0.0 and queue_at_stability < queue_at_first
    slo_restored = all(
        window.p95_ttft_ns is not None and window.p95_ttft_ns <= 2 * NS_PER_SECOND
        for window in windows
    )
    stable = queue_at_stability <= recovery_queue_threshold and all(
        window.passed for window in windows
    )
    eligible = (
        excess
        and queue_drain
        and slo_restored
        and stable
        and queue_at_restore <= recovery_queue_threshold
    )
    return ServingRecoveryEvidence(
        gpu1_first_useful_ns=gpu1_first_useful_ns,
        restore_eligible_ns=restore_eligible_ns,
        offered_rate_per_second=plan.epochs[2].rate_per_second,
        completed_rate_per_second=completed_rate,
        queue_depth_at_gpu1_first_useful=queue_at_first,
        queue_depth_at_stability_start=queue_at_stability,
        queue_depth_at_restore_eligible=queue_at_restore,
        queue_depth_slope_per_second=queue_slope,
        recovery_queue_threshold=recovery_queue_threshold,
        stability_window_ns=stability_window_ns,
        evaluation_window_ns=evaluation_window_ns,
        stability_windows=tuple(windows),
        two_gpu_excess_capacity_pass=excess,
        queue_drain_pass=queue_drain,
        slo_restoration_pass=slo_restored,
        pre_restore_stability_pass=stable,
        restore_eligible=eligible,
    )


def device_phase_counts(plan: GlobalServingPlan) -> dict[str, dict[str, int]]:
    """Small diagnostic projection used by offline checks and artifacts."""

    result: dict[str, dict[str, int]] = {}
    for epoch in plan.epochs:
        counts = Counter(
            request.planned_device for request in plan.requests if request.phase == epoch.name
        )
        result[epoch.name] = {device: counts[device] for device in (GPU0, GPU1)}
    return result


__all__ = [
    "ExecutedServingRequest",
    "GlobalServingPlan",
    "ReclamationTriggerEvidence",
    "RoutedServingRequest",
    "RoutingEpoch",
    "ServingExecutionAudit",
    "ServingRecoveryEvidence",
    "assess_serving_recovery",
    "audit_serving_execution",
    "build_global_serving_plan",
    "build_reclamation_trigger_evidence",
    "device_phase_counts",
    "outstanding_depth",
]
