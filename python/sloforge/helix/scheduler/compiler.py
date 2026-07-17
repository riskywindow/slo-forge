"""Deterministic CPU reference compiler for learning-aware Helix scheduling."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import NoReturn

from .models import (
    LEARNING_WORK_CLASSES,
    AuditDecision,
    BudgetAccounting,
    ClassResourceVectors,
    DecisionKind,
    FaultKind,
    PreemptionRecord,
    PreservationAccounting,
    PreservationMode,
    PrivacyClass,
    ResourceVector,
    SchedulerFault,
    SchedulerPlan,
    SchedulerPolicy,
    SchedulerRequest,
    TickAllocation,
    WorkClass,
    WorkOutcome,
    WorkStatus,
    WorkUnit,
)


class SchedulingInfeasibleError(ValueError):
    """A hard serving, budget, or bounded-preemption constraint cannot be met."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(slots=True)
class _WorkState:
    work: WorkUnit
    resources: ResourceVector
    progress_ticks: int = 0
    executed_ticks: int = 0
    lost_work_ticks: int = 0
    preemptions: int = 0
    ready_tick: int = 0
    started_at_tick: int | None = None
    completed_at_tick: int | None = None
    rejection_reason: str | None = None
    ran_last_tick: bool = False
    adjusted_value_at_completion: float | None = None

    @property
    def remaining_ticks(self) -> int:
        return self.work.duration_ticks - self.progress_ticks

    @property
    def complete(self) -> bool:
        return self.progress_ticks >= self.work.duration_ticks


@dataclass(slots=True)
class _Budget:
    mandatory_serving: int
    learning: int = 0
    preservation: int = 0


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tie(seed: int, identifier: str) -> int:
    payload = f"sloforge.helix.scheduler-tie/v1\0{seed}\0{identifier}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _replace_dimensions(
    vector: ResourceVector,
    *,
    cpu_millicores: int | None = None,
    memory_mib: int | None = None,
    gpu_milliunits: int | None = None,
    storage_mib: int | None = None,
    storage_iops: int | None = None,
    network_mbps: int | None = None,
) -> ResourceVector:
    return ResourceVector(
        cpu_millicores=vector.cpu_millicores if cpu_millicores is None else cpu_millicores,
        memory_mib=vector.memory_mib if memory_mib is None else memory_mib,
        gpu_milliunits=vector.gpu_milliunits if gpu_milliunits is None else gpu_milliunits,
        storage_mib=vector.storage_mib if storage_mib is None else storage_mib,
        storage_iops=vector.storage_iops if storage_iops is None else storage_iops,
        network_mbps=vector.network_mbps if network_mbps is None else network_mbps,
    )


def _active_faults(request: SchedulerRequest, tick: int) -> tuple[SchedulerFault, ...]:
    return tuple(
        sorted((fault for fault in request.faults if fault.active(tick)), key=lambda f: f.fault_id)
    )


def _effective_capacity(
    capacity: ResourceVector, faults: tuple[SchedulerFault, ...]
) -> ResourceVector:
    result = capacity
    for fault in faults:
        remaining = 1.0 - fault.magnitude
        if fault.kind is FaultKind.CPU_EXHAUSTION:
            result = _replace_dimensions(
                result,
                cpu_millicores=result.scaled_down(remaining).cpu_millicores,
            )
        elif fault.kind is FaultKind.GPU_LOSS:
            result = _replace_dimensions(
                result,
                gpu_milliunits=result.scaled_down(remaining).gpu_milliunits,
            )
        elif fault.kind is FaultKind.STORAGE_SLOWDOWN:
            scaled = result.scaled_down(remaining)
            result = _replace_dimensions(
                result,
                storage_mib=scaled.storage_mib,
                storage_iops=scaled.storage_iops,
            )
        elif fault.kind is FaultKind.NETWORK_SLOWDOWN:
            result = _replace_dimensions(
                result,
                network_mbps=result.scaled_down(remaining).network_mbps,
            )
    return result


def _serving_resources(
    request: SchedulerRequest, tick: int, faults: tuple[SchedulerFault, ...]
) -> ResourceVector:
    sample = request.serving_forecast[tick]
    result = request.resource_vectors.serving.multiply(sample.resource_units)
    for fault in faults:
        if fault.kind is FaultKind.TRAFFIC_SPIKE:
            result = result.scaled_up(1.0 + fault.magnitude)
    return result


def _effective_value(work: WorkUnit, faults: tuple[SchedulerFault, ...]) -> float:
    result = work.predicted_learning_value.value
    for fault in faults:
        if fault.kind is not FaultKind.VALUE_PREDICTION_ERROR:
            continue
        if fault.target_work_id is not None and fault.target_work_id != work.work_id:
            continue
        result *= max(0.0, 1.0 + fault.direction * fault.magnitude)
    return result


def _class_vectors(
    serving: ResourceVector, selected: tuple[_WorkState, ...]
) -> ClassResourceVectors:
    totals = {work_class: ResourceVector.zero() for work_class in WorkClass}
    totals[WorkClass.SERVING] = serving
    for state in selected:
        work_class = state.work.work_class
        totals[work_class] = totals[work_class].add(state.resources)
    return ClassResourceVectors(
        serving=totals[WorkClass.SERVING],
        rollout=totals[WorkClass.ROLLOUT],
        environment=totals[WorkClass.ENVIRONMENT],
        reward=totals[WorkClass.REWARD],
        verifier=totals[WorkClass.VERIFIER],
        training=totals[WorkClass.TRAINING],
        evaluation=totals[WorkClass.EVALUATION],
    )


def _learning_total(vectors: ClassResourceVectors) -> ResourceVector:
    total = ResourceVector.zero()
    for work_class in LEARNING_WORK_CLASSES:
        total = total.add(vectors.for_class(work_class))
    return total


def _validate_serving_forecast(request: SchedulerRequest) -> None:
    for sample in request.serving_forecast:
        if sample.predicted_latency_ms > request.serving_slo.maximum_predicted_latency_ms:
            raise SchedulingInfeasibleError(
                "serving_latency_slo",
                f"forecast sample {sample.tick} exceeds the hard predicted-latency constraint",
            )
        if sample.predicted_queue_depth > request.serving_slo.maximum_predicted_queue_depth:
            raise SchedulingInfeasibleError(
                "serving_queue_slo",
                f"forecast sample {sample.tick} exceeds the hard predicted queue-depth constraint",
            )


def _mandatory_serving_cost(request: SchedulerRequest) -> int:
    total = 0
    for tick in range(request.horizon_ticks):
        faults = _active_faults(request, tick)
        capacity = _effective_capacity(request.capacity, faults)
        serving = _serving_resources(request, tick, faults)
        if not request.serving_slo.reserved_capacity.fits_within(capacity):
            raise SchedulingInfeasibleError(
                "serving_reservation_slo",
                f"serving reservation at tick {tick} exceeds fault-adjusted capacity",
            )
        if not serving.fits_within(capacity):
            raise SchedulingInfeasibleError(
                "serving_capacity_slo",
                f"serving demand at tick {tick} exceeds fault-adjusted capacity",
            )
        total += request.constraints.prices.cost(serving)
    if total > request.constraints.max_budget_microunits:
        raise SchedulingInfeasibleError(
            "serving_budget",
            "mandatory serving allocation alone exceeds the hard budget",
        )
    return total


def _governance_rejection(request: SchedulerRequest, state: _WorkState) -> str | None:
    work = state.work
    constraints = request.constraints
    if work.tenant_id not in constraints.allowed_tenant_ids:
        return "tenant is outside the privacy allowlist"
    privacy_rank = {
        PrivacyClass.PUBLIC: 0,
        PrivacyClass.TENANT_PRIVATE: 1,
        PrivacyClass.RESTRICTED: 2,
    }
    if privacy_rank[work.privacy] > privacy_rank[constraints.maximum_privacy]:
        return "work privacy class exceeds the configured maximum"
    if work.effect not in constraints.allowed_effects:
        return "work effect class is not authorized"
    if work.policy_age_ticks + work.duration_ticks > constraints.max_policy_staleness_ticks:
        return "work cannot complete inside the hard policy-staleness limit"
    if (
        work.deadline_tick is not None
        and work.arrival_tick + work.duration_ticks > work.deadline_tick
    ):
        return "work cannot complete by its hard deadline"
    if state.resources.is_zero():
        return "learning work must request a nonzero resource vector"
    if not state.resources.fits_within(request.capacity):
        return "work resource vector exceeds total capacity"
    if (
        request.policy is SchedulerPolicy.HELIX_VALUE_AWARE
        and work.predicted_learning_value.value <= 0.0
    ):
        return "Helix does not spend capacity on non-positive predicted learning value"
    return None


def _branch_order(request: SchedulerRequest, states: tuple[_WorkState, ...]) -> tuple[str, ...]:
    by_branch: dict[str, list[_WorkState]] = {}
    for state in states:
        if state.rejection_reason is None:
            by_branch.setdefault(state.work.branch_id, []).append(state)

    def key(item: tuple[str, list[_WorkState]]) -> tuple[float | int, ...]:
        branch_id, members = item
        if request.policy is SchedulerPolicy.HELIX_VALUE_AWARE:
            total_value = sum(member.work.predicted_learning_value.value for member in members)
            total_cost = sum(
                max(1, request.constraints.prices.cost(member.resources))
                * member.work.duration_ticks
                for member in members
            )
            return (-total_value / total_cost, -total_value, _tie(request.seed, branch_id))
        if request.policy is SchedulerPolicy.UTILIZATION:
            work_volume = sum(
                sum(member.resources.as_tuple()) * member.work.duration_ticks for member in members
            )
            return (-work_volume, _tie(request.seed, branch_id))
        first_arrival = min(member.work.arrival_tick for member in members)
        return (first_arrival, _tie(request.seed, branch_id))

    ordered = sorted(by_branch.items(), key=key)
    return tuple(branch_id for branch_id, _members in ordered)


def _rank_states(
    request: SchedulerRequest,
    states: list[_WorkState],
    tick: int,
    faults: tuple[SchedulerFault, ...],
) -> list[_WorkState]:
    if request.policy is SchedulerPolicy.HELIX_VALUE_AWARE:

        def value_key(state: _WorkState) -> tuple[float | int, ...]:
            value = _effective_value(state.work, faults)
            cost = max(1, request.constraints.prices.cost(state.resources))
            density = value / (cost * max(1, state.remaining_ticks))
            deadline = state.work.deadline_tick if state.work.deadline_tick is not None else 10**9
            return (-density, -value, deadline, _tie(request.seed + tick, state.work.work_id))

        return sorted(states, key=value_key)
    if request.policy is SchedulerPolicy.UTILIZATION:
        return sorted(
            states,
            key=lambda state: (
                0 if state.ran_last_tick else 1,
                -sum(state.resources.as_tuple()),
                state.work.arrival_tick,
                _tie(request.seed, state.work.work_id),
            ),
        )
    return sorted(
        states,
        key=lambda state: (
            0 if state.ran_last_tick else 1,
            state.work.arrival_tick,
            _tie(request.seed, state.work.work_id),
        ),
    )


def _preservation_accounting(state: _WorkState) -> tuple[PreservationAccounting, ...]:
    progress = state.progress_ticks
    result: list[PreservationAccounting] = []
    for option in state.work.preservation:
        if option.mode is PreservationMode.RESTART:
            preserved = 0
        elif option.mode is PreservationMode.CHECKPOINT:
            interval = option.checkpoint_interval_ticks
            preserved = (progress // interval) * interval
        else:
            preserved = progress
        result.append(
            PreservationAccounting(
                mode=option.mode,
                progress_before_ticks=progress,
                preserved_work_ticks=preserved,
                lost_work_ticks=progress - preserved,
                pause_ticks=option.pause_ticks,
                storage_mib_written=option.storage_mib_written,
                network_mib_transferred=option.network_mib_transferred,
                cost_microunits=option.cost_microunits,
                method_evidence=option.method_evidence,
            )
        )
    return tuple(sorted(result, key=lambda item: item.mode.value))


def _choose_preservation(
    alternatives: tuple[PreservationAccounting, ...], available_budget: int
) -> PreservationAccounting:
    affordable = [item for item in alternatives if item.cost_microunits <= available_budget]
    if not affordable:
        raise SchedulingInfeasibleError(
            "preservation_budget",
            "no declared preemption-preservation path fits the remaining hard budget",
        )
    return min(
        affordable,
        key=lambda item: (
            item.lost_work_ticks,
            item.pause_ticks,
            item.cost_microunits,
            item.storage_mib_written + item.network_mib_transferred,
            item.mode.value,
        ),
    )


def _static_fits(
    request: SchedulerRequest,
    state: _WorkState,
    selected: tuple[_WorkState, ...],
) -> bool:
    if request.policy is not SchedulerPolicy.STATIC:
        return True
    assert request.static_limits is not None
    used = ResourceVector.zero()
    for member in selected:
        if member.work.work_class is state.work.work_class:
            used = used.add(member.resources)
    return used.add(state.resources).fits_within(
        request.static_limits.for_class(state.work.work_class)
    )


def _dynamic_rejection(request: SchedulerRequest, state: _WorkState, tick: int) -> str | None:
    completion_tick = tick + state.remaining_ticks
    if state.work.deadline_tick is not None and completion_tick > state.work.deadline_tick:
        return "remaining work cannot complete by its hard deadline"
    age_at_completion = state.work.policy_age_ticks + max(
        0, completion_tick - state.work.arrival_tick
    )
    if age_at_completion > request.constraints.max_policy_staleness_ticks:
        return "remaining work cannot complete inside the hard policy-staleness limit"
    return None


class _Compiler:
    def __init__(self, request: SchedulerRequest) -> None:
        self.request = request
        self.states = tuple(
            _WorkState(
                work=work,
                resources=request.resource_vectors.for_class(work.work_class).multiply(
                    work.resource_units
                ),
                ready_tick=work.arrival_tick,
            )
            for work in request.work
        )
        self.decisions: list[AuditDecision] = []
        self.preemptions: list[PreemptionRecord] = []
        self.ticks: list[TickAllocation] = []
        self.previous_lent = ResourceVector.zero()
        self.budget = _Budget(mandatory_serving=_mandatory_serving_cost(request))
        self.selected_branches: tuple[str, ...] = ()

    def audit(
        self,
        *,
        tick: int,
        kind: DecisionKind,
        subject_id: str,
        reason: str,
        capacity: ResourceVector,
        requested: ResourceVector,
        value: float | None = None,
        faults: tuple[SchedulerFault, ...] = (),
    ) -> None:
        if len(self.decisions) >= self.request.max_audit_records:
            raise SchedulingInfeasibleError(
                "audit_bound", "scheduler decision audit exceeded max_audit_records"
            )
        self.decisions.append(
            AuditDecision(
                sequence=len(self.decisions),
                tick=tick,
                kind=kind,
                subject_id=subject_id,
                reason=reason,
                effective_capacity=capacity,
                requested_resources=requested,
                effective_predicted_value=value,
                fault_ids=tuple(fault.fault_id for fault in faults),
            )
        )

    def initialize(self) -> None:
        capacity = self.request.capacity
        for state in self.states:
            state.rejection_reason = _governance_rejection(self.request, state)
            if state.rejection_reason is not None:
                self.audit(
                    tick=0,
                    kind=DecisionKind.REJECT,
                    subject_id=state.work.work_id,
                    reason=state.rejection_reason,
                    capacity=capacity,
                    requested=state.resources,
                    value=state.work.predicted_learning_value.value,
                )
        ordered = _branch_order(self.request, self.states)
        self.selected_branches = ordered[: self.request.constraints.max_selected_branches]
        for branch_id in self.selected_branches:
            self.audit(
                tick=0,
                kind=DecisionKind.SELECT_BRANCH,
                subject_id=branch_id,
                reason=f"selected by {self.request.policy.value} branch-count policy",
                capacity=capacity,
                requested=ResourceVector.zero(),
            )
        selected = set(self.selected_branches)
        for state in self.states:
            if state.rejection_reason is None and state.work.branch_id not in selected:
                state.rejection_reason = "branch excluded by the hard branch-count limit"
                self.audit(
                    tick=0,
                    kind=DecisionKind.REJECT,
                    subject_id=state.work.work_id,
                    reason=state.rejection_reason,
                    capacity=capacity,
                    requested=state.resources,
                    value=state.work.predicted_learning_value.value,
                )

    def available_learning_budget(self) -> int:
        return (
            self.request.constraints.max_budget_microunits
            - self.budget.mandatory_serving
            - self.budget.learning
            - self.budget.preservation
        )

    def fail_preemption_bound(self, state: _WorkState, tick: int) -> NoReturn:
        raise SchedulingInfeasibleError(
            "bounded_preemption",
            f"hard serving reclamation at tick {tick} would exceed the preemption bound "
            f"for {state.work.work_id}",
        )

    def compile_tick(self, tick: int) -> None:
        request = self.request
        faults = _active_faults(request, tick)
        capacity = _effective_capacity(request.capacity, faults)
        serving = _serving_resources(request, tick, faults)
        sample = request.serving_forecast[tick]
        if not serving.fits_within(capacity):
            raise SchedulingInfeasibleError(
                "serving_capacity_slo",
                f"serving demand at tick {tick} exceeds fault-adjusted capacity",
            )
        for fault in faults:
            self.audit(
                tick=tick,
                kind=DecisionKind.APPLY_FAULT,
                subject_id=fault.fault_id,
                reason=f"applied deterministic {fault.kind.value} scenario input",
                capacity=capacity,
                requested=ResourceVector.zero(),
                faults=(fault,),
            )

        reserved = request.serving_slo.reserved_capacity
        base_learning_capacity = capacity.subtract(reserved)
        lending_enabled = (
            request.constraints.capacity_lending and request.policy is not SchedulerPolicy.DEDICATED
        )
        if lending_enabled:
            learning_capacity = capacity.subtract(serving)
            lendable = reserved.positive_difference(serving).minimum(learning_capacity)
        else:
            learning_capacity = capacity.subtract(reserved.maximum(serving))
            lendable = ResourceVector.zero()

        candidates: list[_WorkState] = []
        for state in self.states:
            if state.complete or state.rejection_reason is not None:
                continue
            if state.work.arrival_tick > tick or state.ready_tick > tick:
                continue
            dynamic_reason = _dynamic_rejection(request, state, tick)
            if dynamic_reason is not None:
                state.rejection_reason = dynamic_reason
                self.audit(
                    tick=tick,
                    kind=DecisionKind.REJECT,
                    subject_id=state.work.work_id,
                    reason=dynamic_reason,
                    capacity=capacity,
                    requested=state.resources,
                    value=_effective_value(state.work, faults),
                    faults=faults,
                )
                continue
            candidates.append(state)

        ranked = _rank_states(request, candidates, tick, faults)
        total_preemptions = len(self.preemptions)
        pinned = [
            state
            for state in ranked
            if state.ran_last_tick
            and (
                state.preemptions >= request.constraints.max_preemptions_per_work
                or total_preemptions >= request.constraints.max_total_preemptions
            )
        ]
        selected: list[_WorkState] = []
        used = ResourceVector.zero()
        budget_commitment = 0
        for state in [*pinned, *(item for item in ranked if item not in pinned)]:
            proposed = used.add(state.resources)
            full_cost = request.constraints.prices.cost(state.resources) * state.remaining_ticks
            fits = proposed.fits_within(learning_capacity)
            static_fits = _static_fits(request, state, tuple(selected))
            budget_fits = budget_commitment + full_cost <= self.available_learning_budget()
            if fits and static_fits and budget_fits:
                selected.append(state)
                used = proposed
                budget_commitment += full_cost
            elif state in pinned:
                self.fail_preemption_bound(state, tick)

        selected_ids = {state.work.work_id for state in selected}
        previously_running = [
            state
            for state in self.states
            if state.ran_last_tick and not state.complete and state.rejection_reason is None
        ]
        for state in previously_running:
            if state.work.work_id in selected_ids:
                continue
            if (
                state.preemptions >= request.constraints.max_preemptions_per_work
                or len(self.preemptions) >= request.constraints.max_total_preemptions
            ):
                self.fail_preemption_bound(state, tick)
            alternatives = _preservation_accounting(state)
            chosen = _choose_preservation(alternatives, self.available_learning_budget())
            state.progress_ticks = chosen.preserved_work_ticks
            state.lost_work_ticks += chosen.lost_work_ticks
            state.preemptions += 1
            state.ready_tick = tick + chosen.pause_ticks
            self.budget.preservation += chosen.cost_microunits
            reason = (
                "serving capacity reclamation"
                if not used.add(state.resources).fits_within(learning_capacity)
                else "policy selected higher-priority learning work"
            )
            record = PreemptionRecord(
                sequence=len(self.preemptions),
                tick=tick,
                work_id=state.work.work_id,
                reason=reason,
                selected_mode=chosen.mode,
                selected=chosen,
                alternatives=alternatives,
                total_preemptions_for_work=state.preemptions,
            )
            self.preemptions.append(record)
            self.audit(
                tick=tick,
                kind=DecisionKind.PREEMPT,
                subject_id=state.work.work_id,
                reason=f"{reason}; selected {chosen.mode.value} preservation",
                capacity=capacity,
                requested=state.resources,
                value=_effective_value(state.work, faults),
                faults=faults,
            )

        selected_tuple = tuple(selected)
        class_vectors = _class_vectors(serving, selected_tuple)
        learning = _learning_total(class_vectors)
        lent_used = learning.positive_difference(base_learning_capacity).minimum(lendable)
        reclaimed = self.previous_lent.positive_difference(lent_used)
        if not lent_used.is_zero():
            self.audit(
                tick=tick,
                kind=DecisionKind.LEND_CAPACITY,
                subject_id="serving.capacity",
                reason="unused serving reservation lent to bounded learning work",
                capacity=capacity,
                requested=lent_used,
                faults=faults,
            )
        if not reclaimed.is_zero():
            self.audit(
                tick=tick,
                kind=DecisionKind.RECLAIM_CAPACITY,
                subject_id="serving.capacity",
                reason="serving reservation reclaimed before serving allocation",
                capacity=capacity,
                requested=reclaimed,
                faults=faults,
            )
        self.previous_lent = lent_used

        for state in self.states:
            state.ran_last_tick = False
        for state in selected:
            first = state.started_at_tick is None
            if first:
                state.started_at_tick = tick
            state.progress_ticks += 1
            state.executed_ticks += 1
            state.ran_last_tick = True
            self.budget.learning += request.constraints.prices.cost(state.resources)
            self.audit(
                tick=tick,
                kind=DecisionKind.START if first else DecisionKind.CONTINUE,
                subject_id=state.work.work_id,
                reason="resource-feasible work selected by deterministic policy order",
                capacity=capacity,
                requested=state.resources,
                value=_effective_value(state.work, faults),
                faults=faults,
            )
            if state.complete:
                state.completed_at_tick = tick + 1
                state.adjusted_value_at_completion = _effective_value(state.work, faults)
                state.ran_last_tick = False
                self.audit(
                    tick=tick,
                    kind=DecisionKind.COMPLETE,
                    subject_id=state.work.work_id,
                    reason="all declared work ticks completed",
                    capacity=capacity,
                    requested=state.resources,
                    value=state.adjusted_value_at_completion,
                    faults=faults,
                )

        for state in candidates:
            if state.work.work_id not in selected_ids and not state.ran_last_tick:
                self.audit(
                    tick=tick,
                    kind=DecisionKind.DEFER,
                    subject_id=state.work.work_id,
                    reason="not selected within current capacity, static-share, and budget bounds",
                    capacity=capacity,
                    requested=state.resources,
                    value=_effective_value(state.work, faults),
                    faults=faults,
                )

        tick_cost = request.constraints.prices.cost(serving.add(learning))
        self.ticks.append(
            TickAllocation(
                tick=tick,
                effective_capacity=capacity,
                allocations=class_vectors,
                serving_resources=serving,
                learning_resources=learning,
                lent_capacity=lent_used,
                reclaimed_capacity=reclaimed,
                running_work_ids=tuple(state.work.work_id for state in selected),
                active_fault_ids=tuple(fault.fault_id for fault in faults),
                serving_predicted_latency_ms=sample.predicted_latency_ms,
                serving_predicted_queue_depth=sample.predicted_queue_depth,
                cost_microunits=tick_cost,
            )
        )

    def finish(self) -> SchedulerPlan:
        outcomes: list[WorkOutcome] = []
        for state in sorted(self.states, key=lambda item: item.work.work_id):
            if state.complete:
                status = WorkStatus.COMPLETED
                reason = "completed within all hard modeled constraints"
            elif state.rejection_reason is not None:
                status = WorkStatus.REJECTED
                reason = state.rejection_reason
            else:
                status = WorkStatus.DEFERRED
                reason = "scheduling horizon ended before feasible completion"
            outcomes.append(
                WorkOutcome(
                    work_id=state.work.work_id,
                    branch_id=state.work.branch_id,
                    work_class=state.work.work_class,
                    status=status,
                    reason=reason,
                    progress_ticks=state.progress_ticks,
                    executed_ticks=state.executed_ticks,
                    lost_work_ticks=state.lost_work_ticks,
                    preemptions=state.preemptions,
                    started_at_tick=state.started_at_tick,
                    completed_at_tick=state.completed_at_tick,
                    predicted_learning_value=state.work.predicted_learning_value.value,
                    prediction_evidence=state.work.predicted_learning_value.evidence,
                )
            )
        completed = tuple(
            state
            for state in sorted(self.states, key=lambda item: item.work.work_id)
            if state.complete
        )
        predicted = math.fsum(state.work.predicted_learning_value.value for state in completed)
        adjusted = math.fsum(
            state.adjusted_value_at_completion
            if state.adjusted_value_at_completion is not None
            else state.work.predicted_learning_value.value
            for state in completed
        )
        total = self.budget.mandatory_serving + self.budget.learning + self.budget.preservation
        if total > self.request.constraints.max_budget_microunits:
            raise SchedulingInfeasibleError(
                "budget", "compiled allocation exceeded the hard resource budget"
            )
        budget = BudgetAccounting(
            limit_microunits=self.request.constraints.max_budget_microunits,
            serving_microunits=self.budget.mandatory_serving,
            learning_microunits=self.budget.learning,
            preservation_microunits=self.budget.preservation,
            total_microunits=total,
            remaining_microunits=self.request.constraints.max_budget_microunits - total,
        )
        request_digest = _canonical_digest(self.request.model_dump(mode="json"))
        draft = {
            "schema_version": "sloforge.helix.scheduler-plan/v1",
            "request_digest": request_digest,
            "request_id": self.request.request_id,
            "seed": self.request.seed,
            "policy": self.request.policy,
            "selected_branch_ids": self.selected_branches,
            "ticks": tuple(item.model_dump(mode="json") for item in self.ticks),
            "outcomes": tuple(item.model_dump(mode="json") for item in outcomes),
            "preemptions": tuple(item.model_dump(mode="json") for item in self.preemptions),
            "decisions": tuple(item.model_dump(mode="json") for item in self.decisions),
            "budget": budget.model_dump(mode="json"),
            "predicted_learning_value": predicted,
            "scheduler_adjusted_predicted_value": adjusted,
            "completed_work_ids": tuple(state.work.work_id for state in completed),
            "limitations": (
                "CPU deterministic tick model; it is not a wall-clock or GPU performance measurement",
                "serving latency and queue-depth results are hard feasibility checks against supplied predictions, not observed SLO measurements",
                "serving forecasts and learning-value estimates retain caller-supplied evidence references",
                "faults are deterministic scenario inputs and do not claim observed production failures",
            ),
        }
        plan_id = _canonical_digest(draft)
        return SchedulerPlan(
            plan_id=plan_id,
            request_digest=request_digest,
            request_id=self.request.request_id,
            seed=self.request.seed,
            policy=self.request.policy,
            selected_branch_ids=self.selected_branches,
            ticks=tuple(self.ticks),
            outcomes=tuple(outcomes),
            preemptions=tuple(self.preemptions),
            decisions=tuple(self.decisions),
            budget=budget,
            predicted_learning_value=predicted,
            scheduler_adjusted_predicted_value=adjusted,
            completed_work_ids=tuple(state.work.work_id for state in completed),
            limitations=(
                "CPU deterministic tick model; it is not a wall-clock or GPU performance measurement",
                "serving latency and queue-depth results are hard feasibility checks against supplied predictions, not observed SLO measurements",
                "serving forecasts and learning-value estimates retain caller-supplied evidence references",
                "faults are deterministic scenario inputs and do not claim observed production failures",
            ),
        )


def compile_resource_plan(request: SchedulerRequest) -> SchedulerPlan:
    """Compile a bounded deterministic schedule while enforcing hard serving constraints."""

    _validate_serving_forecast(request)
    compiler = _Compiler(request)
    compiler.initialize()
    for tick in range(request.horizon_ticks):
        compiler.compile_tick(tick)
    return compiler.finish()


__all__ = ["SchedulingInfeasibleError", "compile_resource_plan"]
