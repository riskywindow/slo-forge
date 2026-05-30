"""Pure guarded recovery state machine with bounded persistent history."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from sloforge.fabric.ir import RecoveryCriterion, RecoveryPlan, canonical_hash

from .models import (
    ActionAttempt,
    AuditField,
    AuditRecord,
    RecoveryMachineConfig,
    RecoveryObservation,
    RecoverySnapshot,
    RecoveryState,
)

_TIMEOUT_FIELD: dict[RecoveryState, str] = {
    RecoveryState.PROPOSED: "proposed_timeout_ms",
    RecoveryState.VALIDATED_IN_SIMULATION: "validation_timeout_ms",
    RecoveryState.BUILDING_REPLACEMENT: "build_timeout_ms",
    RecoveryState.SHADOWING: "shadow_timeout_ms",
    RecoveryState.CANARYING: "canary_timeout_ms",
    RecoveryState.PROMOTING: "promotion_timeout_ms",
    RecoveryState.DRAINING_OLD: "drain_timeout_ms",
}


def _criterion_matches(criterion: RecoveryCriterion, value: float) -> bool:
    return {
        "lt": value < criterion.threshold,
        "le": value <= criterion.threshold,
        "gt": value > criterion.threshold,
        "ge": value >= criterion.threshold,
    }[criterion.comparator]


def _matched_criteria(
    criteria: Iterable[RecoveryCriterion], observation: RecoveryObservation
) -> tuple[RecoveryCriterion, ...]:
    values = {item.name: item for item in observation.metrics}
    return tuple(
        criterion
        for criterion in criteria
        if criterion.metric in values
        and values[criterion.metric].window_seconds >= criterion.window_seconds
        and _criterion_matches(criterion, values[criterion.metric].value)
    )


def _all_criteria_pass(
    criteria: tuple[RecoveryCriterion, ...], observation: RecoveryObservation
) -> bool:
    values = {item.name: item for item in observation.metrics}
    return all(
        criterion.metric in values
        and values[criterion.metric].window_seconds >= criterion.window_seconds
        and _criterion_matches(criterion, values[criterion.metric].value)
        for criterion in criteria
    )


class RecoveryStateMachine:
    """Deterministic state transitions; callers persist every returned snapshot."""

    def __init__(self, plan: RecoveryPlan, config: RecoveryMachineConfig | None = None) -> None:
        self.plan = plan
        self.config = config or RecoveryMachineConfig()
        self.plan_hash = canonical_hash(plan)

    def start(self, *, now_ms: int) -> RecoverySnapshot:
        if now_ms < 0:
            raise ValueError("recovery start time must be non-negative")
        audit = AuditRecord(
            record_id=f"{self.plan.recovery_id}:000000:created",
            sequence=0,
            at_ms=now_ms,
            event="created",
            state_before=RecoveryState.PROPOSED,
            state_after=RecoveryState.PROPOSED,
            reason="recovery proposal loaded; no mutation performed",
            idempotency_key=f"{self.plan.recovery_id}:start",
        )
        return RecoverySnapshot(
            recovery_id=self.plan.recovery_id,
            recovery_plan_hash=self.plan_hash,
            state=RecoveryState.PROPOSED,
            sequence=0,
            started_at_ms=now_ms,
            state_entered_at_ms=now_ms,
            deadline_ms=now_ms + self.config.maximum_total_duration_ms,
            last_observed_at_ms=now_ms,
            audit=(audit,),
        )

    def restore(self, payload: str | bytes) -> RecoverySnapshot:
        snapshot = RecoverySnapshot.model_validate_json(payload, strict=True)
        if snapshot.recovery_id != self.plan.recovery_id:
            raise ValueError("snapshot recovery identifier does not match plan")
        if snapshot.recovery_plan_hash != self.plan_hash:
            raise ValueError("snapshot plan hash does not match plan")
        return snapshot

    def _timeout_ms(self, state: RecoveryState) -> int:
        field = _TIMEOUT_FIELD.get(state)
        return int(getattr(self.config, field)) if field is not None else 0

    def _audit(
        self,
        snapshot: RecoverySnapshot,
        observation: RecoveryObservation,
        *,
        event: Literal["created", "transition", "guard", "action", "wait", "restored"],
        state_after: RecoveryState,
        reason: str,
        fields: tuple[AuditField, ...] = (),
    ) -> RecoverySnapshot:
        sequence = snapshot.sequence + int(state_after is not snapshot.state)
        record = AuditRecord(
            record_id=f"{self.plan.recovery_id}:{sequence:06d}:{event}",
            sequence=sequence,
            at_ms=observation.observed_at_ms,
            event=event,
            state_before=snapshot.state,
            state_after=state_after,
            reason=reason,
            idempotency_key=observation.idempotency_key,
            fields=fields,
        )
        audit = (*snapshot.audit, record)[-self.config.maximum_audit_records :]
        keys = (
            *snapshot.processed_idempotency_keys,
            observation.idempotency_key,
        )[-self.config.maximum_idempotency_keys :]
        cooldown = snapshot.cooldown_until_ms
        if state_after is RecoveryState.PROMOTING:
            cooldown = observation.observed_at_ms + self.config.promotion_cooldown_ms
        return RecoverySnapshot.model_validate(
            {
                **snapshot.model_dump(mode="python"),
                "state": state_after,
                "sequence": sequence,
                "state_entered_at_ms": (
                    observation.observed_at_ms
                    if state_after is not snapshot.state
                    else snapshot.state_entered_at_ms
                ),
                "last_observed_at_ms": observation.observed_at_ms,
                "cooldown_until_ms": cooldown,
                "processed_idempotency_keys": keys,
                "audit": audit,
            }
        )

    def _transition(
        self,
        snapshot: RecoverySnapshot,
        observation: RecoveryObservation,
        state: RecoveryState,
        reason: str,
    ) -> RecoverySnapshot:
        return self._audit(
            snapshot,
            observation,
            event="transition",
            state_after=state,
            reason=reason,
        )

    def record_action_attempts(
        self,
        snapshot: RecoverySnapshot,
        *,
        attempts: tuple[ActionAttempt, ...],
    ) -> RecoverySnapshot:
        if snapshot.state is not RecoveryState.BUILDING_REPLACEMENT:
            raise ValueError("recovery actions execute only while building a replacement")
        known = {action.action_id for action in self.plan.actions}
        if any(attempt.action_id not in known for attempt in attempts):
            raise ValueError("action attempt references an unknown recovery action")
        previous = {attempt.action_id for attempt in snapshot.action_attempts}
        if any(attempt.action_id in previous for attempt in attempts):
            raise ValueError("recovery actions are idempotent and may be attempted only once")
        applied = tuple(
            dict.fromkeys(
                (
                    *snapshot.applied_action_ids,
                    *(attempt.action_id for attempt in attempts if attempt.succeeded),
                )
            )
        )
        audit = list(snapshot.audit)
        for attempt in attempts:
            audit.append(
                AuditRecord(
                    record_id=(
                        f"{self.plan.recovery_id}:{snapshot.sequence:06d}:action:"
                        f"{attempt.action_id}"
                    ),
                    sequence=snapshot.sequence,
                    at_ms=attempt.attempted_at_ms,
                    event="action",
                    state_before=snapshot.state,
                    state_after=snapshot.state,
                    reason=attempt.detail,
                    idempotency_key=attempt.idempotency_key,
                    fields=(AuditField(name="succeeded", value=str(attempt.succeeded).lower()),),
                )
            )
        return RecoverySnapshot.model_validate(
            {
                **snapshot.model_dump(mode="python"),
                "applied_action_ids": applied,
                "action_attempts": (*snapshot.action_attempts, *attempts),
                "audit": tuple(audit[-self.config.maximum_audit_records :]),
            }
        )

    def advance(
        self, snapshot: RecoverySnapshot, observation: RecoveryObservation
    ) -> RecoverySnapshot:
        if snapshot.recovery_plan_hash != self.plan_hash:
            raise ValueError("snapshot plan hash does not match the active recovery plan")
        if observation.idempotency_key in snapshot.processed_idempotency_keys:
            return snapshot
        if observation.observed_at_ms < snapshot.last_observed_at_ms:
            raise ValueError("recovery observations must be monotonic")
        if snapshot.state.terminal:
            return snapshot
        if observation.observed_at_ms > snapshot.deadline_ms:
            terminal = (
                RecoveryState.OPERATOR_REQUIRED
                if snapshot.state is RecoveryState.DRAINING_OLD
                and observation.active_started_streams > 0
                else RecoveryState.ROLLED_BACK
                if snapshot.state
                in {
                    RecoveryState.SHADOWING,
                    RecoveryState.CANARYING,
                    RecoveryState.PROMOTING,
                    RecoveryState.DRAINING_OLD,
                }
                else RecoveryState.ABORTED
            )
            return self._transition(
                snapshot, observation, terminal, "total recovery deadline expired"
            )
        state_timeout = self._timeout_ms(snapshot.state)
        if observation.observed_at_ms - snapshot.state_entered_at_ms > state_timeout:
            terminal = (
                RecoveryState.OPERATOR_REQUIRED
                if snapshot.state is RecoveryState.DRAINING_OLD
                and observation.active_started_streams > 0
                else RecoveryState.ROLLED_BACK
                if snapshot.state
                in {
                    RecoveryState.SHADOWING,
                    RecoveryState.CANARYING,
                    RecoveryState.PROMOTING,
                    RecoveryState.DRAINING_OLD,
                }
                else RecoveryState.ABORTED
            )
            return self._transition(
                snapshot, observation, terminal, "state transition deadline expired"
            )
        aborts = _matched_criteria(self.plan.abort_criteria, observation)
        if aborts:
            return self._transition(
                snapshot,
                observation,
                RecoveryState.ABORTED,
                f"abort criterion matched: {aborts[0].metric}",
            )
        if snapshot.state in {
            RecoveryState.SHADOWING,
            RecoveryState.CANARYING,
            RecoveryState.PROMOTING,
        }:
            rollbacks = _matched_criteria(self.plan.rollback_criteria, observation)
            if rollbacks:
                return self._transition(
                    snapshot,
                    observation,
                    RecoveryState.ROLLED_BACK,
                    f"rollback criterion matched: {rollbacks[0].metric}",
                )
        if snapshot.state is RecoveryState.PROPOSED:
            if self.plan.confidence < self.config.minimum_plan_confidence:
                return self._transition(
                    snapshot,
                    observation,
                    RecoveryState.REJECTED,
                    "recovery confidence is below the execution threshold",
                )
            if any(action.requires_external_mutation for action in self.plan.actions) and not (
                self.plan.external_mutation_authorized and self.config.external_mutation_authorized
            ):
                return self._transition(
                    snapshot,
                    observation,
                    RecoveryState.OPERATOR_REQUIRED,
                    "external mutation requires an authorized external action driver",
                )
            if not observation.simulation_validated:
                return self._audit(
                    snapshot,
                    observation,
                    event="wait",
                    state_after=snapshot.state,
                    reason="waiting for counterfactual simulation validation",
                )
            return self._transition(
                snapshot,
                observation,
                RecoveryState.VALIDATED_IN_SIMULATION,
                "counterfactual simulation validation passed",
            )
        if snapshot.state is RecoveryState.VALIDATED_IN_SIMULATION:
            return self._transition(
                snapshot,
                observation,
                RecoveryState.BUILDING_REPLACEMENT,
                "beginning idempotent recovery actions",
            )
        if snapshot.state is RecoveryState.BUILDING_REPLACEMENT:
            failed = next(
                (attempt for attempt in snapshot.action_attempts if not attempt.succeeded), None
            )
            if failed is not None or observation.failed_action_id is not None:
                return self._transition(
                    snapshot,
                    observation,
                    RecoveryState.ABORTED,
                    f"recovery action failed: {failed.action_id if failed else observation.failed_action_id}",
                )
            completed = {*snapshot.applied_action_ids, *observation.completed_action_ids}
            required = {action.action_id for action in self.plan.actions}
            unknown = completed - required
            if unknown:
                raise ValueError(f"observation references unknown completed actions: {unknown}")
            if completed != required or not observation.replacement_ready:
                return self._audit(
                    snapshot,
                    observation,
                    event="wait",
                    state_after=snapshot.state,
                    reason="waiting for replacement readiness and recovery actions",
                    fields=(
                        AuditField(name="completed_actions", value=str(len(completed))),
                        AuditField(name="required_actions", value=str(len(required))),
                    ),
                )
            return self._transition(
                snapshot,
                observation,
                RecoveryState.SHADOWING,
                "replacement is ready; starting shadow traffic",
            )
        if snapshot.state is RecoveryState.SHADOWING:
            if observation.shadow_samples < self.plan.traffic_migration.minimum_shadow_samples:
                return self._audit(
                    snapshot,
                    observation,
                    event="wait",
                    state_after=snapshot.state,
                    reason="minimum shadow sample requirement not met",
                )
            if not _all_criteria_pass(self.plan.promotion_criteria, observation):
                return self._audit(
                    snapshot,
                    observation,
                    event="guard",
                    state_after=snapshot.state,
                    reason="shadow promotion metrics are missing or outside criteria",
                )
            return self._transition(
                snapshot,
                observation,
                RecoveryState.CANARYING,
                "shadow sample and SLO criteria passed",
            )
        if snapshot.state is RecoveryState.CANARYING:
            if observation.canary_samples < self.plan.traffic_migration.minimum_canary_samples:
                return self._audit(
                    snapshot,
                    observation,
                    event="wait",
                    state_after=snapshot.state,
                    reason="minimum canary sample requirement not met",
                )
            if not _all_criteria_pass(self.plan.promotion_criteria, observation):
                return self._audit(
                    snapshot,
                    observation,
                    event="guard",
                    state_after=snapshot.state,
                    reason="canary promotion metrics are missing or outside criteria",
                )
            return self._transition(
                snapshot,
                observation,
                RecoveryState.PROMOTING,
                "canary sample and SLO criteria passed",
            )
        if snapshot.state is RecoveryState.PROMOTING:
            if (
                snapshot.cooldown_until_ms is not None
                and observation.observed_at_ms < snapshot.cooldown_until_ms
            ):
                return self._audit(
                    snapshot,
                    observation,
                    event="wait",
                    state_after=snapshot.state,
                    reason="promotion cooldown is active",
                )
            if not observation.traffic_migration_complete:
                return self._audit(
                    snapshot,
                    observation,
                    event="wait",
                    state_after=snapshot.state,
                    reason="waiting for traffic migration completion",
                )
            return self._transition(
                snapshot,
                observation,
                RecoveryState.DRAINING_OLD,
                "traffic migrated; old workers no longer receive new requests",
            )
        if snapshot.state is RecoveryState.DRAINING_OLD:
            if (
                self.plan.traffic_migration.preserve_started_streams
                and observation.active_started_streams > 0
            ):
                return self._audit(
                    snapshot,
                    observation,
                    event="wait",
                    state_after=snapshot.state,
                    reason="preserving started streaming requests during graceful drain",
                    fields=(
                        AuditField(
                            name="active_started_streams",
                            value=str(observation.active_started_streams),
                        ),
                    ),
                )
            return self._transition(
                snapshot,
                observation,
                RecoveryState.COMPLETED,
                "old workers drained; recovery completed",
            )
        raise AssertionError(f"unhandled recovery state {snapshot.state}")
