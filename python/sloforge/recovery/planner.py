"""Diagnosis-driven recovery planning with deterministic evidence links."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sloforge.autopsy import BottleneckKind, DiagnosisRecord
from sloforge.fabric.ir import (
    DocumentReference,
    MetricInterval,
    PhysicalExecutionPlan,
    PhysicalMetrics,
    RecoveryAction,
    RecoveryActionKind,
    RecoveryCriterion,
    RecoveryPlan,
    RecoveryScope,
    RecoveryVariant,
    TrafficMigrationPlan,
    canonical_hash,
)
from sloforge.ir import ArtifactDigest

from .models import ExecutionTarget, RecoveryPolicy


@dataclass(frozen=True)
class _ActionTemplate:
    kind: RecoveryActionKind
    scope: RecoveryScope
    timeout_seconds: float


_ACTIONS: dict[BottleneckKind, tuple[_ActionTemplate, ...]] = {
    BottleneckKind.ARRIVAL_OVERLOAD: (
        _ActionTemplate(
            RecoveryActionKind.REDUCE_REQUEST_CONCURRENCY, RecoveryScope.REQUEST_PATH, 10.0
        ),
        _ActionTemplate(RecoveryActionKind.SHED_LOW_PRIORITY, RecoveryScope.REQUEST_PATH, 10.0),
    ),
    BottleneckKind.GATEWAY_QUEUEING: (
        _ActionTemplate(
            RecoveryActionKind.REDUCE_REQUEST_CONCURRENCY, RecoveryScope.REQUEST_PATH, 10.0
        ),
    ),
    BottleneckKind.BACKEND_QUEUEING: (
        _ActionTemplate(RecoveryActionKind.CHANGE_WORKER_RATIO, RecoveryScope.REPLICA_LOCAL, 30.0),
    ),
    BottleneckKind.INSUFFICIENT_WARM_CAPACITY: (
        _ActionTemplate(RecoveryActionKind.REPLACE_REPLICA, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.COLD_START_REGRESSION: (
        _ActionTemplate(RecoveryActionKind.REPLACE_REPLICA, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.MODEL_LOADING_REGRESSION: (
        _ActionTemplate(RecoveryActionKind.RESTART_WORKER, RecoveryScope.WORKER_LOCAL, 120.0),
    ),
    BottleneckKind.CPU_LAUNCH_BOTTLENECK: (
        _ActionTemplate(RecoveryActionKind.CHANGE_NUMA_AFFINITY, RecoveryScope.WORKER_LOCAL, 30.0),
    ),
    BottleneckKind.EXCESSIVE_KERNEL_LAUNCHES: (
        _ActionTemplate(
            RecoveryActionKind.REDUCE_REQUEST_CONCURRENCY, RecoveryScope.REQUEST_PATH, 10.0
        ),
    ),
    BottleneckKind.GPU_COMPUTE_REGRESSION: (
        _ActionTemplate(RecoveryActionKind.STOP_ROUTING, RecoveryScope.REQUEST_PATH, 10.0),
        _ActionTemplate(RecoveryActionKind.QUARANTINE_GPU, RecoveryScope.REPLICA_LOCAL, 30.0),
        _ActionTemplate(RecoveryActionKind.CHANGE_RANK_PLACEMENT, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.GPU_MEMORY_BANDWIDTH_REGRESSION: (
        _ActionTemplate(RecoveryActionKind.STOP_ROUTING, RecoveryScope.REQUEST_PATH, 10.0),
        _ActionTemplate(RecoveryActionKind.QUARANTINE_GPU, RecoveryScope.REPLICA_LOCAL, 30.0),
        _ActionTemplate(RecoveryActionKind.CHANGE_RANK_PLACEMENT, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.GPU_CLOCK_THROTTLING: (
        _ActionTemplate(RecoveryActionKind.STOP_ROUTING, RecoveryScope.REQUEST_PATH, 10.0),
        _ActionTemplate(RecoveryActionKind.QUARANTINE_GPU, RecoveryScope.REPLICA_LOCAL, 30.0),
        _ActionTemplate(RecoveryActionKind.REPLACE_REPLICA, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.NUMA_MISPLACEMENT: (
        _ActionTemplate(RecoveryActionKind.CHANGE_NUMA_AFFINITY, RecoveryScope.WORKER_LOCAL, 30.0),
    ),
    BottleneckKind.PCIE_BOTTLENECK: (
        _ActionTemplate(RecoveryActionKind.CHANGE_RANK_PLACEMENT, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.NVLINK_DEGRADATION: (
        _ActionTemplate(RecoveryActionKind.CHANGE_RANK_ORDERING, RecoveryScope.NEW_REPLICA, 120.0),
        _ActionTemplate(RecoveryActionKind.CHANGE_RANK_PLACEMENT, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION: (
        _ActionTemplate(RecoveryActionKind.QUARANTINE_RAIL, RecoveryScope.REPLICA_LOCAL, 30.0),
        _ActionTemplate(RecoveryActionKind.CHANGE_NIC_AFFINITY, RecoveryScope.NEW_REPLICA, 120.0),
        _ActionTemplate(RecoveryActionKind.CHANGE_RANK_PLACEMENT, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.NETWORK_LATENCY_DEGRADATION: (
        _ActionTemplate(RecoveryActionKind.QUARANTINE_NIC, RecoveryScope.REPLICA_LOCAL, 30.0),
        _ActionTemplate(RecoveryActionKind.CHANGE_NIC_AFFINITY, RecoveryScope.NEW_REPLICA, 120.0),
    ),
    BottleneckKind.RANK_STRAGGLER: (
        _ActionTemplate(RecoveryActionKind.STOP_ROUTING, RecoveryScope.REQUEST_PATH, 10.0),
        _ActionTemplate(RecoveryActionKind.DRAIN_WORKER, RecoveryScope.WORKER_LOCAL, 120.0),
        _ActionTemplate(RecoveryActionKind.RESTART_WORKER, RecoveryScope.WORKER_LOCAL, 120.0),
    ),
    BottleneckKind.COLLECTIVE_IMBALANCE: (
        _ActionTemplate(RecoveryActionKind.CHANGE_RANK_ORDERING, RecoveryScope.NEW_REPLICA, 120.0),
        _ActionTemplate(
            RecoveryActionKind.REDUCE_COMMUNICATION_CONCURRENCY, RecoveryScope.REPLICA_LOCAL, 30.0
        ),
    ),
    BottleneckKind.COLLECTIVE_ALGORITHM_REGRESSION: (
        _ActionTemplate(RecoveryActionKind.SWITCH_COLLECTIVE, RecoveryScope.NEW_REPLICA, 120.0),
    ),
    BottleneckKind.EXPERT_LOAD_IMBALANCE: (
        _ActionTemplate(RecoveryActionKind.REPLICATE_HOT_EXPERTS, RecoveryScope.NEW_REPLICA, 180.0),
        _ActionTemplate(RecoveryActionKind.MOVE_EXPERT_GROUP, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.PREFILL_POOL_SATURATION: (
        _ActionTemplate(RecoveryActionKind.CHANGE_WORKER_RATIO, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.DECODE_POOL_SATURATION: (
        _ActionTemplate(RecoveryActionKind.CHANGE_WORKER_RATIO, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.KV_TRANSFER_BOTTLENECK: (
        _ActionTemplate(RecoveryActionKind.SWITCH_KV_TRANSPORT, RecoveryScope.NEW_REPLICA, 120.0),
    ),
    BottleneckKind.UNHEALTHY_WORKER: (
        _ActionTemplate(RecoveryActionKind.STOP_ROUTING, RecoveryScope.REQUEST_PATH, 10.0),
        _ActionTemplate(RecoveryActionKind.DRAIN_WORKER, RecoveryScope.WORKER_LOCAL, 120.0),
        _ActionTemplate(RecoveryActionKind.RESTART_WORKER, RecoveryScope.WORKER_LOCAL, 120.0),
    ),
    BottleneckKind.WORKER_CRASH: (
        _ActionTemplate(RecoveryActionKind.STOP_ROUTING, RecoveryScope.REQUEST_PATH, 10.0),
        _ActionTemplate(RecoveryActionKind.REPLACE_REPLICA, RecoveryScope.NEW_REPLICA, 180.0),
    ),
    BottleneckKind.TOPOLOGY_MISMATCH: (
        _ActionTemplate(
            RecoveryActionKind.REBUILD_DEPLOYMENT, RecoveryScope.DEPLOYMENT_REBUILD, 300.0
        ),
    ),
    BottleneckKind.INVALID_PHYSICAL_PLAN: (
        _ActionTemplate(
            RecoveryActionKind.REBUILD_DEPLOYMENT, RecoveryScope.DEPLOYMENT_REBUILD, 300.0
        ),
    ),
}


def _digest(value: str) -> ArtifactDigest:
    return ArtifactDigest(algorithm="sha256", value=value)


def _diagnosis_reference(diagnosis: DiagnosisRecord) -> DocumentReference:
    return DocumentReference(
        kind="DiagnosisRecord",
        api_version="sloforge.autopsy.diagnosis/v1",
        uri=f"artifacts/autopsy/{diagnosis.degraded_run_id}/diagnosis.json",
        digest=_digest(canonical_hash(diagnosis)),
        uid=diagnosis.diagnosis_id,
        generation=1,
    )


def _physical_plan_reference(plan: PhysicalExecutionPlan) -> DocumentReference:
    return DocumentReference(
        kind="PhysicalExecutionPlan",
        api_version=plan.api_version,
        uri=f"artifacts/physical-plans/{plan.plan_id}.json",
        digest=_digest(canonical_hash(plan)),
        uid=plan.plan_id,
        generation=1,
    )


def _matching_variant(
    diagnosis: DiagnosisRecord, plan: PhysicalExecutionPlan
) -> RecoveryVariant | None:
    eligible = [
        variant
        for variant in plan.recovery_variants
        if any(
            trigger.diagnosis_code
            in {diagnosis.top_hypothesis.value, "fabric_resource_degradation"}
            and diagnosis.confidence >= trigger.minimum_confidence
            for trigger in variant.triggers
        )
    ]
    return min(
        eligible,
        key=lambda item: (
            item.expected_degraded_metrics.p99_tpot_ms.upper,
            item.transition_cost_usd,
            item.transition_seconds,
            item.variant_id,
        ),
        default=None,
    )


def _improvement_fraction(diagnosis: DiagnosisRecord, plan: PhysicalExecutionPlan) -> float:
    top = diagnosis.hypotheses[0]
    if top.counterfactual is not None:
        baseline = max(plan.predicted_metrics.p95_end_to_end_ms.estimate, 1.0)
        return min(0.8, max(0.02, top.counterfactual.expected_improvement_ms / baseline))
    return min(0.5, max(0.05, diagnosis.confidence * 0.30))


def _interval(estimate: float, confidence: float, unit: str) -> MetricInterval:
    nonnegative = max(0.0, estimate)
    width = nonnegative * max(0.08, 1.0 - confidence)
    return MetricInterval(
        estimate=nonnegative,
        lower=max(0.0, nonnegative - width),
        upper=nonnegative + width,
        confidence=confidence,
        unit=unit,
    )


def _improvements(diagnosis: DiagnosisRecord, plan: PhysicalExecutionPlan) -> PhysicalMetrics:
    fraction = _improvement_fraction(diagnosis, plan)
    metrics = plan.predicted_metrics
    confidence = diagnosis.confidence
    return PhysicalMetrics(
        p95_ttft_ms=_interval(metrics.p95_ttft_ms.estimate * fraction, confidence, "ms"),
        p99_tpot_ms=_interval(metrics.p99_tpot_ms.estimate * fraction, confidence, "ms"),
        p95_end_to_end_ms=_interval(
            metrics.p95_end_to_end_ms.estimate * fraction, confidence, "ms"
        ),
        throughput_tokens_per_second=_interval(
            metrics.throughput_tokens_per_second.estimate * fraction,
            confidence,
            "tokens/s",
        ),
        goodput_tokens_per_second=_interval(
            metrics.goodput_tokens_per_second.estimate * fraction,
            confidence,
            "tokens/s",
        ),
        cost_usd_per_million_tokens=_interval(0.0, confidence, "USD/Mtoken"),
        availability=_interval(
            min(1.0 - metrics.availability.estimate, 0.01 * fraction), confidence, "ratio"
        ),
        communication_overhead_fraction=_interval(
            metrics.communication_overhead_fraction.estimate * fraction,
            confidence,
            "ratio",
        ),
    )


def _target(diagnosis: DiagnosisRecord, variant: RecoveryVariant | None) -> tuple[str, ...]:
    hypothesis_target = diagnosis.hypotheses[0].target
    if variant is None:
        return (hypothesis_target,)
    return (hypothesis_target, variant.variant_id)


def _actions(
    diagnosis: DiagnosisRecord,
    variant: RecoveryVariant | None,
    policy: RecoveryPolicy,
    recovery_id: str,
) -> tuple[RecoveryAction, ...]:
    templates = list(_ACTIONS[diagnosis.top_hypothesis])
    if variant is not None and not any(
        item.kind is RecoveryActionKind.CHANGE_RANK_PLACEMENT for item in templates
    ):
        templates.append(
            _ActionTemplate(
                RecoveryActionKind.CHANGE_RANK_PLACEMENT,
                RecoveryScope.NEW_REPLICA,
                max(30.0, variant.transition_seconds),
            )
        )
    if not policy.allow_degraded_model:
        templates = [
            item for item in templates if item.kind is not RecoveryActionKind.DEGRADED_MODEL
        ]
    external = policy.execution_target is ExecutionTarget.EXTERNAL
    targets = _target(diagnosis, variant)
    return tuple(
        RecoveryAction(
            action_id=f"action-{order:02d}-{template.kind.value}",
            kind=template.kind,
            scope=template.scope,
            target_ids=targets,
            order=order,
            idempotency_key=f"{recovery_id}/{order}/{template.kind.value}",
            timeout_seconds=min(template.timeout_seconds, policy.maximum_build_seconds),
            rollback_action_id=None,
            requires_external_mutation=external,
        )
        for order, template in enumerate(templates)
    )


def plan_recovery(
    diagnosis: DiagnosisRecord,
    physical_plan: PhysicalExecutionPlan,
    *,
    policy: RecoveryPolicy | None = None,
) -> RecoveryPlan:
    """Compile a deterministic, evidence-linked recovery plan.

    External mutations are rejected unless the caller supplies an external
    policy carrying explicit authorization. The default output is executable
    only by the local deterministic recovery driver.
    """

    active_policy = policy or RecoveryPolicy()
    if diagnosis.confidence < active_policy.minimum_diagnosis_confidence:
        raise ValueError(
            "diagnosis confidence is below the recovery policy minimum; operator review required"
        )
    variant = _matching_variant(diagnosis, physical_plan)
    identity = "|".join(
        (
            diagnosis.diagnosis_id,
            physical_plan.plan_id,
            diagnosis.top_hypothesis.value,
            variant.variant_id if variant is not None else "direct",
            active_policy.execution_target.value,
        )
    )
    recovery_id = f"recovery-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
    actions = _actions(diagnosis, variant, active_policy, recovery_id)
    scopes = {action.scope for action in actions}
    build_seconds = (
        variant.transition_seconds
        if variant is not None
        else max(action.timeout_seconds for action in actions)
        if RecoveryScope.NEW_REPLICA in scopes or RecoveryScope.DEPLOYMENT_REBUILD in scopes
        else 0.0
    )
    disruption_seconds = (
        5.0
        if RecoveryScope.DEPLOYMENT_REBUILD in scopes
        else 1.0
        if RecoveryScope.WORKER_LOCAL in scopes
        else 0.0
    )
    transition_cost = variant.transition_cost_usd if variant is not None else 0.0
    evidence = tuple(
        DocumentReference(
            kind="AutopsyEvidence",
            api_version="sloforge.autopsy.event/v1",
            uri=item.artifact_uri,
            digest=_digest(item.sha256),
            uid=None,
            generation=None,
        )
        for item in diagnosis.evidence
    )
    compatibility = ["same-model-digest", "preserve-started-streams"]
    if variant is not None:
        compatibility.extend(variant.compatibility_constraints)
        compatibility.append(f"recovery-variant:{variant.variant_id}")
    return RecoveryPlan(
        recovery_id=recovery_id,
        diagnosis=_diagnosis_reference(diagnosis),
        physical_plan=_physical_plan_reference(physical_plan),
        actions=actions,
        expected_slo_improvement=_improvements(diagnosis, physical_plan),
        expected_cost_usd=transition_cost,
        expected_disruption_seconds=disruption_seconds,
        expected_build_seconds=build_seconds,
        confidence=diagnosis.confidence,
        compatibility_constraints=tuple(dict.fromkeys(compatibility)),
        traffic_migration=TrafficMigrationPlan(
            shadow_fraction=active_policy.shadow_fraction,
            canary_fraction=active_policy.canary_fraction,
            minimum_shadow_samples=active_policy.minimum_shadow_samples,
            minimum_canary_samples=active_policy.minimum_canary_samples,
            maximum_inflight_streams_at_drain=(active_policy.maximum_inflight_streams_at_drain),
            preserve_started_streams=active_policy.preserve_started_streams,
        ),
        promotion_criteria=(
            RecoveryCriterion(
                metric="p99_tpot_ms",
                comparator="le",
                threshold=active_policy.target_p99_tpot_ms,
                window_seconds=30.0,
            ),
            RecoveryCriterion(
                metric="p95_ttft_ms",
                comparator="le",
                threshold=active_policy.target_p95_ttft_ms,
                window_seconds=30.0,
            ),
            RecoveryCriterion(
                metric="error_rate",
                comparator="le",
                threshold=active_policy.maximum_error_rate,
                window_seconds=30.0,
            ),
        ),
        rollback_criteria=(
            RecoveryCriterion(
                metric="error_rate",
                comparator="gt",
                threshold=active_policy.maximum_error_rate,
                window_seconds=10.0,
            ),
            RecoveryCriterion(
                metric="p99_tpot_ms",
                comparator="gt",
                threshold=active_policy.target_p99_tpot_ms * 1.20,
                window_seconds=30.0,
            ),
        ),
        abort_criteria=(
            RecoveryCriterion(
                metric="replacement_readiness_seconds",
                comparator="gt",
                threshold=active_policy.maximum_build_seconds,
                window_seconds=active_policy.maximum_build_seconds,
            ),
        ),
        evidence=evidence,
        external_mutation_authorized=active_policy.external_mutation_authorized,
    )
