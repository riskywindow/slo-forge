from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.fabric.simulation import (
    CalibrationProvenance,
    FabricSimulationRequest,
    PhysicalResource,
    ProvenanceKind,
    ResourceKind,
    SchedulingMode,
    ServiceCurve,
    ServiceCurvePoint,
)
from sloforge.helix.scheduler import (
    ClassResourceVectors,
    DecisionKind,
    EffectClass,
    EvidenceRef,
    FabricCapacityMapping,
    FabricResourceBinding,
    FaultKind,
    PreservationMode,
    PreservationOption,
    PrivacyClass,
    RawLearningValueSample,
    ResourcePrices,
    ResourceVector,
    SchedulerConstraints,
    SchedulerFault,
    SchedulerPlan,
    SchedulerPolicy,
    SchedulerRequest,
    SchedulingInfeasibleError,
    ServingDemandSample,
    ServingSLO,
    ValuePrediction,
    WorkClass,
    WorkStatus,
    WorkUnit,
    capacity_from_fabric,
    compile_resource_plan,
    evaluate_learning_value,
    load_scheduler_request,
)

ROOT = Path(__file__).resolve().parents[2]
ZERO_DIGEST = "0" * 64


def _evidence(name: str, *sample_ids: str) -> EvidenceRef:
    return EvidenceRef(
        artifact_uri=f"raw://helix-scheduler/{name}",
        artifact_sha256=ZERO_DIGEST,
        sample_ids=tuple(sample_ids or (f"sample.{name}",)),
    )


def _resources(
    cpu: int,
    *,
    memory: int = 0,
    gpu: int = 0,
    storage: int = 0,
    iops: int = 0,
    network: int = 0,
) -> ResourceVector:
    return ResourceVector(
        cpu_millicores=cpu,
        memory_mib=memory,
        gpu_milliunits=gpu,
        storage_mib=storage,
        storage_iops=iops,
        network_mbps=network,
    )


def _vectors(*, serving: int = 10, rollout: int = 40) -> ClassResourceVectors:
    return ClassResourceVectors(
        serving=_resources(serving),
        rollout=_resources(rollout),
        environment=_resources(15),
        reward=_resources(20),
        verifier=_resources(25),
        training=_resources(50),
        evaluation=_resources(30),
    )


def _prediction(work_id: str, value: float) -> ValuePrediction:
    return ValuePrediction(
        value=value,
        model_id="value-model",
        model_version="value-model/v1",
        evidence=_evidence(f"prediction.{work_id}"),
    )


def _work(
    work_id: str,
    *,
    branch: str = "branch.a",
    work_class: WorkClass = WorkClass.ROLLOUT,
    value: float = 5.0,
    duration: int = 2,
    arrival: int = 0,
    units: int = 1,
    tenant: str = "tenant.a",
    privacy: PrivacyClass = PrivacyClass.TENANT_PRIVATE,
    effect: EffectClass = EffectClass.PURE,
    preservation: tuple[PreservationOption, ...] | None = None,
) -> WorkUnit:
    return WorkUnit(
        work_id=work_id,
        branch_id=branch,
        work_class=work_class,
        tenant_id=tenant,
        privacy=privacy,
        effect=effect,
        arrival_tick=arrival,
        duration_ticks=duration,
        deadline_tick=None,
        policy_age_ticks=0,
        resource_units=units,
        predicted_learning_value=_prediction(work_id, value),
        preservation=preservation
        if preservation is not None
        else (
            PreservationOption(
                mode=PreservationMode.RESTART,
                pause_ticks=0,
                checkpoint_interval_ticks=0,
                storage_mib_written=0,
                network_mib_transferred=0,
                cost_microunits=0,
            ),
        ),
    )


def _request(
    *,
    policy: SchedulerPolicy = SchedulerPolicy.HELIX_VALUE_AWARE,
    work: tuple[WorkUnit, ...] = (),
    horizon: int = 6,
    forecast_units: tuple[int, ...] | None = None,
    vectors: ClassResourceVectors | None = None,
    capacity: ResourceVector | None = None,
    reservation: ResourceVector | None = None,
    faults: tuple[SchedulerFault, ...] = (),
    lending: bool = True,
    branch_limit: int = 8,
    per_work_preemptions: int = 2,
    total_preemptions: int = 8,
    budget: int = 1_000_000,
    seed: int = 19,
) -> SchedulerRequest:
    actual_vectors = vectors or _vectors()
    units = forecast_units or tuple(2 for _ in range(horizon))
    static_limits = (
        ClassResourceVectors(
            serving=reservation or _resources(40),
            rollout=_resources(40),
            environment=_resources(20),
            reward=_resources(20),
            verifier=_resources(30),
            training=_resources(50),
            evaluation=_resources(30),
        )
        if policy is SchedulerPolicy.STATIC
        else None
    )
    return SchedulerRequest(
        request_id="scheduler.test",
        seed=seed,
        policy=policy,
        horizon_ticks=horizon,
        capacity=capacity or _resources(100),
        resource_vectors=actual_vectors,
        serving_slo=ServingSLO(
            reserved_capacity=reservation or _resources(40),
            maximum_predicted_latency_ms=100.0,
            maximum_predicted_queue_depth=20,
        ),
        serving_forecast=tuple(
            ServingDemandSample(
                tick=tick,
                resource_units=unit,
                predicted_latency_ms=20.0 + tick,
                predicted_queue_depth=unit,
                evidence=_evidence(f"serving.{tick}"),
            )
            for tick, unit in enumerate(units)
        ),
        work=work,
        constraints=SchedulerConstraints(
            max_budget_microunits=budget,
            prices=ResourcePrices(
                cpu_millicore_tick=1,
                memory_mib_tick=0,
                gpu_milliunit_tick=0,
                storage_mib_tick=0,
                storage_iop_tick=0,
                network_mbps_tick=0,
            ),
            max_policy_staleness_ticks=100,
            allowed_tenant_ids=("tenant.a",),
            maximum_privacy=PrivacyClass.TENANT_PRIVATE,
            allowed_effects=(EffectClass.PURE, EffectClass.READ_ONLY),
            max_selected_branches=branch_limit,
            max_preemptions_per_work=per_work_preemptions,
            max_total_preemptions=total_preemptions,
            capacity_lending=lending,
        ),
        static_limits=static_limits,
        faults=faults,
        max_audit_records=100_000,
    )


def _fault(
    kind: FaultKind,
    *,
    start: int = 0,
    end: int = 1,
    magnitude: float = 0.1,
    direction: int = 1,
    target: str | None = None,
) -> SchedulerFault:
    return SchedulerFault(
        fault_id=f"fault.{kind.value}",
        kind=kind,
        start_tick=start,
        end_tick=end,
        magnitude=magnitude,
        direction=direction,  # type: ignore[arg-type]
        target_work_id=target,
        evidence=_evidence(f"fault.{kind.value}"),
    )


def test_strict_models_and_explicit_resource_vectors() -> None:
    request = _request(work=(_work("work.a"),))
    assert tuple(request.resource_vectors.for_class(kind) for kind in WorkClass) == (
        request.resource_vectors.serving,
        request.resource_vectors.rollout,
        request.resource_vectors.environment,
        request.resource_vectors.reward,
        request.resource_vectors.verifier,
        request.resource_vectors.training,
        request.resource_vectors.evaluation,
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResourceVector.model_validate(
            {
                **_resources(1).model_dump(),
                "fabricated_accelerator": 1,
            }
        )
    with pytest.raises(ValidationError, match="valid integer"):
        ResourceVector.model_validate(
            {
                **_resources(1).model_dump(),
                "cpu_millicores": 1.5,
            }
        )


def test_helix_selects_branch_count_and_highest_value_density_deterministically() -> None:
    high = _work("work.high", branch="branch.high", value=20.0)
    low = _work("work.low", branch="branch.low", value=1.0)
    request = _request(work=(low, high), branch_limit=1)
    first = compile_resource_plan(request)
    second = compile_resource_plan(request)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert first.selected_branch_ids == ("branch.high",)
    assert first.completed_work_ids == ("work.high",)
    rejected = {item.work_id: item for item in first.outcomes}
    assert rejected["work.low"].status is WorkStatus.REJECTED
    assert "branch-count" in rejected["work.low"].reason
    assert all(tick.serving_slo_satisfied for tick in first.ticks)


@pytest.mark.parametrize(
    "policy",
    [
        SchedulerPolicy.DEDICATED,
        SchedulerPolicy.STATIC,
        SchedulerPolicy.UTILIZATION,
        SchedulerPolicy.FIFO,
        SchedulerPolicy.HELIX_VALUE_AWARE,
    ],
)
def test_baseline_and_helix_policies_share_hard_feasibility(policy: SchedulerPolicy) -> None:
    plan = compile_resource_plan(
        _request(
            policy=policy,
            work=(
                _work("work.rollout", work_class=WorkClass.ROLLOUT),
                _work("work.reward", work_class=WorkClass.REWARD, value=8.0),
            ),
        )
    )
    assert len(plan.ticks) == 6
    assert all(
        tick.serving_resources.add(tick.learning_resources).fits_within(tick.effective_capacity)
        for tick in plan.ticks
    )
    assert plan.budget.total_microunits <= plan.budget.limit_microunits
    assert tuple(decision.sequence for decision in plan.decisions) == tuple(
        range(len(plan.decisions))
    )


def test_capacity_lending_reclamation_and_preservation_alternatives() -> None:
    preservation = (
        PreservationOption(
            mode=PreservationMode.RESTART,
            pause_ticks=0,
            checkpoint_interval_ticks=0,
            storage_mib_written=0,
            network_mib_transferred=0,
            cost_microunits=0,
        ),
        PreservationOption(
            mode=PreservationMode.CHECKPOINT,
            pause_ticks=1,
            checkpoint_interval_ticks=2,
            storage_mib_written=4,
            network_mib_transferred=0,
            cost_microunits=2,
            method_evidence=_evidence("preservation.checkpoint"),
        ),
        PreservationOption(
            mode=PreservationMode.CONTINUUM,
            pause_ticks=0,
            checkpoint_interval_ticks=0,
            storage_mib_written=1,
            network_mib_transferred=1,
            cost_microunits=3,
            method_evidence=_evidence("preservation.continuum"),
        ),
    )
    request = _request(
        work=(_work("work.long", duration=4, preservation=preservation),),
        horizon=6,
        vectors=_vectors(serving=10, rollout=70),
        forecast_units=(2, 2, 8, 2, 2, 2),
        reservation=_resources(80),
    )
    plan = compile_resource_plan(request)
    assert not plan.ticks[0].lent_capacity.is_zero()
    assert not plan.ticks[2].reclaimed_capacity.is_zero()
    assert len(plan.preemptions) == 1
    preemption = plan.preemptions[0]
    assert preemption.selected_mode is PreservationMode.CONTINUUM
    assert {item.mode for item in preemption.alternatives} == set(PreservationMode)
    assert preemption.selected.preserved_work_ticks == 2
    assert preemption.selected.lost_work_ticks == 0
    assert preemption.selected.method_evidence == _evidence("preservation.continuum")
    outcome = next(item for item in plan.outcomes if item.work_id == "work.long")
    assert outcome.status is WorkStatus.COMPLETED
    assert outcome.preemptions == 1
    assert plan.budget.preservation_microunits == 3
    assert any(item.kind is DecisionKind.RECLAIM_CAPACITY for item in plan.decisions)


def test_no_lending_and_bounded_preemption_fail_closed() -> None:
    no_lending = compile_resource_plan(
        _request(
            work=(_work("work.large", duration=1),),
            vectors=_vectors(serving=10, rollout=70),
            reservation=_resources(80),
            lending=False,
        )
    )
    outcome = next(item for item in no_lending.outcomes if item.work_id == "work.large")
    assert outcome.status is WorkStatus.DEFERRED
    assert all(tick.lent_capacity.is_zero() for tick in no_lending.ticks)

    with pytest.raises(SchedulingInfeasibleError, match="bounded_preemption"):
        compile_resource_plan(
            _request(
                work=(_work("work.pinned", duration=4),),
                horizon=5,
                vectors=_vectors(serving=10, rollout=70),
                forecast_units=(2, 2, 8, 2, 2),
                reservation=_resources(80),
                per_work_preemptions=0,
                total_preemptions=0,
            )
        )


def test_hard_slo_budget_staleness_privacy_and_effect_constraints() -> None:
    illegal = (
        _work("work.tenant", tenant="tenant.other"),
        _work("work.privacy", privacy=PrivacyClass.RESTRICTED),
        _work("work.effect", effect=EffectClass.EXTERNAL),
    )
    plan = compile_resource_plan(_request(work=illegal))
    assert all(item.status is WorkStatus.REJECTED for item in plan.outcomes)
    assert not plan.completed_work_ids

    latency_request = _request()
    bad_sample = latency_request.serving_forecast[0].model_copy(
        update={"predicted_latency_ms": 101.0}
    )
    latency_request = latency_request.model_copy(
        update={"serving_forecast": (bad_sample, *latency_request.serving_forecast[1:])}
    )
    with pytest.raises(SchedulingInfeasibleError, match="serving_latency_slo"):
        compile_resource_plan(latency_request)

    with pytest.raises(SchedulingInfeasibleError, match="serving_budget"):
        compile_resource_plan(_request(budget=1))

    reservation_fault = _fault(FaultKind.CPU_EXHAUSTION, magnitude=0.3)
    with pytest.raises(SchedulingInfeasibleError, match="serving_reservation_slo"):
        compile_resource_plan(
            _request(
                reservation=_resources(80),
                capacity=_resources(100),
                faults=(reservation_fault,),
            )
        )

    stale_request = _request(work=(_work("work.stale", duration=3),))
    stale_request = stale_request.model_copy(
        update={
            "constraints": stale_request.constraints.model_copy(
                update={"max_policy_staleness_ticks": 2}
            )
        }
    )
    stale_plan = compile_resource_plan(stale_request)
    assert stale_plan.outcomes[0].status is WorkStatus.REJECTED
    assert "staleness" in stale_plan.outcomes[0].reason


def test_all_deterministic_fault_inputs_are_audited_and_value_error_is_separate() -> None:
    fault_work = _work("work.fault", value=10.0, duration=1)
    faults = (
        _fault(FaultKind.TRAFFIC_SPIKE, magnitude=0.1),
        _fault(FaultKind.GPU_LOSS, magnitude=0.2),
        _fault(FaultKind.CPU_EXHAUSTION, magnitude=0.1),
        _fault(FaultKind.STORAGE_SLOWDOWN, magnitude=0.3),
        _fault(FaultKind.NETWORK_SLOWDOWN, magnitude=0.4),
        _fault(
            FaultKind.VALUE_PREDICTION_ERROR,
            magnitude=0.5,
            direction=-1,
            target="work.fault",
        ),
    )
    plan = compile_resource_plan(
        _request(
            work=(fault_work,),
            faults=faults,
            capacity=_resources(120, memory=10, gpu=1000, storage=100, iops=100, network=100),
        )
    )
    applied = {
        decision.subject_id
        for decision in plan.decisions
        if decision.kind is DecisionKind.APPLY_FAULT
    }
    assert applied == {fault.fault_id for fault in faults}
    assert plan.predicted_learning_value == 10.0
    assert plan.scheduler_adjusted_predicted_value == 5.0
    assert plan.ticks[0].effective_capacity.cpu_millicores == 108
    assert plan.ticks[0].effective_capacity.gpu_milliunits == 800
    assert plan.ticks[0].effective_capacity.storage_iops == 70
    assert plan.ticks[0].effective_capacity.network_mbps == 60


def test_predicted_vs_observed_learning_value_preserves_raw_samples() -> None:
    plan = compile_resource_plan(_request(work=(_work("work.value", value=10.0),)))
    samples = (
        RawLearningValueSample(
            sample_id="observed.1",
            work_id="work.value",
            seed=41,
            value=6.0,
            observed_at_tick=10,
            evidence=_evidence("observed.1", "observed.1"),
        ),
        RawLearningValueSample(
            sample_id="observed.2",
            work_id="work.value",
            seed=73,
            value=10.0,
            observed_at_tick=11,
            evidence=_evidence("observed.2", "observed.2"),
        ),
    )
    evaluation = evaluate_learning_value(plan, samples, seed=44)
    assert evaluation == evaluate_learning_value(plan, samples, seed=44)
    assert evaluation.predicted_total_for_compared_work == 10.0
    assert evaluation.observed_total == 8.0
    assert evaluation.signed_error_total == -2.0
    assert evaluation.mean_absolute_error == 2.0
    assert evaluation.observation_seeds == (41, 73)
    assert evaluation.raw_sample_count == 2
    assert evaluation.comparisons[0].raw_sample_ids == ("observed.1", "observed.2")
    assert evaluation.comparisons[0].sample_seeds == (41, 73)
    assert evaluation.comparisons[0].observed_mean_lower_95 < 0.0
    assert evaluation.comparisons[0].observed_mean_upper_95 > 1.0
    assert not evaluation.missing_observation_work_ids

    duplicate_seed = samples[1].model_copy(update={"seed": 41})
    with pytest.raises(ValueError, match="at most one observation per seed"):
        evaluate_learning_value(plan, (samples[0], duplicate_seed), seed=44)


def test_checked_in_scenario_loads_strictly_and_compiles() -> None:
    path = ROOT / "scenarios/helix/resource/cpu-learning-aware.json"
    request = load_scheduler_request(path)
    plan = compile_resource_plan(request)
    assert request.seed == 1701
    assert plan.request_id == "helix-resource-reference"
    assert plan.policy is SchedulerPolicy.HELIX_VALUE_AWARE
    assert all(tick.serving_slo_satisfied for tick in plan.ticks)
    assert json.loads(plan.model_dump_json())["schema_version"] == (
        "sloforge.helix.scheduler-plan/v1"
    )
    references = (
        {sample.evidence for sample in request.serving_forecast}
        | {work.predicted_learning_value.evidence for work in request.work}
        | {fault.evidence for fault in request.faults}
        | {
            option.method_evidence
            for work in request.work
            for option in work.preservation
            if option.method_evidence is not None
        }
    )
    for reference in references:
        raw_path = path.parent / reference.artifact_uri.rsplit("/", 1)[-1]
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == reference.artifact_sha256


def test_plan_identity_is_tamper_evident() -> None:
    plan = compile_resource_plan(_request(work=(_work("work.identity"),)))
    payload = plan.model_dump()
    payload["scheduler_adjusted_predicted_value"] = 99.0
    with pytest.raises(ValidationError, match="scheduler-adjusted value"):
        SchedulerPlan.model_validate(payload)


def test_fabric_adapter_requires_explicit_unit_mapping() -> None:
    curve = ServiceCurve(
        id="curve.cpu-0",
        points=(ServiceCurvePoint(message_bytes=0, latency_us=1.0, bandwidth_gbps=1.0),),
        provenance=CalibrationProvenance(
            kind=ProvenanceKind.ANALYTICAL,
            artifact_uri="raw://fabric/cpu-0",
            artifact_sha256=ZERO_DIGEST,
            environment_fingerprint="cpu-test",
            collected_at="2026-08-03T00:00:00Z",
        ),
    )
    fabric = FabricSimulationRequest(
        seed=7,
        resources=(
            PhysicalResource(
                id="cpu-0",
                kind=ResourceKind.CPU_CORE_GROUP,
                scheduling=SchedulingMode.FAIR_SHARE,
                capacity_units=8.0,
                max_concurrency=8,
                curve=curve,
            ),
        ),
        operations=(),
    )
    mapping = FabricCapacityMapping(
        bindings=(
            FabricResourceBinding(
                fabric_resource_id="cpu-0", helix_capacity=_resources(8000, memory=16384)
            ),
        )
    )
    assert capacity_from_fabric(fabric, mapping) == _resources(8000, memory=16384)
    unknown = FabricCapacityMapping(
        bindings=(
            FabricResourceBinding(
                fabric_resource_id="cpu-missing", helix_capacity=_resources(1000)
            ),
        )
    )
    with pytest.raises(ValueError, match="unknown physical resources"):
        capacity_from_fabric(fabric, unknown)
