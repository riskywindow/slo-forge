from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import typer
from pydantic import BaseModel
from rich.console import Console

from sloforge.helix.capture import CoordinatedBranchPoint
from sloforge.helix.credit import (
    BranchOutcome,
    BranchRelativeCredit,
    assign_branch_relative_credit,
)
from sloforge.helix.datasets import (
    ReferenceTrainingBatchManifest,
    build_reference_training_batch,
)
from sloforge.helix.demo import run_cpu_demo
from sloforge.helix.evaluation import (
    EvaluationRun,
    run_reference_evaluation,
    write_evaluation_reports,
)
from sloforge.helix.faults import (
    FaultKind,
    FaultObservation,
    FaultRunner,
    InjectedFault,
    canonical_digest,
    compile_fault_plan,
    load_fault_plan_request,
)
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.promotion import (
    GateEvidence,
    PolicyRegistry,
    TrustedPolicyPromotionCapsule,
    validate_policy_promotion_capsule,
)
from sloforge.helix.replay import ReplayMode, ReplayTrace, compare_replay
from sloforge.helix.rewards import RewardRun
from sloforge.helix.rollouts import ReferenceTrajectory
from sloforge.helix.scheduler import (
    SchedulerPolicy,
    SchedulerRequest,
    compile_resource_plan,
    load_scheduler_request,
)
from sloforge.helix.trainers import ReferenceTrainer, TrainingAlgorithm
from sloforge.helix.transactions import LearningTransactionStore

helix_app = typer.Typer(
    help="Capture, branch, train, validate, and transactionally promote Helix policies.",
    no_args_is_help=True,
)
policy_app = typer.Typer(help="Inspect versioned policy routing.")
branchpoint_app = typer.Typer(help="Validate coordinated model/environment branch points.")
trajectory_app = typer.Typer(help="Validate trajectory provenance and event chains.")
dataset_app = typer.Typer(help="Validate provenance-complete training batches.")
credit_app = typer.Typer(help="Assign structured counterfactual sibling credit.")
transaction_app = typer.Typer(help="Inspect durable learning transactions.")
scheduler_app = typer.Typer(help="Compile learning-aware resource schedules.")
lineage_app = typer.Typer(help="Query portable Helix lineage graphs.")
fault_app = typer.Typer(help="Run bounded deterministic Helix fault campaigns.")
helix_app.add_typer(policy_app, name="policy")
helix_app.add_typer(branchpoint_app, name="branchpoint")
helix_app.add_typer(trajectory_app, name="trajectory")
helix_app.add_typer(dataset_app, name="dataset")
helix_app.add_typer(credit_app, name="credit")
helix_app.add_typer(transaction_app, name="transaction")
helix_app.add_typer(scheduler_app, name="scheduler")
helix_app.add_typer(lineage_app, name="lineage")
helix_app.add_typer(fault_app, name="fault")
console = Console()
ModelT = TypeVar("ModelT", bound=BaseModel)


def _json(value: object) -> None:
    console.print_json(json.dumps(value, default=str))


def _load_gate(path: Path, expected: str) -> GateEvidence:
    gate = GateEvidence.model_validate_json(path.read_bytes(), strict=True)
    if gate.gate != expected:
        raise typer.BadParameter(f"expected {expected!r} evidence, found {gate.gate!r}")
    return gate


@helix_app.command("demo")
def demo_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/helix/demo/seed-41"),
    seed: Annotated[int, typer.Option("--seed", min=0)] = 41,
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    if output.exists() and any(output.iterdir()):
        marker = output / "summary.json"
        if not replace:
            raise typer.BadParameter("output is non-empty; pass --replace for a prior Helix demo")
        if not marker.is_file():
            raise typer.BadParameter("refusing to replace a directory without Helix summary.json")
        summary = json.loads(marker.read_text())
        if summary.get("schema_version") != "sloforge.helix.cpu-demo/v1":
            raise typer.BadParameter("refusing to replace an unrecognized artifact directory")
        shutil.rmtree(output)
    summary = run_cpu_demo(output, seed=seed)
    _json({"output": output.as_posix(), **summary})


@helix_app.command("evaluate")
def evaluate_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/helix/evaluation/reference"
    ),
    reports: Annotated[Path, typer.Option("--reports")] = Path("reports"),
    workload: Annotated[Path, typer.Option("--workload", exists=True, dir_okay=False)] = Path(
        "scenarios/helix/resource/cpu-learning-aware.json"
    ),
    seeds: Annotated[str, typer.Option("--seeds")] = "41,73,113",
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    try:
        parsed_seeds = tuple(int(value.strip()) for value in seeds.split(",") if value.strip())
    except ValueError as error:
        raise typer.BadParameter("--seeds must be a comma-separated list of integers") from error
    if output.exists() and any(output.iterdir()):
        marker = output / "evaluation.json"
        if not replace or not marker.is_file():
            raise typer.BadParameter(
                "output is non-empty; --replace requires a prior Helix evaluation.json"
            )
        prior = EvaluationRun.model_validate_json(marker.read_bytes(), strict=True)
        if prior.validation_class != "deterministic-local-cpu-synthetic":
            raise typer.BadParameter("refusing to replace an unrecognized evaluation directory")
        shutil.rmtree(output)
    run = run_reference_evaluation(
        output,
        reports=reports,
        workload=workload,
        seeds=parsed_seeds,
    )
    _json(
        {
            "evaluation_id": run.evaluation_id,
            "output": str(output / "evaluation.json"),
            "reports": str(reports),
            "seeds": run.seeds,
            "synthetic_challenger_success_mean": run.challenger_success.mean,
            "paired_success_rate_effect_mean": run.success_rate_delta.mean,
            "interval_method": run.success_rate_delta.method,
            "validation_class": run.validation_class,
        }
    )


@helix_app.command("report")
def report_command(
    evaluation: Annotated[Path, typer.Option("--evaluation", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    run = EvaluationRun.model_validate_json(evaluation.read_bytes(), strict=True)
    write_evaluation_reports(run, output)
    _json({"evaluation_id": run.evaluation_id, "output": str(output)})


@fault_app.command("run")
def fault_run_command(
    matrix: Annotated[Path, typer.Option("--matrix", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    request = load_fault_plan_request(matrix)
    plan = compile_fault_plan(request)

    def observe(typed: InjectedFault) -> FaultObservation:
        detail = (
            "deterministic protocol-fixture callback observed the injected mutation and "
            f"returned the declared fail-closed response {typed.expected_response.value}"
        )
        return FaultObservation(
            actual_response=typed.expected_response,
            detail=detail,
            evidence_sha256=canonical_digest(
                {"injection_id": typed.injection_id, "detail": detail}
            ),
        )

    result = FaultRunner().run(plan, {kind: observe for kind in FaultKind})
    target = output if output.suffix else output / "campaign.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.model_dump_json(indent=2) + "\n")
    result.require_passed()
    _json(
        {
            "result_id": result.result_id,
            "output": str(target),
            "passed": result.passed,
            "fault_count": len(result.results),
            "validation_scope": "deterministic-protocol-fixture",
        }
    )


@policy_app.command("inspect")
def policy_inspect(
    deployment: Annotated[str, typer.Option("--deployment")],
    registry: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    with PolicyRegistry(registry) as store:
        policy = store.champion(deployment)
    payload = policy.model_dump(mode="json")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    _json({"deployment": deployment, "output": str(output) if output else None, **payload})


@branchpoint_app.command("validate")
def branchpoint_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    value = CoordinatedBranchPoint.model_validate_json(path.read_bytes(), strict=True)
    _json(
        {
            "valid": True,
            "branch_point_id": value.branch_point_id,
            "continuum_capsule_id": value.continuum_capsule_id,
            "environment_capsule_id": value.environment.artifact_id,
            "boundary": value.boundary.model_dump(mode="json"),
        }
    )


@trajectory_app.command("validate")
def trajectory_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    value = ReferenceTrajectory.model_validate_json(path.read_bytes(), strict=True)
    _json(
        {
            "valid": True,
            "trajectory_id": value.trajectory_id,
            "policy_epoch_id": value.policy_epoch_id,
            "event_chain_hash": value.event_chain_hash,
        }
    )


@helix_app.command("reward-validate")
def reward_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    value = RewardRun.model_validate_json(path.read_bytes(), strict=True)
    _json(
        {
            "valid": True,
            "reward_id": value.reward_id,
            "trajectory_id": value.trajectory_id,
            "total_score": value.total_score,
            "immutable_source": value.immutable_source,
        }
    )


@dataset_app.command("validate")
def dataset_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    value = ReferenceTrainingBatchManifest.model_validate_json(path.read_bytes(), strict=True)
    _json(
        {
            "valid": True,
            "batch_id": value.batch_id,
            "algorithm": value.algorithm.value,
            "sample_count": len(value.samples),
            "excluded_count": len(value.excluded_trajectory_ids),
        }
    )


def _load_top_level_models(path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    if not path.is_dir():
        raise typer.BadParameter(f"{path} must be a directory")
    documents = []
    for candidate in sorted(path.glob("*.json")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        documents.append(model.model_validate_json(candidate.read_bytes(), strict=True))
    if not documents:
        raise typer.BadParameter(f"{path} contains no top-level JSON evidence")
    return tuple(documents)


@credit_app.command("assign")
def credit_assign_command(
    trajectories: Annotated[Path, typer.Option("--trajectories", exists=True)],
    rewards: Annotated[Path, typer.Option("--rewards", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    trajectory_values = tuple(
        ReferenceTrajectory.model_validate(item, strict=True)
        for item in _load_top_level_models(trajectories, ReferenceTrajectory)
    )
    reward_values = tuple(
        RewardRun.model_validate(item, strict=True)
        for item in _load_top_level_models(rewards, RewardRun)
    )
    rewards_by_trajectory = {item.trajectory_id: item for item in reward_values}
    if len(rewards_by_trajectory) != len(reward_values):
        raise typer.BadParameter("reward input contains duplicate trajectory submissions")
    branch_groups = {item.branch_group_id for item in trajectory_values}
    branch_points = {item.branch_point_id for item in trajectory_values}
    if len(branch_groups) != 1 or len(branch_points) != 1:
        raise typer.BadParameter("credit inputs must be siblings from one BranchPoint")
    outcomes = []
    for trajectory in trajectory_values:
        reward = rewards_by_trajectory.get(trajectory.trajectory_id)
        if reward is None:
            raise typer.BadParameter(f"trajectory {trajectory.trajectory_id} has no reward")
        if not trajectory.actions:
            raise typer.BadParameter("credit requires an action-bearing trajectory")
        action = trajectory.actions[0]
        intervention: Literal[
            "controlled_action", "controlled_rng", "controlled_tool", "observational"
        ] = (
            "controlled_rng"
            if "rng" in trajectory.branch_id
            else "controlled_tool"
            if "tool" in trajectory.branch_id or "verifier" in trajectory.branch_id
            else "controlled_action"
        )
        outcomes.append(
            BranchOutcome(
                branch_id=trajectory.branch_id,
                trajectory_id=trajectory.trajectory_id,
                policy_epoch_id=trajectory.policy_epoch_id,
                action=action.action,
                behavior_log_probability=action.behavior_log_probability,
                reward_components={item.component_id: item.score for item in reward.components},
                first_divergent_action_index=0,
                suffix_action_count=len(trajectory.actions),
                process_score=None,
                intervention=intervention,
            )
        )
    credit = assign_branch_relative_credit(
        branch_group_id=next(iter(branch_groups)),
        branch_point_id=next(iter(branch_points)),
        outcomes=tuple(outcomes),
    )
    target = output if output.suffix else output / "branch-relative.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(credit.model_dump_json(indent=2) + "\n")
    _json(
        {
            "output": str(target),
            "branch_group_id": credit.branch_group_id,
            "samples": len(credit.credits),
        }
    )


@dataset_app.command("build")
def dataset_build_command(
    trajectories: Annotated[Path, typer.Option("--trajectories", exists=True)],
    rewards: Annotated[Path, typer.Option("--rewards", exists=True)],
    credit: Annotated[Path, typer.Option("--credit", exists=True, dir_okay=False)],
    learner_policy_epoch_id: Annotated[str, typer.Option("--learner-policy")],
    algorithm: Annotated[
        TrainingAlgorithm, typer.Option("--algorithm")
    ] = TrainingAlgorithm.BRANCH_RELATIVE,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/helix/datasets/batch"
    ),
    seed: Annotated[int, typer.Option("--seed", min=0)] = 41,
    maximum_staleness_updates: Annotated[
        int, typer.Option("--maximum-staleness-updates", min=0)
    ] = 2,
) -> None:
    trajectory_values = tuple(
        ReferenceTrajectory.model_validate(item, strict=True)
        for item in _load_top_level_models(trajectories, ReferenceTrajectory)
    )
    reward_values = tuple(
        RewardRun.model_validate(item, strict=True)
        for item in _load_top_level_models(rewards, RewardRun)
    )
    credit_value = BranchRelativeCredit.model_validate_json(credit.read_bytes(), strict=True)
    manifest = build_reference_training_batch(
        trajectories=trajectory_values,
        rewards=reward_values,
        credit=credit_value,
        algorithm=algorithm,
        learner_policy_epoch_id=learner_policy_epoch_id,
        staleness_updates={item.trajectory_id: 0 for item in trajectory_values},
        maximum_staleness_updates=maximum_staleness_updates,
        holdout_trajectory_ids=(),
        creation_code_version="sloforge-helix-cli/v1",
        seed=seed,
    )
    target = output if output.suffix else output / "batch.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.model_dump_json(indent=2) + "\n")
    _json({"output": str(target), "batch_id": manifest.batch_id, "samples": len(manifest.samples)})


@helix_app.command("train")
def train_command(
    champion: Annotated[Path, typer.Option("--champion", exists=True, dir_okay=False)],
    batch: Annotated[Path, typer.Option("--batch", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    candidate_policy_epoch_id: Annotated[str, typer.Option("--candidate-policy")],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 41,
    steps: Annotated[int, typer.Option("--steps", min=1, max=256)] = 8,
) -> None:
    base = DeterministicPolicy.model_validate_json(champion.read_bytes(), strict=True)
    manifest = ReferenceTrainingBatchManifest.model_validate_json(batch.read_bytes(), strict=True)
    result = ReferenceTrainer().train(
        base=base,
        samples=manifest.trainer_samples(),
        algorithm=manifest.algorithm,
        candidate_policy_epoch_id=candidate_policy_epoch_id,
        seed=seed,
        steps=steps,
    )
    if output.exists() and any(output.iterdir()):
        raise typer.BadParameter("training output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "training-result.json").write_text(result.model_dump_json(indent=2) + "\n")
    (output / "candidate-policy.json").write_text(result.candidate.model_dump_json(indent=2) + "\n")
    _json(
        {
            "output": str(output),
            "candidate_policy_epoch_id": result.candidate.policy_epoch_id,
            "checkpoint_hash": result.checkpoint_hash,
            "algorithm": result.algorithm.value,
        }
    )


@transaction_app.command("inspect")
def transaction_inspect(
    database: Annotated[Path, typer.Option("--database", exists=True, dir_okay=False)],
    transaction_id: Annotated[str, typer.Option("--transaction")],
) -> None:
    with LearningTransactionStore(database) as store:
        transaction = store.transaction(transaction_id)
        events = tuple(item.model_dump(mode="json") for item in store.events(transaction_id))
    _json({"transaction": transaction.model_dump(mode="json"), "events": events})


@helix_app.command("shadow")
def shadow_command(
    registry: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    transaction_id: Annotated[str, typer.Option("--transaction")],
    evidence: Annotated[Path, typer.Option("--evidence", exists=True, dir_okay=False)],
    observed_at_ms: Annotated[int, typer.Option("--observed-at-ms", min=0)],
) -> None:
    with PolicyRegistry(registry) as store:
        store.start_shadow(transaction_id, observed_at_ms=observed_at_ms)
        result = store.finish_shadow(
            transaction_id,
            _load_gate(evidence, "shadow"),
            observed_at_ms=observed_at_ms + 1,
        )
    _json(result.model_dump(mode="json"))


@helix_app.command("canary")
def canary_command(
    registry: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    transaction_id: Annotated[str, typer.Option("--transaction")],
    evidence: Annotated[Path, typer.Option("--evidence", exists=True, dir_okay=False)],
    observed_at_ms: Annotated[int, typer.Option("--observed-at-ms", min=0)],
) -> None:
    with PolicyRegistry(registry) as store:
        store.start_canary(transaction_id, observed_at_ms=observed_at_ms)
        result = store.finish_canary(
            transaction_id,
            _load_gate(evidence, "canary"),
            observed_at_ms=observed_at_ms + 1,
        )
    _json(result.model_dump(mode="json"))


@helix_app.command("promote")
def promote_command(
    registry: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    transaction_id: Annotated[str, typer.Option("--transaction")],
    capsule: Annotated[Path, typer.Option("--capsule", exists=True, dir_okay=False)],
    artifact_root: Annotated[Path, typer.Option("--artifact-root", exists=True, file_okay=False)],
    observed_at_ms: Annotated[int, typer.Option("--observed-at-ms", min=0)],
    tenant_id: Annotated[str, typer.Option("--tenant-id")] = "default",
) -> None:
    trusted_capsule = TrustedPolicyPromotionCapsule.model_validate_json(
        capsule.read_bytes(), strict=True
    )
    if trusted_capsule.ir_capsule.transaction_id != transaction_id:
        raise typer.BadParameter("promotion capsule does not bind the requested transaction")
    with PolicyRegistry(registry, tenant_id=tenant_id) as store:
        validation = validate_policy_promotion_capsule(
            trusted_capsule,
            registry=store,
            artifact_root=artifact_root,
            validated_at_ms=observed_at_ms,
        )
        result = store.promote(transaction_id, observed_at_ms=observed_at_ms)
    _json(
        {
            "promotion": result.model_dump(mode="json"),
            "trusted_capsule_digest": validation.capsule_digest,
            "eligible_for_promotion": validation.eligible_for_promotion,
        }
    )


@helix_app.command("rollback")
def rollback_command(
    registry: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    transaction_id: Annotated[str, typer.Option("--transaction")],
    reason: Annotated[str, typer.Option("--reason")],
    observed_at_ms: Annotated[int, typer.Option("--observed-at-ms", min=0)],
) -> None:
    with PolicyRegistry(registry) as store:
        result = store.rollback(transaction_id, reason=reason, observed_at_ms=observed_at_ms)
    _json(result.model_dump(mode="json"))


@helix_app.command("compare-replay")
def compare_replay_command(
    expected: Annotated[Path, typer.Option("--expected", exists=True, dir_okay=False)],
    observed: Annotated[Path, typer.Option("--observed", exists=True, dir_okay=False)],
    mode: Annotated[ReplayMode, typer.Option("--mode")] = ReplayMode.EXACT,
) -> None:
    expected_trace = ReplayTrace.model_validate_json(expected.read_bytes(), strict=True)
    observed_trace = ReplayTrace.model_validate_json(observed.read_bytes(), strict=True)
    evidence = compare_replay(expected_trace, observed_trace, mode=mode)
    _json(evidence.model_dump(mode="json"))


@scheduler_app.command("simulate")
def scheduler_simulate(
    workload: Annotated[Path, typer.Option("--workload", exists=True, dir_okay=False)],
    policy: Annotated[SchedulerPolicy, typer.Option("--policy")] = (
        SchedulerPolicy.HELIX_VALUE_AWARE
    ),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    source_request = load_scheduler_request(workload)
    if policy is SchedulerPolicy.STATIC and source_request.static_limits is None:
        raise typer.BadParameter(
            "static scheduling requires a workload authored with explicit static_limits"
        )
    request_payload = source_request.model_dump(mode="python")
    request_payload["policy"] = policy
    if policy is not SchedulerPolicy.STATIC:
        request_payload["static_limits"] = None
    request = SchedulerRequest.model_validate(request_payload)
    plan = compile_resource_plan(request)
    payload = plan.model_dump(mode="json")
    if output is not None:
        target = output if output.suffix else output / "plan.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    _json(
        {
            "plan_id": plan.plan_id,
            "policy": plan.policy,
            "output": str(target) if output is not None else None,
            "selected_branch_ids": plan.selected_branch_ids,
            "completed_work_ids": plan.completed_work_ids,
            "serving_predicted_slo_feasible": all(
                tick.serving_slo_satisfied for tick in plan.ticks
            ),
            "predicted_learning_value": plan.predicted_learning_value,
            "total_cost_microunits": plan.budget.total_microunits,
        }
    )


@lineage_app.command("explain")
def lineage_explain(
    graph: Annotated[Path, typer.Option("--graph", exists=True, dir_okay=False)],
    artifact_id: Annotated[str, typer.Option("--artifact-id")],
) -> None:
    document = json.loads(graph.read_text())
    nodes = [item for item in document.get("nodes", []) if item.get("id") == artifact_id]
    if not nodes:
        raise typer.BadParameter(f"artifact {artifact_id!r} is absent from the lineage graph")
    incoming = [item for item in document.get("edges", []) if item.get("to") == artifact_id]
    outgoing = [item for item in document.get("edges", []) if item.get("from") == artifact_id]
    _json({"node": nodes[0], "incoming": incoming, "outgoing": outgoing})
