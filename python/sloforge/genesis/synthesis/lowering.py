"""Trusted lowering of typed synthesis choices into canonical genome deltas."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue

from sloforge.genesis.ir import (
    ArtifactDigest,
    CancellationBehavior,
    DecodeScheduling,
    EstimatedChange,
    Extensions,
    GenomePattern,
    InferenceGenome,
    LearnedConstraint,
    ProofObligation,
    QueueDiscipline,
    SemanticCategory,
    StateLayout,
    Transformation,
    TransformationDesignation,
    TransformationFamily,
    VerificationLevel,
    canonical_hash,
)
from sloforge.genesis.search import CandidateDesign, MutationChoice


@dataclass(frozen=True, slots=True)
class LoweredCandidate:
    design: CandidateDesign
    genome: InferenceGenome
    transformations: tuple[Transformation, ...]


def _batching_delta(
    baseline: InferenceGenome,
    design: CandidateDesign,
    mutation: MutationChoice,
    *,
    applied_transformation_ids: tuple[str, ...],
) -> tuple[InferenceGenome, Transformation]:
    if set(mutation.regions) != {"request", "serving"}:
        raise ValueError("deadline batching lowering requires request and serving regions")
    for region in (baseline.request, baseline.serving):
        if region.node.frozen:
            raise ValueError(f"cannot mutate frozen genome node {region.node.stable_id}")
        if "genesis.rules/baseline-exact-v1" not in region.node.legal_rewrite_rules:
            raise ValueError(
                f"genome node {region.node.stable_id} does not permit exact policy lowering"
            )
    parameters = {item.key: item.value for item in mutation.parameters}
    if set(parameters) != {"cancel_check_before_emit", "queue_policy"}:
        raise ValueError("deadline batching parameters do not match the lowering registry")
    if parameters["queue_policy"] != "deadline_bucket":
        raise ValueError("unsupported queue policy")
    if parameters["cancel_check_before_emit"] not in {"true", "false"}:
        raise ValueError("cancel_check_before_emit must be Boolean text")
    safe = parameters["cancel_check_before_emit"] == "true"
    expected_policy = "deadline_cancel_batch" if safe else "deadline_batch"
    existing_policy = baseline.request.node.extensions.root.get("sloforge.dev/synthesized-policy")
    if (
        baseline.request.queue_discipline is QueueDiscipline.EARLIEST_DEADLINE
        and baseline.request.cancellation_behavior
        is (CancellationBehavior.IMMEDIATE if safe else CancellationBehavior.SAFE_POINT)
        and baseline.serving.decode_scheduling is DecodeScheduling.SLO_SLACK
        and isinstance(existing_policy, dict)
        and existing_policy.get("policy") == expected_policy
        and existing_policy.get("cancel_check_before_emit") is safe
    ):
        raise ValueError("deadline batching mutation has no executable effect")

    source_hash = canonical_hash(baseline)
    prior_candidate = baseline.extensions.root.get("sloforge.dev/synthesis-candidate")
    root_parent_hash = (
        prior_candidate.get("parent_genome_hash") if isinstance(prior_candidate, dict) else None
    )
    if not isinstance(root_parent_hash, str):
        root_parent_hash = source_hash
    policy_extension: dict[str, JsonValue] = {
        "cancel_check_before_emit": safe,
        "candidate_id": design.candidate_id,
        "policy": expected_policy,
    }
    request_node = baseline.request.node.model_copy(
        update={
            "extensions": Extensions(
                root={
                    **baseline.request.node.extensions.root,
                    "sloforge.dev/synthesized-policy": policy_extension,
                }
            )
        }
    )
    serving_node = baseline.serving.node.model_copy(
        update={
            "extensions": Extensions(
                root={
                    **baseline.serving.node.extensions.root,
                    "sloforge.dev/synthesized-policy": policy_extension,
                }
            )
        }
    )
    request = baseline.request.model_copy(
        update={
            "node": request_node,
            "queue_discipline": QueueDiscipline.EARLIEST_DEADLINE,
            "cancellation_behavior": (
                CancellationBehavior.IMMEDIATE if safe else CancellationBehavior.SAFE_POINT
            ),
        }
    )
    serving = baseline.serving.model_copy(
        update={"node": serving_node, "decode_scheduling": DecodeScheduling.SLO_SLACK}
    )
    target = baseline.model_copy(
        update={
            "genome_id": f"{baseline.genome_id}-{design.candidate_id[-12:]}",
            "request": request,
            "serving": serving,
            "extensions": Extensions(
                root={
                    **baseline.extensions.root,
                    "sloforge.dev/synthesis-candidate": {
                        "candidate_id": design.candidate_id,
                        "parent_genome_hash": root_parent_hash,
                        "immediate_parent_genome_hash": source_hash,
                        "transformation_ids": list(applied_transformation_ids),
                    },
                }
            ),
        }
    )
    target_hash = canonical_hash(target)
    obligation = ProofObligation(
        obligation_id=f"{mutation.transformation_id}.cancellation",
        property="compiled policy prevents scheduling after cancellation and preserves token order",
        minimum_level=VerificationLevel.PROPERTY,
        scope="bounded deadline-batching runtime and declared reference corpus",
        assumptions=("generated code remains untrusted until independent replay",),
    )
    transformation = Transformation(
        transformation_id=mutation.transformation_id,
        family=TransformationFamily.BATCHING,
        source_pattern=GenomePattern(
            region="request",
            node_ids=(baseline.request.node.stable_id, baseline.serving.node.stable_id),
            structural_constraints=(f"source_genome_sha256 == {source_hash}",),
        ),
        target_pattern=GenomePattern(
            region="request",
            node_ids=(target.request.node.stable_id, target.serving.node.stable_id),
            structural_constraints=(f"target_genome_sha256 == {target_hash}",),
        ),
        semantic_category=SemanticCategory.POLICY,
        designation=TransformationDesignation.POLICY,
        preconditions=(
            "request and serving nodes are mutable",
            "genesis.rules/baseline-exact-v1 is legal on both nodes",
            "runtime supplies typed queue length, SLO slack, and cancellation state",
        ),
        postconditions=(
            "request queue discipline is earliest-deadline",
            "serving decode scheduling is SLO-slack adaptive",
            "policy bytecode digest is bound to candidate runtime evidence",
        ),
        expected_quality_cost=(),
        expected_resource_change=(),
        expected_performance_change=(
            EstimatedChange(
                metric="modeled_completion_objective_ratio",
                lower=-mutation.expected_upside,
                expected=-mutation.expected_upside,
                upper=0.0,
                unit="ratio",
            ),
        ),
        affected_regions=("request", "serving"),
        verification_obligations=(obligation,),
        required_verifier_stages=("bytecode_validation", "candidate_runtime_replay"),
        required_benchmark_stages=("deterministic_cpu_runtime",),
        rollback_strategy=f"restore genome {source_hash}",
        proposal_source=design.proposal_engine,
        parent_transformations=applied_transformation_ids[-2:-1],
        extensions=Extensions(
            root={
                "sloforge.dev/applied-delta": {
                    "source_genome_hash": source_hash,
                    "target_genome_hash": target_hash,
                    "parameters": [
                        {"key": key, "value": parameters[key]} for key in sorted(parameters)
                    ],
                }
            }
        ),
    )
    return target, transformation


def _state_layout_delta(
    baseline: InferenceGenome,
    design: CandidateDesign,
    mutation: MutationChoice,
    *,
    applied_transformation_ids: tuple[str, ...],
) -> tuple[InferenceGenome, Transformation]:
    """Lower the restricted bounded-paged state allocator transformation."""

    if mutation.regions != ("state",):
        raise ValueError("paged state lowering requires exactly the state region")
    nodes = (baseline.state.node, *(state.node for state in baseline.state.states))
    for node in nodes:
        if node.frozen:
            raise ValueError(f"cannot mutate frozen genome node {node.stable_id}")
        if "genesis.rules/baseline-exact-v1" not in node.legal_rewrite_rules:
            raise ValueError(f"genome node {node.stable_id} does not permit exact state lowering")
    parameters = {item.key: item.value for item in mutation.parameters}
    if set(parameters) != {"layout", "page_bytes"}:
        raise ValueError("paged state parameters do not match the lowering registry")
    if parameters["layout"] != StateLayout.PAGED.value:
        raise ValueError("unsupported state layout")
    try:
        page_bytes = int(parameters["page_bytes"])
    except ValueError as error:
        raise ValueError("page_bytes must be an integer") from error
    if not 16 <= page_bytes <= 65536 or page_bytes & (page_bytes - 1):
        raise ValueError("page_bytes must be a power of two in [16, 65536]")

    existing_layout = baseline.state.node.extensions.root.get(
        "sloforge.dev/synthesized-state-layout"
    )
    if (
        all(state.layout is StateLayout.PAGED for state in baseline.state.states)
        and isinstance(existing_layout, dict)
        and existing_layout.get("layout") == StateLayout.PAGED.value
        and existing_layout.get("page_bytes") == page_bytes
    ):
        raise ValueError("paged state mutation has no executable effect")

    source_hash = canonical_hash(baseline)
    prior_candidate = baseline.extensions.root.get("sloforge.dev/synthesis-candidate")
    root_parent_hash = (
        prior_candidate.get("parent_genome_hash") if isinstance(prior_candidate, dict) else None
    )
    if not isinstance(root_parent_hash, str):
        root_parent_hash = source_hash
    layout_extension: dict[str, JsonValue] = {
        "candidate_id": design.candidate_id,
        "layout": StateLayout.PAGED.value,
        "page_bytes": page_bytes,
        "policy": "bounded_request_pages",
    }
    states = tuple(
        state.model_copy(
            update={
                "layout": StateLayout.PAGED,
                "node": state.node.model_copy(
                    update={
                        "extensions": Extensions(
                            root={
                                **state.node.extensions.root,
                                "sloforge.dev/synthesized-state-layout": layout_extension,
                            }
                        )
                    }
                ),
            }
        )
        for state in baseline.state.states
    )
    state_genome = baseline.state.model_copy(
        update={
            "states": states,
            "node": baseline.state.node.model_copy(
                update={
                    "extensions": Extensions(
                        root={
                            **baseline.state.node.extensions.root,
                            "sloforge.dev/synthesized-state-layout": layout_extension,
                        }
                    )
                }
            ),
        }
    )
    target = baseline.model_copy(
        update={
            "genome_id": f"{baseline.genome_id}-{design.candidate_id[-12:]}",
            "state": state_genome,
            "extensions": Extensions(
                root={
                    **baseline.extensions.root,
                    "sloforge.dev/synthesis-candidate": {
                        "candidate_id": design.candidate_id,
                        "parent_genome_hash": root_parent_hash,
                        "immediate_parent_genome_hash": source_hash,
                        "transformation_ids": list(applied_transformation_ids),
                    },
                }
            ),
        }
    )
    target_hash = canonical_hash(target)
    obligations = (
        ProofObligation(
            obligation_id=f"{mutation.transformation_id}.ownership",
            property="paged allocation preserves one owner and releases all request pages",
            minimum_level=VerificationLevel.PROPERTY,
            scope="bounded generated runtime queue and declared per-request state domain",
            assumptions=("reference state tensors remain owned by their request",),
        ),
        ProofObligation(
            obligation_id=f"{mutation.transformation_id}.resource-bound",
            property="page rounding and concurrent allocation remain within the declared bound",
            minimum_level=VerificationLevel.PROPERTY,
            scope=f"page_bytes={page_bytes} and generated runtime queue capacity",
            assumptions=("state field maximum-byte contracts are conservative",),
        ),
    )
    transformation = Transformation(
        transformation_id=mutation.transformation_id,
        family=TransformationFamily.STATE_LAYOUT,
        source_pattern=GenomePattern(
            region="state",
            node_ids=tuple(node.stable_id for node in nodes),
            structural_constraints=(f"source_genome_sha256 == {source_hash}",),
        ),
        target_pattern=GenomePattern(
            region="state",
            node_ids=tuple(
                node.stable_id
                for node in (target.state.node, *(state.node for state in target.state.states))
            ),
            structural_constraints=(f"target_genome_sha256 == {target_hash}",),
        ),
        semantic_category=SemanticCategory.RESOURCE,
        designation=TransformationDesignation.RUNTIME_IMPLEMENTATION,
        preconditions=(
            "state nodes are mutable",
            "genesis.rules/baseline-exact-v1 is legal on every state node",
            "per-request state byte bounds are declared",
            f"page size is the power-of-two constant {page_bytes}",
        ),
        postconditions=(
            "every persistent state field uses paged layout",
            "the generated runtime reserves whole pages from a bounded allocator",
            "request completion, cancellation, timeout, and failure release all reserved pages",
        ),
        expected_quality_cost=(),
        expected_resource_change=(
            EstimatedChange(
                metric="maximum_internal_fragmentation_per_request",
                lower=0.0,
                expected=float(page_bytes // 2),
                upper=float(page_bytes - 1),
                unit="bytes",
            ),
        ),
        expected_performance_change=(
            EstimatedChange(
                metric="modeled_state_allocator_objective_ratio",
                lower=-mutation.expected_upside,
                expected=-mutation.expected_upside,
                upper=0.0,
                unit="ratio",
            ),
        ),
        affected_regions=("state",),
        verification_obligations=obligations,
        required_verifier_stages=(
            "state_ownership_trace",
            "conservative_resource_analysis",
            "candidate_runtime_replay",
        ),
        required_benchmark_stages=("deterministic_cpu_runtime",),
        rollback_strategy=f"drain requests and restore genome {source_hash}",
        proposal_source=design.proposal_engine,
        parent_transformations=applied_transformation_ids[-2:-1],
        extensions=Extensions(
            root={
                "sloforge.dev/applied-delta": {
                    "source_genome_hash": source_hash,
                    "target_genome_hash": target_hash,
                    "parameters": [
                        {"key": key, "value": parameters[key]} for key in sorted(parameters)
                    ],
                }
            }
        ),
    )
    return target, transformation


def lower_candidate(
    baseline: InferenceGenome,
    design: CandidateDesign,
    *,
    learned_constraints: tuple[LearnedConstraint, ...] = (),
    counterexample_references: tuple[str, ...] = (),
) -> LoweredCandidate:
    """Apply every registered mutation and recompute the actual target hash."""

    families = [mutation.family for mutation in design.mutations]
    if len(families) != len(set(families)):
        raise ValueError(
            "trusted lowering permits at most one executable mutation per transformation family"
        )
    current = baseline
    transformations: list[Transformation] = []
    for mutation in design.mutations:
        if mutation.family is TransformationFamily.BATCHING:
            lowering = _batching_delta
        elif mutation.family is TransformationFamily.STATE_LAYOUT:
            lowering = _state_layout_delta
        else:
            raise ValueError(f"no trusted lowering registered for {mutation.family.value}")
        applied_ids = (
            *tuple(item.transformation_id for item in transformations),
            mutation.transformation_id,
        )
        current, transformation = lowering(
            current,
            design,
            mutation,
            applied_transformation_ids=applied_ids,
        )
        if learned_constraints or counterexample_references:
            transformation = transformation.model_copy(
                update={
                    "learned_constraints": learned_constraints,
                    "counterexample_references": counterexample_references,
                }
            )
        transformations.append(transformation)
    target_hash = ArtifactDigest(value=canonical_hash(current))
    applied_design = design.model_copy(update={"genome_hash": target_hash})
    return LoweredCandidate(applied_design, current, tuple(transformations))


__all__ = ["LoweredCandidate", "lower_candidate"]
