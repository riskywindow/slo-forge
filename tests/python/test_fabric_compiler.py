from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from sloforge.fabric.compiler import (
    CompilerAssumptions,
    CompilerConstraints,
    CompilerObjective,
    CompilerRequest,
    OptimizationStrategy,
    compile_physical_plan,
)
from sloforge.fabric.ir import (
    DocumentReference,
    canonical_hash,
    load_fabric_profile,
    load_model_graph,
    load_topology_graph,
)
from sloforge.ir import ArtifactDigest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "fabric"


def _digest(value: str) -> ArtifactDigest:
    return ArtifactDigest(value=value * 64)


def _request(strategy: OptimizationStrategy) -> CompilerRequest:
    topology = load_topology_graph(FIXTURES / "topology-graph-v1.json")
    profile = load_fabric_profile(FIXTURES / "fabric-profile-v1.json").model_copy(
        update={"topology_fingerprint": ArtifactDigest(value=canonical_hash(topology))}
    )
    return CompilerRequest(
        logical_deployment_plan=DocumentReference(
            kind="DeploymentPlan",
            api_version="sloforge.io/v1",
            uri="artifacts/plans/logical.json",
            digest=_digest("a"),
            uid="logical-fixture",
            generation=1,
        ),
        model=load_model_graph(FIXTURES / "model-graph-v1.json"),
        topology=topology,
        fabric_profile=profile,
        constraints=CompilerConstraints(
            prompt_tokens_p95=512,
            output_tokens_p95=64,
            maximum_concurrent_requests=8,
            p95_ttft_ms=1_000.0,
            p99_tpot_ms=100.0,
            maximum_ranks=2,
        ),
        assumptions=CompilerAssumptions(
            prefill_tokens_per_second_per_gpu=8_000.0,
            decode_tokens_per_second_per_gpu=120.0,
            gpu_hourly_price_usd=2.0,
            base_availability=0.999,
            cold_start_ms=2_000.0,
            measurement_relative_uncertainty=0.10,
        ),
        objective=CompilerObjective.ROBUST_BALANCED,
        strategy=strategy,
        generated_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
        seed=7,
        git_commit="fixture",
        environment_digest=_digest("b"),
    )


def test_hierarchical_compiler_produces_valid_explainable_plan() -> None:
    result = compile_physical_plan(_request(OptimizationStrategy.HIERARCHICAL))
    assert result.selected.rank_placement.bindings
    assert result.selected.memory.allocations
    assert result.selected.optimizer_history
    assert result.selected.rejected_alternatives
    assert result.pareto_frontier
    assert result.simulator_calls == 0
    assert all(entry.simulator_calls == 0 for entry in result.selected.optimizer_history)
    assert len(result.selected.rank_placement.bindings) == (
        result.selected.parallelism.expected_rank_count
    )
    assert result.selected.predicted_metrics.p95_ttft_ms.upper >= (
        result.selected.predicted_metrics.p95_ttft_ms.estimate
    )


def test_physical_end_to_end_metrics_use_requested_output_length() -> None:
    request = _request(OptimizationStrategy.HIERARCHICAL)
    output_tokens_p95 = 96
    result = compile_physical_plan(
        request.model_copy(
            update={
                "constraints": request.constraints.model_copy(
                    update={"output_tokens_p95": output_tokens_p95}
                )
            }
        )
    )
    metrics = result.selected.predicted_metrics
    assert metrics.p95_end_to_end_ms.estimate == pytest.approx(
        metrics.p95_ttft_ms.estimate
        + metrics.p99_tpot_ms.estimate * output_tokens_p95
    )
    for variant in result.selected.recovery_variants:
        degraded = variant.expected_degraded_metrics
        assert degraded.p95_end_to_end_ms.estimate == pytest.approx(
            degraded.p95_ttft_ms.estimate
            + degraded.p99_tpot_ms.estimate * output_tokens_p95
        )


def test_compiler_is_deterministic() -> None:
    request = _request(OptimizationStrategy.EXHAUSTIVE)
    first = compile_physical_plan(request)
    second = compile_physical_plan(request)
    assert first == second
    assert canonical_hash(first.selected) == canonical_hash(second.selected)


def test_topology_unaware_baseline_is_available() -> None:
    result = compile_physical_plan(_request(OptimizationStrategy.TOPOLOGY_UNAWARE))
    assert result.strategy is OptimizationStrategy.TOPOLOGY_UNAWARE
    assert result.selected.rank_placement.bindings[0].gpu_id == "gpu-0"


def test_random_placement_baseline_is_seeded_and_public() -> None:
    first = compile_physical_plan(_request(OptimizationStrategy.RANDOM_PLACEMENT))
    second = compile_physical_plan(_request(OptimizationStrategy.RANDOM_PLACEMENT))
    assert first == second
    assert first.strategy is OptimizationStrategy.RANDOM_PLACEMENT
    assert first.selected.rank_placement.bindings


def test_fixed_parallelism_exposes_topology_comparison_space() -> None:
    request = _request(OptimizationStrategy.GREEDY_TOPOLOGY_AWARE)
    constrained = request.model_copy(
        update={
            "constraints": request.constraints.model_copy(
                update={
                    "tensor_parallel_degree": 2,
                    "pipeline_parallel_degree": 1,
                    "data_parallel_degree": 1,
                    "expert_parallel_degree": 1,
                }
            )
        }
    )
    result = compile_physical_plan(constrained)
    assert result.selected.parallelism.tensor_parallel_degree == 2
    assert result.selected.parallelism.pipeline_parallel_degree == 1
    assert len(result.selected.rank_placement.bindings) == 2


def test_fixed_parallelism_cannot_exceed_rank_limit() -> None:
    with pytest.raises(ValueError, match="fixed TP x PP x DP"):
        CompilerConstraints(
            prompt_tokens_p95=512,
            output_tokens_p95=64,
            maximum_concurrent_requests=8,
            p95_ttft_ms=1_000.0,
            p99_tpot_ms=100.0,
            maximum_ranks=2,
            tensor_parallel_degree=2,
            pipeline_parallel_degree=2,
        )


def test_stale_fabric_profile_is_rejected() -> None:
    request = _request(OptimizationStrategy.HIERARCHICAL)
    stale = request.fabric_profile.model_copy(update={"topology_fingerprint": _digest("c")})
    try:
        request.model_copy(update={"fabric_profile": stale}).model_dump()
        CompilerRequest.model_validate(
            {**request.model_dump(mode="python"), "fabric_profile": stale.model_dump(mode="python")}
        )
    except ValueError as error:
        assert "does not match topology" in str(error)
    else:
        raise AssertionError("a stale fabric profile was accepted")


def test_disaggregation_assigns_complete_data_replicas_to_worker_pools() -> None:
    request = _request(OptimizationStrategy.HIERARCHICAL)
    constrained = request.model_copy(
        update={
            "constraints": request.constraints.model_copy(
                update={
                    "tensor_parallel_degree": 1,
                    "pipeline_parallel_degree": 1,
                    "data_parallel_degree": 2,
                    "expert_parallel_degree": 1,
                    "require_disaggregation": True,
                }
            )
        }
    )
    result = compile_physical_plan(constrained)
    bindings_by_replica: dict[str, set[str]] = {}
    for binding in result.selected.rank_placement.bindings:
        bindings_by_replica.setdefault(binding.replica_id, set()).add(binding.worker_role.value)
    assert bindings_by_replica == {"replica-0": {"prefill"}, "replica-1": {"decode"}}
    groups = {group.kind.value: set(group.rank_ids) for group in result.selected.parallelism.groups}
    assert groups["prefill"] == {0}
    assert groups["decode"] == {1}
    search = compile_physical_plan(
        request.model_copy(
            update={
                "constraints": request.constraints.model_copy(
                    update={"require_disaggregation": True}
                )
            }
        )
    )
    assert any(
        "disaggregation_requires_two_data_replicas" in candidate.rejection_codes
        for candidate in search.all_candidates
    )


def test_disaggregated_replicas_each_contain_the_full_expert_set() -> None:
    request = _request(OptimizationStrategy.HIERARCHICAL)
    constrained = request.model_copy(
        update={
            "constraints": request.constraints.model_copy(
                update={
                    "tensor_parallel_degree": 1,
                    "pipeline_parallel_degree": 1,
                    "data_parallel_degree": 2,
                    "expert_parallel_degree": 1,
                    "require_disaggregation": True,
                }
            )
        }
    )
    result = compile_physical_plan(constrained)
    placement = result.selected.expert_placement
    assert placement is not None
    binding_by_rank = {
        binding.rank_id: binding for binding in result.selected.rank_placement.bindings
    }
    expected_experts = {
        expert.expert_id for layer in request.model.layers for expert in layer.experts
    }
    experts_by_replica: dict[str, set[str]] = {}
    roles_by_replica: dict[str, set[str]] = {}
    for assignment in placement.assignments:
        assert len(assignment.rank_ids) == 2
        for rank_id in assignment.rank_ids:
            binding = binding_by_rank[rank_id]
            experts_by_replica.setdefault(binding.replica_id, set()).add(
                assignment.expert_id
            )
            roles_by_replica.setdefault(binding.replica_id, set()).add(
                binding.worker_role.value
            )
    assert experts_by_replica == {
        "replica-0": expected_experts,
        "replica-1": expected_experts,
    }
    assert roles_by_replica == {"replica-0": {"prefill"}, "replica-1": {"decode"}}
    assert placement.maximum_replicas_per_expert == 2


def test_data_parallelism_duplicates_kv_while_pipeline_parallelism_shards_it() -> None:
    request = _request(OptimizationStrategy.HIERARCHICAL)

    def compile_fixed(*, pp: int, dp: int) -> int:
        result = compile_physical_plan(
            request.model_copy(
                update={
                    "constraints": request.constraints.model_copy(
                        update={
                            "tensor_parallel_degree": 1,
                            "pipeline_parallel_degree": pp,
                            "data_parallel_degree": dp,
                            "expert_parallel_degree": 1,
                        }
                    )
                }
            )
        )
        return result.selected.memory.allocations[0].kv_cache_bytes

    single_replica_kv = compile_fixed(pp=1, dp=1)
    assert compile_fixed(pp=1, dp=2) == single_replica_kv
    assert compile_fixed(pp=2, dp=1) == (single_replica_kv + 1) // 2
