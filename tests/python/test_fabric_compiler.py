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
from sloforge.fabric.compiler.core import (
    CandidateSummary,
    _degrees,
    _ordered_candidate_gpus,
    _pareto,
    _profile_duration,
    _service_availability,
)
from sloforge.fabric.ir import (
    DocumentReference,
    FabricMeasurementSeries,
    RankBinding,
    RankPlacement,
    WorkerRole,
    canonical_hash,
    load_fabric_profile,
    load_model_graph,
    load_topology_graph,
)
from sloforge.fabric.topology import build_canonical_fixture
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
        metrics.p95_ttft_ms.estimate + metrics.p99_tpot_ms.estimate * output_tokens_p95
    )
    for variant in result.selected.recovery_variants:
        degraded = variant.expected_degraded_metrics
        assert degraded.p95_end_to_end_ms.estimate == pytest.approx(
            degraded.p95_ttft_ms.estimate + degraded.p99_tpot_ms.estimate * output_tokens_p95
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
            experts_by_replica.setdefault(binding.replica_id, set()).add(assignment.expert_id)
            roles_by_replica.setdefault(binding.replica_id, set()).add(binding.worker_role.value)
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


def test_exhaustive_degree_space_includes_non_power_of_two_choices() -> None:
    assert _degrees(4, OptimizationStrategy.EXHAUSTIVE) == (1, 2, 3, 4)
    assert _degrees(8, OptimizationStrategy.HIERARCHICAL, fixed=3) == (3,)


def test_profile_curve_collapses_duplicate_sizes_before_extrapolation() -> None:
    profile = load_fabric_profile(FIXTURES / "fabric-profile-v1.json")
    base = next(item for item in profile.measurements if item.primitive == "collective")

    def point(identifier: str, size: int, duration: float) -> FabricMeasurementSeries:
        return base.model_copy(
            update={
                "measurement_id": identifier,
                "message_bytes": size,
                "summary_median_us": duration,
                "confidence_low_us": duration - 1.0,
                "confidence_high_us": duration + 1.0,
            }
        )

    profile = profile.model_copy(
        update={
            "measurements": (
                point("small", 100, 10.0),
                point("large-a", 200, 20.0),
                point("large-b", 200, 22.0),
            )
        }
    )
    duration, uncertainty = _profile_duration(
        profile, "collective", base.transport, base.rank_count, 300
    )
    assert duration == pytest.approx(32.0)
    assert uncertainty == pytest.approx(1.0)


def test_required_collective_transport_cannot_silently_disappear() -> None:
    request = _request(OptimizationStrategy.HIERARCHICAL)
    constrained = request.model_copy(
        update={
            "constraints": request.constraints.model_copy(
                update={
                    "tensor_parallel_degree": 2,
                    "pipeline_parallel_degree": 1,
                    "data_parallel_degree": 1,
                    "expert_parallel_degree": 1,
                    "permitted_transports": ("tcp",),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="collective_transport_not_permitted"):
        compile_physical_plan(constrained)


def test_hard_latency_slo_uses_prediction_interval_upper_bound() -> None:
    request = _request(OptimizationStrategy.HIERARCHICAL)
    constrained = request.model_copy(
        update={
            "constraints": request.constraints.model_copy(
                update={
                    "tensor_parallel_degree": 1,
                    "pipeline_parallel_degree": 1,
                    "data_parallel_degree": 1,
                    "expert_parallel_degree": 1,
                    # The fixed candidate estimate is about 64 ms, but its 10%
                    # uncertainty bound exceeds this hard constraint.
                    "p95_ttft_ms": 68.0,
                }
            )
        }
    )
    with pytest.raises(ValueError, match="p95_ttft_slo"):
        compile_physical_plan(constrained)


def test_pareto_frontier_preserves_goodput_tradeoff() -> None:
    def candidate(identifier: str, *, latency: float, goodput: float) -> CandidateSummary:
        return CandidateSummary(
            candidate_id=identifier,
            tensor_parallel=1,
            pipeline_parallel=1,
            data_parallel=1,
            expert_parallel=1,
            disaggregated=False,
            gpu_ids=("gpu-0",),
            communication_us=10.0,
            p95_ttft_ms=latency,
            p99_tpot_ms=latency,
            goodput_tokens_per_second=goodput,
            cost_per_million_tokens=1.0,
            availability=0.99,
            failure_exposure_score=0.01,
            objective_score=latency,
            feasible=True,
            rejection_codes=(),
        )

    frontier = _pareto(
        (
            candidate("lower-latency", latency=10.0, goodput=100.0),
            candidate("higher-goodput", latency=11.0, goodput=200.0),
        )
    )
    assert {item.candidate_id for item in frontier} == {
        "lower-latency",
        "higher-goodput",
    }


def test_robust_placement_spreads_complete_replicas_across_hosts() -> None:
    topology = build_canonical_fixture("two_node_infiniband")
    placement = _ordered_candidate_gpus(
        topology,
        tp=2,
        pp=1,
        dp=2,
        strategy=OptimizationStrategy.ROBUST_FAILURE,
        seed=7,
    )
    assert len(placement) == 4
    first_replica_hosts = {gpu.host_id for gpu in placement[:2]}
    second_replica_hosts = {gpu.host_id for gpu in placement[2:]}
    assert len(first_replica_hosts) == len(second_replica_hosts) == 1
    assert first_replica_hosts != second_replica_hosts


def test_availability_preserves_shared_fault_domain_correlation() -> None:
    def binding(rank: int, replica: str, role: WorkerRole, domain: str) -> RankBinding:
        return RankBinding(
            rank_id=rank,
            host_id=domain,
            gpu_id=f"gpu-{rank}",
            numa_domain_id=f"numa-{rank}",
            process_cpu_affinity=str(rank),
            worker_role=role,
            replica_id=replica,
            fault_domain=domain,
        )

    shared_aggregated = RankPlacement(
        bindings=(
            binding(0, "replica-0", WorkerRole.AGGREGATED, "host-a"),
            binding(1, "replica-1", WorkerRole.AGGREGATED, "host-a"),
        )
    )
    independent_aggregated = RankPlacement(
        bindings=(
            binding(0, "replica-0", WorkerRole.AGGREGATED, "host-a"),
            binding(1, "replica-1", WorkerRole.AGGREGATED, "host-b"),
        )
    )
    shared_disaggregated = RankPlacement(
        bindings=(
            binding(0, "prefill", WorkerRole.PREFILL, "host-a"),
            binding(1, "decode", WorkerRole.DECODE, "host-a"),
        )
    )
    independent_disaggregated = RankPlacement(
        bindings=(
            binding(0, "prefill", WorkerRole.PREFILL, "host-a"),
            binding(1, "decode", WorkerRole.DECODE, "host-b"),
        )
    )
    assert _service_availability(shared_aggregated, 0.9) == pytest.approx(0.9)
    assert _service_availability(independent_aggregated, 0.9) == pytest.approx(0.99)
    assert _service_availability(shared_disaggregated, 0.9) == pytest.approx(0.9)
    assert _service_availability(independent_disaggregated, 0.9) == pytest.approx(0.81)
