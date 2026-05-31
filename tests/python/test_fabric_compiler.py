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
