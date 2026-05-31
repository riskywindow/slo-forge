from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sloforge.fabric.ir import (
    CollectiveOperation,
    ConnectionType,
    CurvePoint,
    DiscoverySource,
    FactProvenance,
    GpuNode,
    HealthState,
    MigState,
    PhysicalExecutionPlan,
    RankBinding,
    RankPlacement,
    TopologyEdge,
    TopologyGraph,
    WorkerRole,
    canonical_hash,
)
from sloforge.fabric.performance import (
    CalibrationError,
    CalibrationMode,
    CollectiveTraceEvidence,
    IntegrationStatus,
    RankOrderingExperiment,
    RankOrderingExperimentConfig,
    RankOrderingExperimentInput,
    execute_rank_ordering_experiment,
    optimize_rank_order,
    write_experiment_artifacts,
)
from sloforge.ir import ArtifactDigest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "fabric"
RANK_ORDERING_BENCHMARK = Path(__file__).parents[2] / "benchmarks" / "fabric" / "rank_ordering"
WHEN = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _provenance() -> tuple[FactProvenance, ...]:
    return (
        FactProvenance(
            source=DiscoverySource.SYNTHETIC,
            observed_at=WHEN,
            confidence=1.0,
            source_uri="benchmarks/fabric/rank_ordering/synthetic-calibration.json",
            field="topology",
        ),
    )


def _curve(message_bytes: int, median: float) -> tuple[CurvePoint, ...]:
    return (
        CurvePoint(
            message_bytes=message_bytes,
            median=median,
            p95=median * 1.05,
            robust_dispersion=median * 0.01,
            confidence_low=median * 0.98,
            confidence_high=median * 1.02,
            sample_count=40,
        ),
    )


def _topology(rank_count: int = 4, *, calibrated: bool = True) -> TopologyGraph:
    nodes = tuple(
        GpuNode(
            node_id=f"gpu-{rank}",
            host_id="host-0" if rank < rank_count // 2 else "host-1",
            gpu_index=rank,
            uuid=f"GPU-{rank}",
            product="Synthetic H100",
            architecture="Hopper",
            memory_bytes=80 * 1024**3,
            compute_capability="9.0",
            mig_state=MigState.DISABLED,
            numa_domain_id=f"numa-{rank // 2}",
            pci_address=f"0000:{rank + 1:02x}:00.0",
            health=HealthState.HEALTHY,
            provenance=_provenance(),
        )
        for rank in range(rank_count)
    )
    # For four ranks, the fast physical ring is 0-2-3-1-0 while the reference
    # 0-1-2-3-0 crosses two oversubscribed links. Larger cases repeat the same
    # deterministic asymmetric pattern to exercise bounded search.
    fast_pairs = {frozenset((rank, (rank + 2) % rank_count)) for rank in range(rank_count)}
    if rank_count == 4:
        fast_pairs = {
            frozenset((0, 2)),
            frozenset((2, 3)),
            frozenset((3, 1)),
            frozenset((1, 0)),
        }
    edges: list[TopologyEdge] = []
    for left in range(rank_count):
        for right in range(left + 1, rank_count):
            fast = frozenset((left, right)) in fast_pairs
            edges.append(
                TopologyEdge(
                    edge_id=f"link-{left}-{right}",
                    source_node_id=f"gpu-{left}",
                    target_node_id=f"gpu-{right}",
                    connection=ConnectionType.NVLINK if fast else ConnectionType.PCIE,
                    directionality="bidirectional",
                    duplex="full",
                    theoretical_bandwidth_gbps=100.0 if fast else 20.0,
                    bandwidth_curve_gbps=_curve(256 * 1024, 100.0 if fast else 20.0)
                    if calibrated
                    else (),
                    latency_curve_us=_curve(256 * 1024, 1.0 if fast else 5.0) if calibrated else (),
                    sharing_group=f"fabric-{left}-{right}",
                    contention_domain=f"fabric-{left}-{right}",
                    health=HealthState.HEALTHY,
                    measurement_confidence=0.98 if calibrated else None,
                    measured_at=WHEN if calibrated else None,
                    measurement_environment_digest=ArtifactDigest(value="a" * 64)
                    if calibrated
                    else None,
                    discovery_provenance=_provenance(),
                )
            )
    return TopologyGraph(
        topology_id=f"asymmetric-{rank_count}",
        discovered_at=WHEN,
        nodes=nodes,
        edges=tuple(edges),
        container_limited=False,
    )


def _placement(rank_count: int) -> RankPlacement:
    return RankPlacement(
        bindings=tuple(
            RankBinding(
                rank_id=rank,
                host_id="host-0" if rank < rank_count // 2 else "host-1",
                gpu_id=f"gpu-{rank}",
                numa_domain_id=f"numa-{rank // 2}",
                nic_id=None,
                network_rail_id=None,
                process_cpu_affinity=f"{rank * 2}-{rank * 2 + 1}",
                worker_role=WorkerRole.AGGREGATED,
                replica_id="replica-0",
                fault_domain=f"host-{rank // max(1, rank_count // 2)}",
            )
            for rank in range(rank_count)
        )
    )


def _operation(rank_count: int) -> CollectiveOperation:
    ranks = tuple(range(rank_count))
    return CollectiveOperation(
        operation_id="tp-all-reduce",
        operation="all_reduce",
        participating_ranks=ranks,
        message_size_intercept_bytes=0,
        message_size_bytes_per_token=2048.0,
        algorithm="ring",
        transport="nvlink",
        channel_count=4,
        rail_ids=(),
        rank_order=ranks,
        expected_duration_us=100.0,
        uncertainty_us=5.0,
        overlap_window_id=None,
        depends_on=(),
        fallback="host_staged",
    )


def _plan(topology: TopologyGraph) -> PhysicalExecutionPlan:
    source = json.loads((FIXTURES / "physical-execution-plan-v1.json").read_text())
    ranks = list(range(4))
    source["plan_id"] = "rank-ordering-fixture"
    source["topology_fingerprint"]["value"] = canonical_hash(topology)
    source["parallelism"].update(
        {
            "tensor_parallel_degree": 4,
            "expert_parallel_degree": 1,
            "groups": [{"group_id": "tp-0", "kind": "tensor", "rank_ids": ranks}],
            "replica_groups": [{"group_id": "replica-0", "kind": "data", "rank_ids": ranks}],
            "prefill_decode_disaggregated": False,
        }
    )
    binding = source["rank_placement"]["bindings"][0]
    source["rank_placement"]["bindings"] = [
        {
            **binding,
            "rank_id": rank,
            "gpu_id": f"gpu-{rank}",
            "process_cpu_affinity": f"{rank * 2}-{rank * 2 + 1}",
            "worker_role": "aggregated",
        }
        for rank in ranks
    ]
    allocation = source["memory"]["allocations"][0]
    source["memory"]["allocations"] = [{**allocation, "rank_id": rank} for rank in ranks]
    source["collectives"]["operations"] = [_operation(4).model_dump(mode="json")]
    source["expert_placement"] = None
    source["kv_transfer"] = None
    source["communication_overlap"]["windows"] = []
    source["recovery_variants"] = []
    return PhysicalExecutionPlan.model_validate_json(json.dumps(source))


def _evidence() -> CollectiveTraceEvidence:
    return CollectiveTraceEvidence(
        artifact_uri="benchmarks/fabric/rank_ordering/observed-collective-trace.json",
        artifact_sha256="951c3362a0ea563531a9e66ffca6267bcd5579c9ad1b421a8ff7ca5481910351",
        plan_id="rank-ordering-fixture",
        operation_id="tp-all-reduce",
        calibration_mode=CalibrationMode.SYNTHETIC_CALIBRATED,
        observed_duration_microseconds=(120.0, 124.0, 122.0, 121.0, 125.0),
        collective_critical_path_fraction=0.42,
        rank_wait=(),
        fault_free=True,
    )


def _config() -> RankOrderingExperimentConfig:
    return RankOrderingExperimentConfig(
        seed=17,
        message_sizes_bytes=(64 * 1024, 1024 * 1024),
        optimization_message_bytes=1024 * 1024,
        warmup_trials=3,
        measured_trials=25,
        bootstrap_rounds=400,
        link_variation_fraction=0.02,
        minimum_improvement_percent=1.0,
    )


def test_exact_order_uses_valid_routes_and_beats_sequential_reference() -> None:
    topology = _topology()
    result = optimize_rank_order(
        topology,
        _placement(4),
        _operation(4),
        message_bytes=1024 * 1024,
    )
    assert result.method.value == "exact"
    assert set(result.optimized.rank_order) == {0, 1, 2, 3}
    assert result.optimized.predicted_duration_microseconds < (
        result.reference.predicted_duration_microseconds
    )
    assert result.predicted_improvement_percent > 1.0
    for route in result.optimized.routes:
        assert route.traversals
        assert route.traversals[0].edge_id.startswith("link-")


def test_large_group_uses_bounded_deterministic_heuristic() -> None:
    topology = _topology(10)
    first = optimize_rank_order(
        topology,
        _placement(10),
        _operation(10),
        message_bytes=256 * 1024,
        exact_rank_limit=5,
        heuristic_pass_limit=3,
    )
    second = optimize_rank_order(
        topology,
        _placement(10),
        _operation(10),
        message_bytes=256 * 1024,
        exact_rank_limit=5,
        heuristic_pass_limit=3,
    )
    assert first == second
    assert first.method.value == "greedy_two_opt"
    assert len(first.optimized.rank_order) == 10
    assert first.candidates_evaluated < 500


def test_missing_measured_curve_fails_without_theoretical_fallback() -> None:
    with pytest.raises(CalibrationError, match="no fully measured route"):
        optimize_rank_order(
            _topology(calibrated=False),
            _placement(4),
            _operation(4),
            message_bytes=1024 * 1024,
        )


def test_experiment_is_paired_deterministic_and_synthetic_never_enables(
    tmp_path: Path,
) -> None:
    topology = _topology()
    plan = _plan(topology)
    first = execute_rank_ordering_experiment(plan, topology, _evidence(), _config())
    second = execute_rank_ordering_experiment(plan, topology, _evidence(), _config())
    assert first == second
    assert first.artifact_hash == second.artifact_hash
    assert first.decision.status is IntegrationStatus.MEASURE_ON_HARDWARE
    assert not first.decision.enabled_by_default
    assert len(first.trials) == 2 * (3 + 25)
    assert all(summary.improvement_ci_low_percent > 1.0 for summary in first.summaries)

    paths = write_experiment_artifacts(tmp_path, first)
    result = json.loads(Path(paths.result_json).read_text())
    raw_lines = Path(paths.raw_samples_jsonl).read_text().splitlines()
    report = Path(paths.report_markdown).read_text()
    assert result["artifact_hash"] == first.artifact_hash
    assert len(raw_lines) == len(first.trials)
    assert "deterministic synthetic-calibration result" in report
    assert "measure_on_hardware" in report


def test_unjustified_trace_keeps_reference() -> None:
    topology = _topology()
    evidence = _evidence().model_copy(update={"collective_critical_path_fraction": 0.01})
    result = execute_rank_ordering_experiment(_plan(topology), topology, evidence, _config())
    assert result.decision.status is IntegrationStatus.KEEP_REFERENCE
    assert not result.decision.trace_justified


def test_invalid_order_and_mislabeled_synthetic_evidence_are_rejected() -> None:
    topology = _topology()
    invalid = _operation(4).model_copy(update={"rank_order": (0, 1, 1, 3)})
    with pytest.raises(ValueError, match="permutation"):
        optimize_rank_order(
            topology,
            _placement(4),
            invalid,
            message_bytes=1024 * 1024,
        )

    mislabeled = _evidence().model_copy(
        update={"calibration_mode": CalibrationMode.MEASURED_HARDWARE}
    )
    with pytest.raises(ValueError, match="cannot be labeled as measured hardware"):
        execute_rank_ordering_experiment(_plan(topology), topology, mislabeled, _config())


def test_checked_benchmark_artifacts_are_reproducible_and_hash_protected() -> None:
    bundle = RankOrderingExperimentInput.model_validate_json(
        (RANK_ORDERING_BENCHMARK / "synthetic-input.json").read_text()
    )
    reproduced = execute_rank_ordering_experiment(
        bundle.physical_plan,
        bundle.topology,
        bundle.trace_evidence,
        bundle.config,
    )
    checked = RankOrderingExperiment.model_validate_json(
        (RANK_ORDERING_BENCHMARK / "results" / "rank-ordering-experiment.json").read_text()
    )
    assert reproduced == checked
    assert (
        reproduced.artifact_hash
        in (Path(__file__).parents[2] / "reports" / "rank-ordering-experiment.md").read_text()
    )

    tampered = checked.model_dump(mode="json")
    tampered["decision"]["rationale"] = "tampered"
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        RankOrderingExperiment.model_validate_json(json.dumps(tampered))
