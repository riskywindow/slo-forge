from __future__ import annotations

from pathlib import Path

from sloforge.fabric.ir import load_topology_graph
from sloforge.fabric.profiling import BenchmarkResult, BenchmarkStatus, MeasurementMode

ROOT = Path(__file__).parents[2]


def test_checked_in_current_host_discovery_and_measurement_are_explicit() -> None:
    artifact_root = ROOT / "artifacts" / "fabric" / "local"
    topology = load_topology_graph(artifact_root / "topology.json")
    payload = (artifact_root / "profile" / "host-memory.json").read_bytes()
    benchmark = BenchmarkResult.model_validate_json(payload)

    assert topology.nodes and topology.edges
    assert benchmark.mode is MeasurementMode.MEASURED
    assert benchmark.status is BenchmarkStatus.SUCCESS
    assert benchmark.case.primitive.value == "host_memcpy"
    assert benchmark.case.placement.gpu_ids == ()
    assert benchmark.case.sample_count == len(benchmark.raw_samples) == 9
    assert all(not sample.synthetic for sample in benchmark.raw_samples)
    assert benchmark.summary is not None
    assert len(benchmark.artifact_hash) == 64
