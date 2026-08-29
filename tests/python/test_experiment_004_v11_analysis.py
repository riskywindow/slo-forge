from __future__ import annotations

import importlib.util
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY_ROOT / "benchmarks/branchfabric/build_experiment_004_v11_analysis.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_experiment_004_v11_analysis", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v10_pass_dag_is_exact_deterministic_and_timing_safe() -> None:
    module = _module()
    first = module.build(REPOSITORY_ROOT)
    second = module.build(REPOSITORY_ROOT)
    assert first == second
    assert first["provenance_verification"]["verified"]
    assert first["aggregate_full_state_pass_count"] == 30
    assert first["raw_segment_record_count"] == 270
    assert first["execution_pass_count"] == 44
    assert len(first["edges"]) == 46
    assert sum(item["physical_touch_bytes"] for item in first["passes"]) == 61_303_947_264
    assert first["restore_subset_derivation"]["sum_logical_bytes"] == 1_056_964_608

    nodes = {item["pass_id"]: item for item in first["execution_passes"]}
    assert nodes["capture-source-native-read"]["dependency_edges"]["successors"] == [
        "capture-native-axis-contiguous"
    ]
    assert nodes["capture-pinned-transport-lifetime"]["critical_path_wall_time_ns"] == 0
    zero_nodes = [
        item
        for item in first["execution_passes"]
        if item["aggregate_pass_id"] == "restore-zero-native-pages"
    ]
    assert all(item["cuda_time_ns"] for item in zero_nodes)

    timing_groups: dict[str, list[dict[str, object]]] = {}
    for node in first["execution_passes"]:
        timing_groups.setdefault(node["timing_group_id"], []).append(node)
    for group in timing_groups.values():
        owners = [item for item in group if item["timing_role"] == "OWNER"]
        assert len(owners) <= 1
        assert sum(int(item["critical_path_wall_time_ns"]) for item in group) == (
            int(owners[0]["wall_time_ns"]) if owners else 0
        )

    indegree = {node_id: 0 for node_id in nodes}
    outgoing = {node_id: [] for node_id in nodes}
    for edge in first["edges"]:
        indegree[edge["destination"]] += 1
        outgoing[edge["source"]].append(edge["destination"])
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        source = ready.pop()
        visited += 1
        for destination in outgoing[source]:
            indegree[destination] -= 1
            if indegree[destination] == 0:
                ready.append(destination)
    assert visited == len(nodes)
    assert "removable_pass_ranking" not in first
    assert all("replacement_contract" in item for item in first["optimization_candidate_ranking"])
