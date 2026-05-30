from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from sloforge.fabric.adapters import (
    DeploymentTarget,
    DynamoBackend,
    FabricAdapterContext,
    RuntimeKind,
    UnsupportedCapabilityError,
    export_physical_plan,
)
from sloforge.fabric.ir import (
    RankBinding,
    RankPlacement,
    WorkerRole,
    load_physical_execution_plan,
    load_topology_graph,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "fabric"


def _context(
    *,
    runtime: RuntimeKind = RuntimeKind.NATIVE,
    runtime_version: str = "0.1.0",
    dynamo_backend: DynamoBackend | None = None,
    allow_advisory_cloud_metadata: bool = False,
) -> FabricAdapterContext:
    return FabricAdapterContext(
        plan=load_physical_execution_plan(FIXTURES / "physical-execution-plan-v1.json"),
        topology=load_topology_graph(FIXTURES / "topology-graph-v1.json"),
        model_id="Qwen/Qwen3-0.6B",
        model_revision="main",
        image="ghcr.io/sloforge/runtime:0.1.0",
        runtime=runtime,
        runtime_version=runtime_version,
        dynamo_backend=dynamo_backend,
        allow_advisory_cloud_metadata=allow_advisory_cloud_metadata,
    )


def _aggregated_dynamo_context() -> FabricAdapterContext:
    base = _context(
        runtime=RuntimeKind.DYNAMO,
        runtime_version="1.0.2",
        dynamo_backend=DynamoBackend.VLLM,
    )
    bindings = tuple(
        RankBinding.model_validate(
            {
                **binding.model_dump(mode="python"),
                "worker_role": WorkerRole.AGGREGATED,
            }
        )
        for binding in base.plan.rank_placement.bindings
    )
    plan = base.plan.model_copy(
        update={
            "parallelism": base.plan.parallelism.model_copy(
                update={"prefill_decode_disaggregated": False}
            ),
            "rank_placement": RankPlacement(bindings=bindings),
            "kv_transfer": None,
        }
    )
    return base.model_copy(update={"plan": plan})


def test_local_adapter_preserves_rank_cpu_gpu_nic_and_rail_bindings(tmp_path: Path) -> None:
    result = export_physical_plan(
        context=_context(), target=DeploymentTarget.LOCAL, output=tmp_path / "local"
    )
    launch = json.loads((tmp_path / "local" / "launch-plan.json").read_text())
    bindings = launch["groups"][0]["rank_bindings"]
    assert [item["expected_gpu_uuid"] for item in bindings] == [
        "GPU-SYNTH-0",
        "GPU-SYNTH-1",
    ]
    assert [item["cpu_affinity"] for item in bindings] == ["0-3", "4-7"]
    assert {item["nic_interface"] for item in bindings} == {"mlx5_0"}
    assert {item["network_rail_id"] for item in bindings} == {"rail-0"}
    assert launch["binding_policy"] == "fail_on_mismatch"
    assert result.deployed is False
    assert all((tmp_path / "local" / item.path).is_file() for item in result.artifacts)


def test_docker_adapter_emits_hardened_bounded_rank_group(tmp_path: Path) -> None:
    export_physical_plan(
        context=_context(), target=DeploymentTarget.DOCKER, output=tmp_path / "docker"
    )
    compose = yaml.safe_load((tmp_path / "docker" / "compose.yaml").read_text())
    service = next(iter(compose["services"].values()))
    assert service["cpuset"] == "0-3,4-7"
    assert service["pids_limit"] == 1024
    assert service["read_only"] is True
    devices = service["deploy"]["resources"]["reservations"]["devices"]
    assert devices[0]["device_ids"] == ["GPU-SYNTH-0", "GPU-SYNTH-1"]
    assert "./topology.json:/etc/sloforge/topology.json:ro" in service["volumes"]


def test_kubernetes_adapter_emits_physical_scheduling_and_safe_rollout(tmp_path: Path) -> None:
    result = export_physical_plan(
        context=_context(),
        target=DeploymentTarget.KUBERNETES,
        output=tmp_path / "kubernetes",
    )
    manifest = yaml.safe_load((tmp_path / "kubernetes" / "physical-plan.yaml").read_text())
    deployment = next(item for item in manifest["items"] if item["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    expression = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"
    ][0]["matchExpressions"][0]
    assert expression == {
        "key": "kubernetes.io/hostname",
        "operator": "In",
        "values": ["host-0"],
    }
    assert pod["topologySpreadConstraints"]
    assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert deployment["spec"]["strategy"]["rollingUpdate"] == {
        "maxUnavailable": 0,
        "maxSurge": 1,
    }
    assert result.capabilities.exact_gpu_uuid_placement is False
    assert result.capabilities.advisory_only


def test_dynamo_adapter_uses_current_v1beta1_component_shape(tmp_path: Path) -> None:
    export_physical_plan(
        context=_aggregated_dynamo_context(),
        target=DeploymentTarget.DYNAMO,
        output=tmp_path / "dynamo",
    )
    graph = yaml.safe_load((tmp_path / "dynamo" / "dynamo-graph-deployment.yaml").read_text())
    schema = json.loads(
        (Path(__file__).parents[2] / "deploy/dynamo/dgd-v1beta1-subset.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(graph)
    assert graph["apiVersion"] == "nvidia.com/v1beta1"
    assert graph["kind"] == "DynamoGraphDeployment"
    worker = next(
        component for component in graph["spec"]["components"] if component["type"] == "worker"
    )
    main = worker["podTemplate"]["spec"]["containers"][0]
    assert main["command"] == ["python3", "-m", "dynamo.vllm"]
    assert "--tensor-parallel-size" in main["args"]
    assert "--enable-expert-parallel" in main["args"]
    assert main["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert (
        worker["podTemplate"]["metadata"]["annotations"]["sloforge.dev/binding-policy"]
        == "fail-on-mismatch"
    )


@pytest.mark.parametrize("target", [DeploymentTarget.MODAL, DeploymentTarget.TRUSS])
def test_cloud_adapters_fail_closed_then_emit_only_supported_metadata(
    target: DeploymentTarget, tmp_path: Path
) -> None:
    with pytest.raises(UnsupportedCapabilityError, match="does not expose exact rank"):
        export_physical_plan(context=_context(), target=target, output=tmp_path / "rejected")
    result = export_physical_plan(
        context=_context(allow_advisory_cloud_metadata=True),
        target=target,
        output=tmp_path / target.value,
    )
    metadata = json.loads((tmp_path / target.value / "physical-plan-metadata.json").read_text())
    assert metadata["enforcement"] == "advisory"
    assert metadata["deployment_mutation_performed"] is False
    assert "rank_to_gpu_uuid" in metadata["unsupported_enforcement"]
    assert result.deployed is False
    if target is DeploymentTarget.MODAL:
        overlay = json.loads((tmp_path / target.value / "modal-image-env.json").read_text())
        assert overlay["api"] == "modal.Image.env"
    else:
        overlay = yaml.safe_load(
            (tmp_path / target.value / "truss-config-overlay.yaml").read_text()
        )
        assert set(overlay) == {"environment_variables", "model_metadata"}


def test_unvalidated_runtime_version_is_rejected_without_fallback(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedCapabilityError, match="outside validated range"):
        export_physical_plan(
            context=_context(runtime=RuntimeKind.VLLM, runtime_version="0.10.2"),
            target=DeploymentTarget.LOCAL,
            output=tmp_path / "old-vllm",
        )


def test_vllm_disaggregation_requires_nixl_route(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedCapabilityError, match="NixlConnector"):
        export_physical_plan(
            context=_context(runtime=RuntimeKind.VLLM, runtime_version="0.12.0"),
            target=DeploymentTarget.DOCKER,
            output=tmp_path / "vllm",
        )


def test_context_rejects_topology_that_does_not_cover_rank_binding() -> None:
    context = _context()
    incomplete = context.topology.model_copy(
        update={
            "nodes": tuple(node for node in context.topology.nodes if node.node_id != "gpu-1"),
            "edges": tuple(
                edge
                for edge in context.topology.edges
                if edge.source_node_id != "gpu-1" and edge.target_node_id != "gpu-1"
            ),
        }
    )
    with pytest.raises(ValueError, match="gpu_id is absent"):
        FabricAdapterContext.model_validate(
            {**context.model_dump(mode="python"), "topology": incomplete}
        )


def test_validated_version_manifest_has_official_provenance() -> None:
    manifest = json.loads(
        (Path(__file__).parents[2] / "deploy/fabric/validated-versions.json").read_text()
    )
    assert manifest["validated_at"] == "2026-08-01"
    assert manifest["offline_only"] is True
    assert {item["name"] for item in manifest["components"]} == {
        "kubernetes",
        "vllm",
        "sglang",
        "nvidia-dynamo",
        "modal",
        "truss",
    }
    assert all(
        source.startswith("https://")
        for component in manifest["components"]
        for source in component["sources"]
    )
