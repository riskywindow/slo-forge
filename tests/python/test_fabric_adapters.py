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
    ParallelGroup,
    ParallelismKind,
    RankBinding,
    RankPlacement,
    WorkerRole,
    canonical_hash,
    load_physical_execution_plan,
    load_topology_graph,
)
from sloforge.ir import ArtifactDigest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "fabric"


def _context(
    *,
    runtime: RuntimeKind = RuntimeKind.NATIVE,
    runtime_version: str = "0.1.0",
    dynamo_backend: DynamoBackend | None = None,
    allow_advisory_cloud_metadata: bool = False,
) -> FabricAdapterContext:
    topology = load_topology_graph(FIXTURES / "topology-graph-v1.json")
    plan = load_physical_execution_plan(FIXTURES / "physical-execution-plan-v1.json")
    plan = plan.model_copy(
        update={"topology_fingerprint": ArtifactDigest(value=canonical_hash(topology))}
    )
    return FabricAdapterContext(
        plan=plan,
        topology=topology,
        model_id="Qwen/Qwen3-0.6B",
        model_revision="main",
        image="ghcr.io/sloforge/runtime:0.1.0",
        runtime=runtime,
        runtime_version=runtime_version,
        dynamo_backend=dynamo_backend,
        allow_advisory_cloud_metadata=allow_advisory_cloud_metadata,
    )


def test_adapter_rejects_plan_topology_artifact_mismatch() -> None:
    context = _context()
    mismatched = context.plan.model_copy(
        update={"topology_fingerprint": ArtifactDigest(value="f" * 64)}
    )
    with pytest.raises(ValueError, match="fingerprint does not match"):
        FabricAdapterContext.model_validate(
            {**context.model_dump(mode="python"), "plan": mismatched}
        )


def test_adapter_rejects_symlink_output_directory(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    output = tmp_path / "linked"
    try:
        output.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    with pytest.raises(ValueError, match="symbolic link"):
        export_physical_plan(context=_context(), target=DeploymentTarget.LOCAL, output=output)


def _aggregated_dynamo_context() -> FabricAdapterContext:
    base = _context(
        runtime=RuntimeKind.DYNAMO,
        runtime_version="1.3.0",
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


def _disaggregated_runtime_context(
    *,
    runtime: RuntimeKind,
    runtime_version: str,
    transfer_adapter: str,
    dynamo_backend: DynamoBackend | None = None,
) -> FabricAdapterContext:
    base = _context()
    assert base.plan.kv_transfer is not None
    first, second = base.plan.rank_placement.bindings
    bindings = (
        first.model_copy(update={"replica_id": "prefill-0", "worker_role": WorkerRole.PREFILL}),
        second.model_copy(update={"replica_id": "decode-0", "worker_role": WorkerRole.DECODE}),
    )
    route = base.plan.kv_transfer.routes[0].model_copy(
        update={"transport_adapter": transfer_adapter}
    )
    parallelism = base.plan.parallelism.model_copy(
        update={
            "tensor_parallel_degree": 1,
            "data_parallel_degree": 2,
            "expert_parallel_degree": 1,
            "groups": (
                ParallelGroup(group_id="prefill-0", kind=ParallelismKind.PREFILL, rank_ids=(0,)),
                ParallelGroup(group_id="decode-0", kind=ParallelismKind.DECODE, rank_ids=(1,)),
            ),
            "replica_groups": (
                ParallelGroup(group_id="replica-p", kind=ParallelismKind.DATA, rank_ids=(0,)),
                ParallelGroup(group_id="replica-d", kind=ParallelismKind.DATA, rank_ids=(1,)),
            ),
        }
    )
    plan = base.plan.model_copy(
        update={
            "parallelism": parallelism,
            "rank_placement": RankPlacement(bindings=bindings),
            "kv_transfer": base.plan.kv_transfer.model_copy(update={"routes": (route,)}),
        }
    )
    return FabricAdapterContext.model_validate(
        {
            **base.model_dump(mode="python"),
            "plan": plan,
            "runtime": runtime,
            "runtime_version": runtime_version,
            "dynamo_backend": dynamo_backend,
        }
    )


def _wide_ep_context(*, runtime: RuntimeKind) -> FabricAdapterContext:
    backend = DynamoBackend.VLLM if runtime is RuntimeKind.DYNAMO else None
    base = _disaggregated_runtime_context(
        runtime=runtime,
        runtime_version="1.3.0" if runtime is RuntimeKind.DYNAMO else "0.26.0",
        transfer_adapter="nixl",
        dynamo_backend=backend,
    )
    bindings = tuple(
        binding.model_copy(update={"worker_role": WorkerRole.AGGREGATED})
        for binding in base.plan.rank_placement.bindings
    )
    parallelism = base.plan.parallelism.model_copy(
        update={
            "expert_parallel_degree": 2,
            "prefill_decode_disaggregated": False,
            "groups": (
                ParallelGroup(group_id="tp-0", kind=ParallelismKind.TENSOR, rank_ids=(0,)),
                ParallelGroup(group_id="tp-1", kind=ParallelismKind.TENSOR, rank_ids=(1,)),
                ParallelGroup(group_id="ep-0", kind=ParallelismKind.EXPERT, rank_ids=(0, 1)),
            ),
        }
    )
    plan = base.plan.model_copy(
        update={
            "parallelism": parallelism,
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
    config_map = next(item for item in manifest["items"] if item["kind"] == "ConfigMap")
    assert isinstance(config_map["data"]["physical-plan.json"], str)
    assert json.loads(config_map["data"]["physical-plan.json"])["kind"] == "PhysicalExecutionPlan"
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
    persisted_result = json.loads(
        (tmp_path / "kubernetes" / "export-result.json").read_text(encoding="utf-8")
    )
    assert result.model_dump(mode="json") == persisted_result


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
    assert "--enable-dp-attention" not in main["args"]
    dp_index = main["args"].index("--data-parallel-size")
    assert main["args"][dp_index + 1] == "1"
    assert main["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert "runtimeVersionOverride" not in worker
    assert "readinessProbe" not in main
    assert "livenessProbe" not in main
    assert "startupProbe" not in main
    assert {item["name"] for item in main["env"]} >= {
        "SLOFORGE_PHYSICAL_PLAN_HASH",
        "SLOFORGE_RANK_BINDINGS",
    }
    assert (
        worker["podTemplate"]["metadata"]["annotations"]["sloforge.dev/binding-policy"]
        == "fail-on-mismatch"
    )
    assert (
        worker["podTemplate"]["metadata"]["annotations"]["sloforge.dev/health-probes"]
        == "dynamo-operator-v1beta1-defaults"
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


def test_pre_v1beta1_dynamo_is_rejected_without_schema_fallback(tmp_path: Path) -> None:
    context = _aggregated_dynamo_context().model_copy(update={"runtime_version": "1.0.2"})
    with pytest.raises(UnsupportedCapabilityError, match="outside validated range"):
        export_physical_plan(
            context=context,
            target=DeploymentTarget.DYNAMO,
            output=tmp_path / "old-dynamo",
        )


def test_generic_kubernetes_rejects_unenforced_multi_node_gang(tmp_path: Path) -> None:
    context = _context()
    first, second = context.plan.rank_placement.bindings
    # The compatibility boundary is evaluated before rendering. A second host
    # proves that setting a scheduler enum does not masquerade as an emitted
    # PodGroup/LeaderWorkerSet contract.
    multi_host_plan = context.plan.model_copy(
        update={
            "rank_placement": RankPlacement(
                bindings=(first, second.model_copy(update={"host_id": "host-1"}))
            )
        }
    )
    with pytest.raises(UnsupportedCapabilityError, match="cannot enforce multi-node gang"):
        export_physical_plan(
            context=context.model_copy(update={"plan": multi_host_plan}),
            target=DeploymentTarget.KUBERNETES,
            output=tmp_path / "multi-node",
        )


def test_vllm_disaggregation_requires_nixl_route(tmp_path: Path) -> None:
    context = _disaggregated_runtime_context(
        runtime=RuntimeKind.VLLM,
        runtime_version="0.26.0",
        transfer_adapter="fixture",
    )
    with pytest.raises(UnsupportedCapabilityError, match="NixlConnector"):
        export_physical_plan(
            context=context,
            target=DeploymentTarget.DOCKER,
            output=tmp_path / "vllm",
        )


def test_direct_vllm_uses_native_kv_config_not_dynamo_wrapper_mode(tmp_path: Path) -> None:
    context = _disaggregated_runtime_context(
        runtime=RuntimeKind.VLLM,
        runtime_version="0.26.0",
        transfer_adapter="nixl",
    )
    export_physical_plan(
        context=context,
        target=DeploymentTarget.DOCKER,
        output=tmp_path / "vllm-disagg",
    )
    compose = yaml.safe_load((tmp_path / "vllm-disagg" / "compose.yaml").read_text())
    for service in compose["services"].values():
        command = service["command"]
        assert "--disaggregation-mode" not in command
        transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
        assert transfer == {"kv_connector": "NixlConnector", "kv_role": "kv_both"}


def test_vllm_wide_ep_fails_when_exporter_emits_independent_replicas(
    tmp_path: Path,
) -> None:
    with pytest.raises(UnsupportedCapabilityError, match="computes EP_SIZE"):
        export_physical_plan(
            context=_wide_ep_context(runtime=RuntimeKind.VLLM),
            target=DeploymentTarget.DOCKER,
            output=tmp_path / "rejected",
        )


def test_dynamo_vllm_wide_ep_uses_role_local_data_parallel_size(tmp_path: Path) -> None:
    export_physical_plan(
        context=_wide_ep_context(runtime=RuntimeKind.DYNAMO),
        target=DeploymentTarget.DYNAMO,
        output=tmp_path / "dynamo-wide-ep",
    )
    graph = yaml.safe_load(
        (tmp_path / "dynamo-wide-ep" / "dynamo-graph-deployment.yaml").read_text()
    )
    worker = next(item for item in graph["spec"]["components"] if item["type"] == "worker")
    args = worker["podTemplate"]["spec"]["containers"][0]["args"]
    dp_index = args.index("--data-parallel-size")
    assert args[dp_index + 1] == "2"
    assert "--enable-expert-parallel" in args


def test_sglang_ep_does_not_enable_dp_attention(tmp_path: Path) -> None:
    context = _aggregated_dynamo_context().model_copy(
        update={
            "runtime": RuntimeKind.SGLANG,
            "runtime_version": "0.5.15",
            "dynamo_backend": None,
        }
    )
    export_physical_plan(
        context=context,
        target=DeploymentTarget.DOCKER,
        output=tmp_path / "sglang",
    )
    compose = yaml.safe_load((tmp_path / "sglang" / "compose.yaml").read_text())
    command = next(iter(compose["services"].values()))["command"]
    assert command[command.index("--ep-size") + 1] == "2"
    assert command[command.index("--dp-size") + 1] == "1"
    assert "--enable-dp-attention" not in command


def test_sglang_disaggregation_emits_validated_transfer_and_nic(tmp_path: Path) -> None:
    context = _disaggregated_runtime_context(
        runtime=RuntimeKind.SGLANG,
        runtime_version="0.5.15",
        transfer_adapter="nixl",
    )
    export_physical_plan(
        context=context,
        target=DeploymentTarget.DOCKER,
        output=tmp_path / "sglang-disagg",
    )
    compose = yaml.safe_load((tmp_path / "sglang-disagg" / "compose.yaml").read_text())
    for service in compose["services"].values():
        command = service["command"]
        assert command[command.index("--disaggregation-transfer-backend") + 1] == "nixl"
        assert command[command.index("--disaggregation-ib-device") + 1] == "mlx5_0"
        assert command[command.index("--dp-size") + 1] == "1"
        assert "--enable-dp-attention" not in command


def test_sglang_rejects_mixed_disaggregation_backends(tmp_path: Path) -> None:
    context = _disaggregated_runtime_context(
        runtime=RuntimeKind.SGLANG,
        runtime_version="0.5.15",
        transfer_adapter="nixl",
    )
    assert context.plan.kv_transfer is not None
    first = context.plan.kv_transfer.routes[0]
    routes = (
        first,
        first.model_copy(update={"route_id": "kv-mooncake", "transport_adapter": "mooncake"}),
    )
    plan = context.plan.model_copy(
        update={"kv_transfer": context.plan.kv_transfer.model_copy(update={"routes": routes})}
    )
    with pytest.raises(UnsupportedCapabilityError, match="mixed NIXL and Mooncake"):
        export_physical_plan(
            context=context.model_copy(update={"plan": plan}),
            target=DeploymentTarget.DOCKER,
            output=tmp_path / "mixed",
        )


def test_dynamo_disaggregated_workers_use_role_local_dp_and_mode(tmp_path: Path) -> None:
    context = _disaggregated_runtime_context(
        runtime=RuntimeKind.DYNAMO,
        runtime_version="1.3.0",
        transfer_adapter="nixl",
        dynamo_backend=DynamoBackend.VLLM,
    )
    export_physical_plan(
        context=context,
        target=DeploymentTarget.DYNAMO,
        output=tmp_path / "dynamo-disagg",
    )
    graph = yaml.safe_load(
        (tmp_path / "dynamo-disagg" / "dynamo-graph-deployment.yaml").read_text()
    )
    workers = [
        item for item in graph["spec"]["components"] if item["type"] in {"prefill", "decode"}
    ]
    assert {item["type"] for item in workers} == {"prefill", "decode"}
    for worker in workers:
        args = worker["podTemplate"]["spec"]["containers"][0]["args"]
        assert args[args.index("--data-parallel-size") + 1] == "1"
        assert args[args.index("--disaggregation-mode") + 1] == worker["type"]


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


def test_context_rejects_node_kind_and_host_ownership_mismatch() -> None:
    context = _context()
    first, *remaining = context.plan.rank_placement.bindings
    mismatched = context.plan.model_copy(
        update={
            "rank_placement": RankPlacement(
                bindings=(first.model_copy(update={"host_id": "gpu-0"}), *remaining)
            )
        }
    )
    with pytest.raises(ValueError, match="host_id is absent or not a host"):
        FabricAdapterContext.model_validate(
            {**context.model_dump(mode="python"), "plan": mismatched.model_dump(mode="python")}
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
