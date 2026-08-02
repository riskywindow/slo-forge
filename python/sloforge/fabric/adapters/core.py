"""Offline physical-plan lowering for local and deployment runtimes."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from sloforge.fabric.adapters.compatibility import validate_runtime
from sloforge.fabric.adapters.models import (
    AdapterCapabilities,
    DeploymentTarget,
    FabricAdapterContext,
    FabricExportResult,
    GeneratedArtifact,
    RuntimeKind,
)
from sloforge.fabric.ir import (
    GpuNode,
    NicNode,
    RankBinding,
    WorkerRole,
    canonical_hash,
    canonical_json,
)
from sloforge.util import sha256_file


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True))


def _write_yaml(path: Path, value: object) -> None:
    _atomic_write(path, yaml.safe_dump(value, sort_keys=False))


def _rank_groups(bindings: Iterable[RankBinding]) -> dict[tuple[str, str], list[RankBinding]]:
    grouped: dict[tuple[str, str], list[RankBinding]] = defaultdict(list)
    for binding in bindings:
        grouped[(binding.replica_id, binding.host_id)].append(binding)
    for group in grouped.values():
        group.sort(key=lambda binding: binding.rank_id)
    return dict(sorted(grouped.items()))


def _gpu_uuid(context: FabricAdapterContext, gpu_id: str) -> str:
    for node in context.topology.nodes:
        if isinstance(node, GpuNode) and node.node_id == gpu_id:
            return node.uuid
    raise ValueError(f"GPU {gpu_id} is absent from topology")


def _nic_interface(context: FabricAdapterContext, nic_id: str | None) -> str | None:
    if nic_id is None:
        return None
    for node in context.topology.nodes:
        if isinstance(node, NicNode) and node.node_id == nic_id:
            return node.interface
    raise ValueError(f"NIC {nic_id} is absent from topology")


def _cpu_union(bindings: Iterable[RankBinding]) -> str:
    return ",".join(binding.process_cpu_affinity for binding in bindings)


def _role(bindings: Iterable[RankBinding]) -> WorkerRole:
    roles = {binding.worker_role for binding in bindings}
    if len(roles) == 1:
        return roles.pop()
    return WorkerRole.AGGREGATED


def _runtime_command(
    context: FabricAdapterContext,
    *,
    role: WorkerRole,
) -> list[str]:
    plan = context.plan
    if context.runtime is RuntimeKind.NATIVE:
        return [
            "sloforge",
            "serve",
            "--plan",
            plan.logical_deployment_plan.uri,
        ]
    if context.runtime is RuntimeKind.VLLM:
        command = [
            "vllm",
            "serve",
            context.model_id,
            "--revision",
            context.model_revision,
            "--tensor-parallel-size",
            str(plan.parallelism.tensor_parallel_degree),
            "--pipeline-parallel-size",
            str(plan.parallelism.pipeline_parallel_degree),
            "--data-parallel-size",
            str(plan.parallelism.data_parallel_degree),
        ]
        if plan.parallelism.expert_parallel_degree > 1:
            command.append("--enable-expert-parallel")
        if plan.parallelism.prefill_decode_disaggregated:
            transfer = {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_both",
            }
            command.extend(["--kv-transfer-config", json.dumps(transfer, separators=(",", ":"))])
        return command
    if context.runtime is RuntimeKind.SGLANG:
        command = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            context.model_id,
            "--tp",
            str(plan.parallelism.tensor_parallel_degree),
            "--pp",
            str(plan.parallelism.pipeline_parallel_degree),
            "--dp",
            str(plan.parallelism.data_parallel_degree),
            "--ep-size",
            str(plan.parallelism.expert_parallel_degree),
        ]
        if plan.parallelism.expert_parallel_degree > 1:
            command.append("--enable-dp-attention")
        if plan.parallelism.prefill_decode_disaggregated:
            if role not in {WorkerRole.PREFILL, WorkerRole.DECODE}:
                raise ValueError("disaggregated SGLang group must have one worker role")
            command.extend(["--disaggregation-mode", role.value])
        return command
    raise ValueError("Dynamo commands are emitted as DynamoGraphDeployment components")


def _binding_payload(
    context: FabricAdapterContext, bindings: list[RankBinding]
) -> list[dict[str, Any]]:
    return [
        {
            "rank_id": binding.rank_id,
            "host_id": binding.host_id,
            "gpu_id": binding.gpu_id,
            "expected_gpu_uuid": _gpu_uuid(context, binding.gpu_id),
            "numa_domain_id": binding.numa_domain_id,
            "nic_id": binding.nic_id,
            "nic_interface": _nic_interface(context, binding.nic_id),
            "network_rail_id": binding.network_rail_id,
            "cpu_affinity": binding.process_cpu_affinity,
            "role": binding.worker_role.value,
            "replica_id": binding.replica_id,
            "fault_domain": binding.fault_domain,
        }
        for binding in bindings
    ]


def _base_environment(
    context: FabricAdapterContext,
    bindings: list[RankBinding],
) -> dict[str, str]:
    interfaces = sorted(
        {
            interface
            for binding in bindings
            if (interface := _nic_interface(context, binding.nic_id)) is not None
        }
    )
    environment = {
        "SLOFORGE_PHYSICAL_PLAN_ID": context.plan.plan_id,
        "SLOFORGE_PHYSICAL_PLAN_HASH": canonical_hash(context.plan),
        "SLOFORGE_RANK_BINDINGS": json.dumps(
            _binding_payload(context, bindings), separators=(",", ":"), sort_keys=True
        ),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    if interfaces:
        environment["NCCL_SOCKET_IFNAME"] = ",".join(interfaces)
    return environment


def _render_common(context: FabricAdapterContext, output: Path) -> None:
    _atomic_write(output / "physical-plan.json", canonical_json(context.plan).decode("utf-8"))
    _atomic_write(output / "topology.json", canonical_json(context.topology).decode("utf-8"))


def _render_local(context: FabricAdapterContext, output: Path) -> None:
    _render_common(context, output)
    launch_groups: list[dict[str, Any]] = []
    for (replica_id, host_id), bindings in _rank_groups(
        context.plan.rank_placement.bindings
    ).items():
        launch_groups.append(
            {
                "group_id": f"{replica_id}-{host_id}",
                "host_id": host_id,
                "rank_bindings": _binding_payload(context, bindings),
                "cpu_affinity": _cpu_union(bindings),
                "command": _runtime_command(context, role=_role(bindings)),
                "environment": _base_environment(context, bindings),
                "startup_timeout_seconds": 600,
                "shutdown_grace_seconds": context.shutdown_grace_seconds,
                "maximum_pending_requests": 256,
            }
        )
    _write_json(
        output / "launch-plan.json",
        {
            "schema_version": "sloforge.fabric-launch/v1",
            "plan_id": context.plan.plan_id,
            "plan_hash": canonical_hash(context.plan),
            "runtime": context.runtime.value,
            "runtime_version": context.runtime_version,
            "groups": launch_groups,
            "binding_policy": "fail_on_mismatch",
        },
    )


def _service_name(replica_id: str, host_id: str) -> str:
    normalized = "-".join((replica_id, host_id)).lower()
    return "".join(
        character if character.isalnum() or character == "-" else "-" for character in normalized
    )


def _render_docker(context: FabricAdapterContext, output: Path) -> None:
    _render_common(context, output)
    services: dict[str, Any] = {}
    for (replica_id, host_id), bindings in _rank_groups(
        context.plan.rank_placement.bindings
    ).items():
        name = _service_name(replica_id, host_id)
        environment = _base_environment(context, bindings)
        environment["SLOFORGE_EXPECTED_HOST_ID"] = host_id
        services[name] = {
            "image": context.image,
            "command": _runtime_command(context, role=_role(bindings)),
            "environment": environment,
            "cpuset": _cpu_union(bindings),
            "cpus": context.cpu_limit_per_rank * len(bindings),
            "mem_limit": f"{context.memory_limit_gib_per_rank * len(bindings)}g",
            "pids_limit": context.pids_limit_per_rank * len(bindings),
            "init": True,
            "restart": "unless-stopped",
            "stop_grace_period": f"{context.shutdown_grace_seconds}s",
            "read_only": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "tmpfs": ["/tmp:rw,noexec,nosuid,size=1g"],
            "volumes": [
                "./physical-plan.json:/etc/sloforge/physical-plan.json:ro",
                "./topology.json:/etc/sloforge/topology.json:ro",
            ],
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "device_ids": [
                                    _gpu_uuid(context, binding.gpu_id) for binding in bindings
                                ],
                                "capabilities": ["gpu"],
                            }
                        ]
                    }
                }
            },
            "labels": {
                "sloforge.dev/physical-plan": context.plan.plan_id,
                "sloforge.dev/host": host_id,
                "sloforge.dev/replica": replica_id,
            },
        }
    _write_yaml(
        output / "compose.yaml",
        {
            "name": f"sloforge-fabric-{context.plan.plan_id}",
            "services": services,
        },
    )


def _kubernetes_resources(
    context: FabricAdapterContext, bindings: list[RankBinding]
) -> dict[str, Any]:
    rank_count = len(bindings)
    limits: dict[str, str] = {
        "cpu": str(context.cpu_limit_per_rank * rank_count),
        "memory": f"{context.memory_limit_gib_per_rank * rank_count}Gi",
        context.gpu_resource_name: str(rank_count),
    }
    if context.rdma_resource_name is not None:
        limits[context.rdma_resource_name] = "1"
    return {"requests": dict(limits), "limits": limits}


def _render_kubernetes(context: FabricAdapterContext, output: Path) -> None:
    _render_common(context, output)
    plan_hash = canonical_hash(context.plan)
    resources: list[dict[str, Any]] = [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"sloforge-fabric-{context.plan.plan_id}",
                "namespace": context.namespace,
                "labels": {"sloforge.dev/physical-plan": context.plan.plan_id},
            },
            "data": {
                "physical-plan.json": canonical_json(context.plan),
                "topology.json": canonical_json(context.topology),
            },
        }
    ]
    for (replica_id, host_id), bindings in _rank_groups(
        context.plan.rank_placement.bindings
    ).items():
        name = f"fabric-{_service_name(replica_id, host_id)}"
        group_label = _service_name(replica_id, "group")
        environment = _base_environment(context, bindings)
        environment["SLOFORGE_EXPECTED_HOST_ID"] = host_id
        annotations = {
            "sloforge.dev/physical-plan-hash": plan_hash,
            "sloforge.dev/topology-fingerprint": context.plan.topology_fingerprint.value,
            "sloforge.dev/rank-ids": ",".join(str(item.rank_id) for item in bindings),
            "sloforge.dev/gpu-uuids": ",".join(
                _gpu_uuid(context, item.gpu_id) for item in bindings
            ),
            "sloforge.dev/binding-policy": "fail-on-mismatch",
        }
        affinity = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "kubernetes.io/hostname",
                                    "operator": "In",
                                    "values": [host_id],
                                }
                            ]
                        }
                    ]
                }
            },
            "podAntiAffinity": {
                "preferredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "weight": 100,
                        "podAffinityTerm": {
                            "topologyKey": "kubernetes.io/hostname",
                            "labelSelector": {
                                "matchLabels": {"sloforge.dev/replica-group": group_label}
                            },
                        },
                    }
                ]
            },
        }
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": name,
                "namespace": context.namespace,
                "labels": {
                    "app.kubernetes.io/name": "sloforge-fabric-worker",
                    "sloforge.dev/physical-plan": context.plan.plan_id,
                    "sloforge.dev/replica-group": group_label,
                },
                "annotations": annotations,
            },
            "spec": {
                "replicas": 1,
                "revisionHistoryLimit": 2,
                "progressDeadlineSeconds": 900,
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {"maxUnavailable": 0, "maxSurge": 1},
                },
                "selector": {"matchLabels": {"sloforge.dev/worker": name}},
                "template": {
                    "metadata": {
                        "labels": {
                            "sloforge.dev/worker": name,
                            "sloforge.dev/replica-group": group_label,
                        },
                        "annotations": annotations,
                    },
                    "spec": {
                        "automountServiceAccountToken": False,
                        "terminationGracePeriodSeconds": context.shutdown_grace_seconds,
                        "securityContext": {
                            "runAsNonRoot": True,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "affinity": affinity,
                        "topologySpreadConstraints": [
                            {
                                "maxSkew": 1,
                                "topologyKey": "topology.kubernetes.io/zone",
                                "whenUnsatisfiable": "ScheduleAnyway",
                                "labelSelector": {
                                    "matchLabels": {"sloforge.dev/replica-group": group_label}
                                },
                            }
                        ],
                        "containers": [
                            {
                                "name": "worker",
                                "image": context.image,
                                "command": _runtime_command(context, role=_role(bindings)),
                                "env": [
                                    {"name": key, "value": value}
                                    for key, value in sorted(environment.items())
                                ],
                                "resources": _kubernetes_resources(context, bindings),
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "readOnlyRootFilesystem": True,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                "startupProbe": {
                                    "tcpSocket": {"port": 8000},
                                    "periodSeconds": 5,
                                    "failureThreshold": 120,
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "periodSeconds": 5,
                                    "failureThreshold": 6,
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "periodSeconds": 15,
                                    "failureThreshold": 4,
                                },
                                "volumeMounts": [
                                    {
                                        "name": "fabric-plan",
                                        "mountPath": "/etc/sloforge",
                                        "readOnly": True,
                                    },
                                    {"name": "tmp", "mountPath": "/tmp"},
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "fabric-plan",
                                "configMap": {"name": f"sloforge-fabric-{context.plan.plan_id}"},
                            },
                            {"name": "tmp", "emptyDir": {"sizeLimit": "1Gi"}},
                        ],
                    },
                },
            },
        }
        resources.append(deployment)
    _write_yaml(
        output / "physical-plan.yaml",
        {"apiVersion": "v1", "kind": "List", "items": resources},
    )


def _dynamo_worker_args(context: FabricAdapterContext) -> list[str]:
    plan = context.plan.parallelism
    if context.dynamo_backend is None:
        raise ValueError("Dynamo backend missing")
    if context.dynamo_backend.value == "vllm":
        args = [
            "--model",
            context.model_id,
            "--tensor-parallel-size",
            str(plan.tensor_parallel_degree),
            "--pipeline-parallel-size",
            str(plan.pipeline_parallel_degree),
            "--data-parallel-size",
            str(plan.data_parallel_degree),
        ]
        if plan.expert_parallel_degree > 1:
            args.append("--enable-expert-parallel")
        return args
    args = [
        "--model-path",
        context.model_id,
        "--tp",
        str(plan.tensor_parallel_degree),
        "--pp",
        str(plan.pipeline_parallel_degree),
        "--dp",
        str(plan.data_parallel_degree),
        "--ep-size",
        str(plan.expert_parallel_degree),
    ]
    if plan.expert_parallel_degree > 1:
        args.append("--enable-dp-attention")
    return args


def _render_dynamo(context: FabricAdapterContext, output: Path) -> None:
    _render_common(context, output)
    backend = context.dynamo_backend
    if backend is None:
        raise ValueError("Dynamo backend missing")
    plan_hash = canonical_hash(context.plan)
    by_role: dict[WorkerRole, list[RankBinding]] = defaultdict(list)
    for binding in context.plan.rank_placement.bindings:
        by_role[binding.worker_role].append(binding)
    components: list[dict[str, Any]] = [{"name": "Frontend", "type": "frontend", "replicas": 1}]
    for role, bindings in sorted(by_role.items(), key=lambda item: item[0].value):
        hosts = sorted({binding.host_id for binding in bindings})
        per_host = max(sum(binding.host_id == host for binding in bindings) for host in hosts)
        limits: dict[str, str] = {context.gpu_resource_name: str(per_host)}
        if context.rdma_resource_name is not None:
            limits[context.rdma_resource_name] = "1"
        args = _dynamo_worker_args(context)
        if context.plan.parallelism.prefill_decode_disaggregated:
            if role not in {WorkerRole.PREFILL, WorkerRole.DECODE}:
                raise ValueError("Dynamo disaggregated component has a non-P/D worker role")
            if context.dynamo_backend is not None and context.dynamo_backend.value == "vllm":
                args.extend(
                    [
                        "--kv-transfer-config",
                        json.dumps(
                            {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
                            separators=(",", ":"),
                        ),
                    ]
                )
        component: dict[str, Any] = {
            "name": f"{role.value.title()}Worker",
            "type": role.value if role in {WorkerRole.PREFILL, WorkerRole.DECODE} else "worker",
            "replicas": 1,
            "podTemplate": {
                "metadata": {
                    "annotations": {
                        "sloforge.dev/physical-plan-hash": plan_hash,
                        "sloforge.dev/rank-bindings": json.dumps(
                            _binding_payload(context, bindings),
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "sloforge.dev/binding-policy": "fail-on-mismatch",
                    }
                },
                "spec": {
                    "automountServiceAccountToken": False,
                    "terminationGracePeriodSeconds": context.shutdown_grace_seconds,
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [
                                    {
                                        "matchExpressions": [
                                            {
                                                "key": "kubernetes.io/hostname",
                                                "operator": "In",
                                                "values": hosts,
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    },
                    "containers": [
                        {
                            "name": "main",
                            "image": context.image,
                            "command": ["python3", "-m", f"dynamo.{backend.value}"],
                            "args": args,
                            "resources": {"limits": limits, "requests": dict(limits)},
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "periodSeconds": 5,
                                "failureThreshold": 12,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "periodSeconds": 15,
                                "failureThreshold": 4,
                            },
                        }
                    ],
                },
            },
        }
        if len(hosts) > 1:
            component["multinode"] = {"nodeCount": len(hosts)}
        components.append(component)
    annotations = {
        "sloforge.dev/physical-plan-hash": plan_hash,
        "sloforge.dev/topology-fingerprint": context.plan.topology_fingerprint.value,
    }
    if context.gang_scheduler.value == "lws":
        annotations["nvidia.com/enable-grove"] = "false"
    dgd = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {
            "name": f"sloforge-{context.plan.plan_id}",
            "namespace": context.namespace,
            "annotations": annotations,
        },
        "spec": {
            "backendFramework": backend.value,
            "annotations": annotations,
            "components": components,
        },
    }
    _write_yaml(output / "dynamo-graph-deployment.yaml", dgd)


def _render_cloud_metadata(
    context: FabricAdapterContext,
    output: Path,
    target: DeploymentTarget,
) -> None:
    _render_common(context, output)
    metadata = {
        "schema_version": "sloforge.fabric-cloud-metadata/v1",
        "target": target.value,
        "plan_id": context.plan.plan_id,
        "physical_plan_hash": canonical_hash(context.plan),
        "topology_fingerprint": context.plan.topology_fingerprint.value,
        "environment_variables": {
            "SLOFORGE_PHYSICAL_PLAN_ID": context.plan.plan_id,
            "SLOFORGE_PHYSICAL_PLAN_HASH": canonical_hash(context.plan),
            "SLOFORGE_TOPOLOGY_FINGERPRINT": context.plan.topology_fingerprint.value,
        },
        "enforcement": "advisory",
        "unsupported_enforcement": [
            "rank_to_gpu_uuid",
            "rank_to_numa_domain",
            "rank_to_nic",
            "rank_to_network_rail",
        ],
        "deployment_mutation_performed": False,
    }
    _write_json(output / "physical-plan-metadata.json", metadata)
    if target is DeploymentTarget.MODAL:
        _write_json(
            output / "modal-image-env.json",
            {
                "api": "modal.Image.env",
                "variables": metadata["environment_variables"],
                "source": "https://modal.com/docs/reference/modal.Image#env",
            },
        )
    else:
        _write_yaml(
            output / "truss-config-overlay.yaml",
            {
                "environment_variables": metadata["environment_variables"],
                "model_metadata": {
                    "sloforge_fabric": {
                        "physical_plan_hash": canonical_hash(context.plan),
                        "topology_fingerprint": context.plan.topology_fingerprint.value,
                        "enforcement": "advisory",
                    }
                },
            },
        )


def _capabilities(target: DeploymentTarget) -> AdapterCapabilities:
    if target is DeploymentTarget.LOCAL:
        return AdapterCapabilities(
            exact_host_placement=True,
            exact_gpu_uuid_placement=True,
            numa_affinity=True,
            nic_affinity=True,
            network_rail_affinity=True,
            multi_node_gang_scheduling=False,
            prefill_decode_disaggregation=True,
            expert_parallelism=True,
        )
    if target is DeploymentTarget.DOCKER:
        return AdapterCapabilities(
            exact_host_placement=True,
            exact_gpu_uuid_placement=True,
            numa_affinity=True,
            nic_affinity=True,
            network_rail_affinity=False,
            multi_node_gang_scheduling=False,
            prefill_decode_disaggregation=True,
            expert_parallelism=True,
            advisory_only=("network rail selection requires the host runtime",),
        )
    if target in {DeploymentTarget.KUBERNETES, DeploymentTarget.DYNAMO}:
        return AdapterCapabilities(
            exact_host_placement=True,
            exact_gpu_uuid_placement=False,
            numa_affinity=False,
            nic_affinity=False,
            network_rail_affinity=False,
            multi_node_gang_scheduling=target is DeploymentTarget.DYNAMO,
            prefill_decode_disaggregation=True,
            expert_parallelism=True,
            advisory_only=(
                "GPU UUID, NUMA, NIC, and rail bindings are runtime-asserted because the "
                "Kubernetes device-plugin scheduler allocates resource counts",
            ),
        )
    return AdapterCapabilities(
        exact_host_placement=False,
        exact_gpu_uuid_placement=False,
        numa_affinity=False,
        nic_affinity=False,
        network_rail_affinity=False,
        multi_node_gang_scheduling=False,
        prefill_decode_disaggregation=False,
        expert_parallelism=False,
        advisory_only=("physical fields are exported only as supported metadata",),
    )


def _validate_output(target: DeploymentTarget, output: Path) -> tuple[str, ...]:
    plan = json.loads((output / "physical-plan.json").read_text(encoding="utf-8"))
    topology = json.loads((output / "topology.json").read_text(encoding="utf-8"))
    if plan.get("kind") != "PhysicalExecutionPlan" or topology.get("kind") != "TopologyGraph":
        raise ValueError("canonical Fabric documents are missing from adapter output")
    validations = ["canonical physical plan and topology parsed"]
    if target is DeploymentTarget.LOCAL:
        launch = json.loads((output / "launch-plan.json").read_text(encoding="utf-8"))
        if launch.get("binding_policy") != "fail_on_mismatch" or not launch.get("groups"):
            raise ValueError("local launch plan is not fail-closed")
        validations.append("local process placement and binding policy validated")
    elif target is DeploymentTarget.DOCKER:
        compose = yaml.safe_load((output / "compose.yaml").read_text(encoding="utf-8"))
        if not compose.get("services"):
            raise ValueError("Docker Compose has no rank groups")
        for service in compose["services"].values():
            if service.get("pids_limit", 0) <= 0 or not service.get("read_only"):
                raise ValueError("Docker rank group lacks bounded hardened resources")
        validations.append("Docker rank groups, GPU UUIDs, mounts, and bounds validated")
    elif target is DeploymentTarget.KUBERNETES:
        manifest = yaml.safe_load((output / "physical-plan.yaml").read_text(encoding="utf-8"))
        deployments = [item for item in manifest["items"] if item["kind"] == "Deployment"]
        if not deployments:
            raise ValueError("Kubernetes export contains no workers")
        if any(
            "nodeAffinity" not in item["spec"]["template"]["spec"]["affinity"]
            or "topologySpreadConstraints" not in item["spec"]["template"]["spec"]
            for item in deployments
        ):
            raise ValueError("Kubernetes worker lacks physical scheduling constraints")
        validations.append("Kubernetes affinity, spread, resources, probes, and rollout validated")
    elif target is DeploymentTarget.DYNAMO:
        dgd = yaml.safe_load((output / "dynamo-graph-deployment.yaml").read_text(encoding="utf-8"))
        if dgd.get("apiVersion") != "nvidia.com/v1beta1" or not dgd["spec"].get("components"):
            raise ValueError("Dynamo v1beta1 graph is incomplete")
        validations.append("Dynamo v1beta1 component graph and physical metadata validated")
    else:
        metadata = json.loads((output / "physical-plan-metadata.json").read_text(encoding="utf-8"))
        if metadata.get("enforcement") != "advisory" or metadata.get(
            "deployment_mutation_performed"
        ):
            raise ValueError("cloud metadata must be explicit and offline")
        if target is DeploymentTarget.MODAL:
            overlay = json.loads((output / "modal-image-env.json").read_text(encoding="utf-8"))
            if overlay.get("api") != "modal.Image.env":
                raise ValueError("Modal metadata is not mapped to the documented Image.env API")
        else:
            overlay = yaml.safe_load(
                (output / "truss-config-overlay.yaml").read_text(encoding="utf-8")
            )
            if set(overlay) != {"environment_variables", "model_metadata"}:
                raise ValueError("Truss overlay contains unsupported top-level fields")
        validations.append("offline advisory metadata and unsupported fields validated")
    return tuple(validations)


def export_physical_plan(
    *,
    context: FabricAdapterContext,
    target: DeploymentTarget,
    output: Path,
) -> FabricExportResult:
    """Lower a physical plan without deploying or mutating an external environment."""

    validate_runtime(context, target)
    output.mkdir(parents=True, exist_ok=True)
    renderers = {
        DeploymentTarget.LOCAL: _render_local,
        DeploymentTarget.DOCKER: _render_docker,
        DeploymentTarget.KUBERNETES: _render_kubernetes,
        DeploymentTarget.DYNAMO: _render_dynamo,
    }
    if target in renderers:
        renderers[target](context, output)
    else:
        _render_cloud_metadata(context, output, target)
    validations = _validate_output(target, output)
    artifacts = tuple(
        GeneratedArtifact(path=str(path.relative_to(output)), sha256=sha256_file(path))
        for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file())
    )
    result = FabricExportResult(
        target=target,
        plan_id=context.plan.plan_id,
        plan_hash=canonical_hash(context.plan),
        output_dir=output,
        capabilities=_capabilities(target),
        artifacts=artifacts,
        validations=validations,
    )
    _write_json(output / "export-result.json", result.model_dump(mode="json"))
    return result.model_copy(
        update={
            "artifacts": (
                *artifacts,
                GeneratedArtifact(
                    path="export-result.json",
                    sha256=sha256_file(output / "export-result.json"),
                ),
            )
        }
    )
