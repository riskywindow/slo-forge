"""Current-host topology discovery using structured OS and vendor interfaces."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from collections.abc import Iterable
from pathlib import Path

from sloforge.fabric.ir import (
    TopologyGraph as CanonicalTopologyGraph,
)
from sloforge.fabric.ir import (
    load_topology_graph,
    save_topology_graph,
)
from sloforge.fabric.topology.models import (
    DiscoveryTopologyGraph,
    EdgeKind,
    FactState,
    HealthState,
    NodeKind,
    Observation,
    ObservedFact,
    Provenance,
    SoftwareComponent,
    TopologyEdge,
    TopologyNode,
    Visibility,
    finalize_graph,
)
from sloforge.util import utc_now, write_json


def _provenance(
    source: str,
    source_kind: str,
    captured_at: str,
    *,
    confidence: float = 1.0,
    command: tuple[str, ...] = (),
) -> Provenance:
    return Provenance.model_validate(
        {
            "source": source,
            "source_kind": source_kind,
            "captured_at": captured_at,
            "confidence": confidence,
            "command": command,
        }
    )


def observed(
    name: str,
    observations: Iterable[tuple[object, Provenance]],
    *,
    unit: str | None = None,
) -> ObservedFact:
    """Normalize source observations without hiding disagreements."""
    normalized = tuple(
        Observation(
            value=value if isinstance(value, (str, int, float, bool)) else None, provenance=p
        )
        for value, p in observations
    )
    if not normalized:
        raise ValueError("observed() requires at least one source, even for an unknown fact")
    present = [item.value for item in normalized if item.value is not None]
    distinct = {json.dumps(item, sort_keys=True) for item in present}
    if not present:
        return ObservedFact(
            name=name, unit=unit, state=FactState.UNKNOWN, value=None, observations=normalized
        )
    if len(distinct) > 1:
        return ObservedFact(
            name=name, unit=unit, state=FactState.CONFLICT, value=None, observations=normalized
        )
    return ObservedFact(
        name=name, unit=unit, state=FactState.KNOWN, value=present[0], observations=normalized
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _run(argv: tuple[str, ...], timeout_s: float = 5.0) -> subprocess.CompletedProcess[str] | None:
    executable = shutil.which(argv[0])
    if executable is None:
        return None
    try:
        return subprocess.run(
            [executable, *argv[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _pcie_generation(speed: str | None) -> int | None:
    if speed is None:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GT/s", speed, re.IGNORECASE)
    if match is None:
        return None
    rate = float(match.group(1))
    return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}.get(rate)


def _memory_bytes() -> int | None:
    if platform.system() == "Linux":
        text = _read_text(Path("/proc/meminfo"))
        if text:
            match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", text, re.MULTILINE)
            return int(match.group(1)) * 1024 if match else None
    if platform.system() == "Darwin":
        result = _run(("sysctl", "-n", "hw.memsize"))
        return _parse_int(result.stdout.strip()) if result and result.returncode == 0 else None
    return None


def _visible_memory_bytes() -> int | None:
    for candidate in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        value = _read_text(candidate)
        if value and value != "max":
            parsed = _parse_int(value)
            if parsed is not None and parsed > 0:
                return parsed
    return _memory_bytes()


def _cpu_model() -> str | None:
    if platform.system() == "Linux":
        text = _read_text(Path("/proc/cpuinfo"))
        match = re.search(r"^(?:model name|Hardware)\s*:\s*(.+)$", text or "", re.MULTILINE)
        return match.group(1).strip() if match else platform.processor() or None
    if platform.system() == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            result = _run(("sysctl", "-n", key))
            if result and result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    return platform.processor() or None


def _physical_cpu_count() -> int | None:
    if platform.system() == "Darwin":
        result = _run(("sysctl", "-n", "hw.physicalcpu"))
        return _parse_int(result.stdout.strip()) if result and result.returncode == 0 else None
    if platform.system() == "Linux":
        text = _read_text(Path("/proc/cpuinfo"))
        if text:
            pairs = set(
                re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", text, re.DOTALL)
            )
            if pairs:
                return len(pairs)
    return None


def _socket_ids() -> tuple[int, ...]:
    found: set[int] = set()
    for candidate in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology/physical_package_id"):
        parsed = _parse_int(_read_text(candidate))
        if parsed is not None:
            found.add(parsed)
    if not found:
        found.add(0)
    return tuple(sorted(found))


def _numa_ids() -> tuple[int, ...]:
    found: list[int] = []
    for candidate in Path("/sys/devices/system/node").glob("node[0-9]*"):
        suffix = candidate.name.removeprefix("node")
        if suffix.isdigit():
            found.append(int(suffix))
    return tuple(sorted(set(found))) or (0,)


def _container_state() -> tuple[bool, tuple[str, ...], bool | None]:
    restrictions: list[str] = []
    cgroup = _read_text(Path("/proc/1/cgroup")) or ""
    in_container = (
        Path("/.dockerenv").exists()
        or "container" in os.environ
        or bool(re.search(r"docker|kubepods|containerd", cgroup))
    )
    nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if nvidia_visible is not None:
        restrictions.append(f"NVIDIA_VISIBLE_DEVICES={nvidia_visible}")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        restrictions.append(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
    host_visible: bool | None = None if in_container else True
    return in_container, tuple(restrictions), host_visible


def _discover_gpu_rows() -> tuple[tuple[str, ...], ...]:
    fields = (
        "index",
        "uuid",
        "name",
        "memory.total",
        "pci.bus_id",
        "driver_version",
        "compute_cap",
        "mig.mode.current",
        "ecc.errors.uncorrected.volatile.total",
        "clocks.sm",
        "power.draw",
    )
    result = _run(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ),
        timeout_s=10,
    )
    if result is None or result.returncode != 0:
        return ()
    rows: list[tuple[str, ...]] = []
    for line in result.stdout.splitlines():
        columns = tuple(item.strip() for item in line.split(","))
        if len(columns) == len(fields):
            rows.append(columns)
    return tuple(rows)


def _gpu_topology_edges(
    host_id: str,
    gpu_rows: tuple[tuple[str, ...], ...],
    captured_at: str,
) -> tuple[TopologyEdge, ...]:
    """Normalize the official nvidia-smi topology matrix when available."""
    result = _run(("nvidia-smi", "topo", "-m"), timeout_s=10)
    if result is None or result.returncode != 0 or not gpu_rows:
        return ()
    lines = [line.split() for line in result.stdout.splitlines() if line.strip()]
    header = next((line for line in lines if line and line[0].startswith("GPU")), None)
    if header is None:
        return ()
    gpu_labels = tuple(label for label in header if re.fullmatch(r"GPU\d+", label))
    rows = {line[0]: line[1 : 1 + len(gpu_labels)] for line in lines if line[0] in gpu_labels}
    indexed_nodes = {
        f"GPU{row[0]}": f"{host_id}/gpu/{row[1]}" for row in gpu_rows if row[0].isdigit()
    }
    provenance = _provenance(
        "nvidia-smi-topology", "command", captured_at, command=("nvidia-smi", "topo", "-m")
    )
    edges: list[TopologyEdge] = []
    for left_index, left_label in enumerate(gpu_labels):
        for right_index in range(left_index + 1, len(gpu_labels)):
            right_label = gpu_labels[right_index]
            row = rows.get(left_label)
            if row is None or right_index >= len(row):
                continue
            link = row[right_index]
            if (
                link in {"X", "N/A"}
                or left_label not in indexed_nodes
                or right_label not in indexed_nodes
            ):
                continue
            is_nvlink = link.startswith("NV")
            kind = EdgeKind.GPU_GPU
            left_node, right_node = indexed_nodes[left_label], indexed_nodes[right_label]
            edges.append(
                TopologyEdge(
                    edge_id=f"gpu-link:{left_node}:{right_node}",
                    source=left_node,
                    target=right_node,
                    kind=kind,
                    directed=False,
                    full_duplex=True,
                    sharing_group=f"{host_id}-{'nvlink' if is_nvlink else 'pcie'}",
                    contention_domain=f"{host_id}-{'nvlink' if is_nvlink else link.lower()}",
                    health=HealthState.HEALTHY,
                    facts=(
                        observed(
                            "connection_type", [("nvlink" if is_nvlink else "pcie", provenance)]
                        ),
                        observed("vendor_topology_code", [(link, provenance)]),
                        observed(
                            "theoretical_bandwidth", [(None, provenance)], unit="bytes_per_second"
                        ),
                        observed(
                            "measured_bandwidth", [(None, provenance)], unit="bytes_per_second"
                        ),
                        observed("latency", [(None, provenance)], unit="microseconds"),
                    ),
                )
            )
    return tuple(edges)


def _software(captured_at: str) -> tuple[SoftwareComponent, ...]:
    records: list[SoftwareComponent] = []
    specifications = (
        ("cuda", ("nvcc", "--version"), r"release\s+([0-9.]+)"),
        (
            "nvidia-driver",
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            r"([0-9.]+)",
        ),
        ("nccl", ("nccl-tests", "--version"), r"([0-9]+(?:\.[0-9]+)+)"),
        ("ibverbs", ("ibv_devinfo", "--version"), r"([0-9]+(?:\.[0-9]+)+)"),
        ("hwloc", ("lstopo", "--version"), r"([0-9]+(?:\.[0-9]+)+)"),
    )
    for name, argv, pattern in specifications:
        result = _run(argv)
        version: str | None = None
        if result and result.returncode == 0:
            match = re.search(pattern, result.stdout + result.stderr, re.IGNORECASE)
            version = match.group(1) if match else "present-version-unparsed"
        records.append(
            SoftwareComponent(
                name=name,
                version=version,
                state=FactState.KNOWN if version else FactState.UNKNOWN,
                provenance=_provenance(
                    name,
                    "command",
                    captured_at,
                    confidence=0.95 if version else 1.0,
                    command=argv,
                ),
            )
        )
    try:
        pynvml_version = importlib.metadata.version("nvidia-ml-py")
    except importlib.metadata.PackageNotFoundError:
        pynvml_version = None
    records.append(
        SoftwareComponent(
            name="nvml-python",
            version=pynvml_version,
            state=FactState.KNOWN if pynvml_version else FactState.UNKNOWN,
            provenance=_provenance("python-package-metadata", "api", captured_at),
        )
    )
    return tuple(records)


def discover_topology_records(*, topology_id: str | None = None) -> DiscoveryTopologyGraph:
    """Discover only hardware visible to the current process.

    Host-invisible devices are recorded as a visibility limitation; they are not
    guessed from a host inventory that the process cannot inspect.
    """
    captured_at = utc_now()
    host_id = topology_id or socket.gethostname()
    sys_prov = _provenance("python-platform", "api", captured_at)
    proc_prov = _provenance("procfs/sysfs", "sysfs", captured_at)
    nodes: list[TopologyNode] = [
        TopologyNode(
            node_id=host_id,
            kind=NodeKind.HOST,
            host_id=host_id,
            health=HealthState.HEALTHY,
            facts=(
                observed("hostname", [(socket.gethostname(), sys_prov)]),
                observed("operating_system", [(platform.platform(), sys_prov)]),
                observed("architecture", [(platform.machine() or None, sys_prov)]),
                observed("cpu_model", [(_cpu_model(), proc_prov)]),
                observed("logical_cpu_count", [(os.cpu_count(), sys_prov)], unit="count"),
                observed("memory_capacity", [(_memory_bytes(), proc_prov)], unit="bytes"),
                observed(
                    "visible_memory_capacity", [(_visible_memory_bytes(), proc_prov)], unit="bytes"
                ),
            ),
        )
    ]
    edges: list[TopologyEdge] = []
    sockets = _socket_ids()
    physical_cpu_count = _physical_cpu_count()
    for socket_id in sockets:
        node_id = f"{host_id}/socket/{socket_id}"
        nodes.append(
            TopologyNode(
                node_id=node_id,
                kind=NodeKind.CPU_SOCKET,
                host_id=host_id,
                health=HealthState.HEALTHY,
                facts=(
                    observed("socket_index", [(socket_id, proc_prov)]),
                    observed("cpu_model", [(_cpu_model(), proc_prov)]),
                    observed(
                        "physical_core_count",
                        [
                            (
                                physical_cpu_count // len(sockets)
                                if physical_cpu_count is not None
                                else None,
                                sys_prov,
                            )
                        ],
                        unit="count",
                    ),
                    observed(
                        "logical_cpu_count",
                        [((os.cpu_count() or 1) // len(sockets), sys_prov)],
                        unit="count",
                    ),
                ),
            )
        )
        edges.append(
            TopologyEdge(
                edge_id=f"contains:{host_id}:{node_id}",
                source=host_id,
                target=node_id,
                kind=EdgeKind.CONTAINS,
                directed=True,
                health=HealthState.HEALTHY,
            )
        )
    for numa_id in _numa_ids():
        node_id = f"{host_id}/numa/{numa_id}"
        memory_text = _read_text(Path(f"/sys/devices/system/node/node{numa_id}/meminfo"))
        memory_match = re.search(r"MemTotal:\s+(\d+)\s+kB", memory_text or "")
        capacity = int(memory_match.group(1)) * 1024 if memory_match else None
        if capacity is None and len(_numa_ids()) == 1:
            capacity = _visible_memory_bytes()
        cpu_set = _read_text(Path(f"/sys/devices/system/node/node{numa_id}/cpulist"))
        if cpu_set is None and len(_numa_ids()) == 1:
            cpu_set = f"0-{max(0, (os.cpu_count() or 1) - 1)}"
        nodes.append(
            TopologyNode(
                node_id=node_id,
                kind=NodeKind.NUMA_DOMAIN,
                host_id=host_id,
                health=HealthState.HEALTHY,
                facts=(
                    observed("numa_index", [(numa_id, proc_prov)]),
                    observed("cpu_set", [(cpu_set, proc_prov)]),
                    observed("memory_capacity", [(capacity, proc_prov)], unit="bytes"),
                ),
            )
        )
        socket_node = f"{host_id}/socket/{sockets[min(numa_id, len(sockets) - 1)]}"
        edges.append(
            TopologyEdge(
                edge_id=f"cpu-memory:{socket_node}:{node_id}",
                source=socket_node,
                target=node_id,
                kind=EdgeKind.CPU_MEMORY,
                directed=False,
                full_duplex=True,
                sharing_group=f"numa-{numa_id}",
                contention_domain=f"numa-memory-{numa_id}",
                health=HealthState.HEALTHY,
                facts=(
                    observed("measured_bandwidth", [(None, proc_prov)], unit="bytes_per_second"),
                ),
            )
        )

    gpu_rows = _discover_gpu_rows()
    visible_gpu_ids: list[str] = []
    gpu_prov = _provenance(
        "nvidia-smi-query", "command", captured_at, command=("nvidia-smi", "--query-gpu=...")
    )
    for row in gpu_rows:
        index, uuid, name, memory_mib, pci_bus, driver, compute, mig, ecc, clocks, power = row
        node_id = f"{host_id}/gpu/{uuid}"
        visible_gpu_ids.append(uuid)
        memory_bytes = _parse_int(memory_mib)
        nodes.append(
            TopologyNode(
                node_id=node_id,
                kind=NodeKind.GPU,
                host_id=host_id,
                health=HealthState.DEGRADED if (_parse_int(ecc) or 0) > 0 else HealthState.HEALTHY,
                facts=(
                    observed("index", [(_parse_int(index), gpu_prov)]),
                    observed("uuid", [(uuid, gpu_prov)]),
                    observed("product_name", [(name, gpu_prov)]),
                    observed(
                        "architecture",
                        [
                            (
                                f"compute-{compute}"
                                if compute and compute not in {"N/A", "[N/A]"}
                                else None,
                                gpu_prov,
                            )
                        ],
                    ),
                    observed(
                        "memory_capacity",
                        [
                            (
                                memory_bytes * 1024 * 1024 if memory_bytes is not None else None,
                                gpu_prov,
                            )
                        ],
                        unit="bytes",
                    ),
                    observed("pci_bus_id", [(pci_bus, gpu_prov)]),
                    observed("driver_version", [(driver, gpu_prov)]),
                    observed("compute_capability", [(compute, gpu_prov)]),
                    observed("mig_mode", [(mig, gpu_prov)]),
                    observed("uncorrected_ecc_errors", [(_parse_int(ecc), gpu_prov)], unit="count"),
                    observed("sm_clock", [(_parse_int(clocks), gpu_prov)], unit="megahertz"),
                    observed(
                        "power_draw",
                        [(float(power) if power.replace(".", "", 1).isdigit() else None, gpu_prov)],
                        unit="watts",
                    ),
                ),
            )
        )
        numa_path = Path("/sys/bus/pci/devices") / pci_bus.lower() / "numa_node"
        gpu_numa_id = _parse_int(_read_text(numa_path))
        target_numa = max(0, gpu_numa_id or 0)
        cpu_node = f"{host_id}/numa/{target_numa}"
        edges.append(
            TopologyEdge(
                edge_id=f"cpu-gpu:{cpu_node}:{node_id}",
                source=cpu_node,
                target=node_id,
                kind=EdgeKind.CPU_GPU,
                directed=False,
                full_duplex=True,
                sharing_group=f"pcie-{pci_bus.rsplit(':', 1)[0]}",
                contention_domain=f"pcie-{pci_bus.rsplit(':', 1)[0]}",
                health=HealthState.HEALTHY,
                facts=(
                    observed("pcie_generation", [(None, proc_prov)]),
                    observed("pcie_width", [(None, proc_prov)]),
                    observed("gpudirect_rdma", [(None, proc_prov)]),
                ),
            )
        )
        pci_device_path = Path("/sys/bus/pci/devices") / pci_bus.lower()
        generation = _pcie_generation(_read_text(pci_device_path / "current_link_speed"))
        width = _parse_int(_read_text(pci_device_path / "current_link_width"))
        root_id = f"{host_id}/pcie-root/{target_numa}"
        switch_domain = pci_bus.rsplit(":", 1)[0]
        switch_id = f"{host_id}/pcie-switch/{switch_domain}"
        if not any(item.node_id == root_id for item in nodes):
            nodes.append(
                TopologyNode(
                    node_id=root_id,
                    kind=NodeKind.PCIE_ROOT,
                    host_id=host_id,
                    health=HealthState.HEALTHY,
                    facts=(
                        observed("numa_node", [(target_numa, proc_prov)]),
                        observed("pcie_generation", [(generation, proc_prov)]),
                        observed("pcie_width", [(width, proc_prov)]),
                    ),
                )
            )
            edges.append(
                TopologyEdge(
                    edge_id=f"pcie:{cpu_node}:{root_id}",
                    source=cpu_node,
                    target=root_id,
                    kind=EdgeKind.PCIE,
                    directed=False,
                    full_duplex=True,
                    sharing_group=f"pcie-root-{target_numa}",
                    contention_domain=f"pcie-root-{target_numa}",
                    health=HealthState.HEALTHY,
                    facts=(
                        observed(
                            "theoretical_bandwidth", [(None, proc_prov)], unit="bytes_per_second"
                        ),
                    ),
                )
            )
        if not any(item.node_id == switch_id for item in nodes):
            nodes.append(
                TopologyNode(
                    node_id=switch_id,
                    kind=NodeKind.PCIE_SWITCH,
                    host_id=host_id,
                    health=HealthState.HEALTHY,
                    facts=(
                        observed("pci_bus_id", [(switch_domain, proc_prov)]),
                        observed("pcie_generation", [(generation, proc_prov)]),
                        observed("pcie_width", [(width, proc_prov)]),
                    ),
                )
            )
            edges.append(
                TopologyEdge(
                    edge_id=f"pcie:{root_id}:{switch_id}",
                    source=root_id,
                    target=switch_id,
                    kind=EdgeKind.PCIE,
                    directed=False,
                    full_duplex=True,
                    sharing_group=f"pcie-root-{target_numa}",
                    contention_domain=f"pcie-root-{target_numa}",
                    health=HealthState.HEALTHY,
                    facts=(
                        observed(
                            "theoretical_bandwidth", [(None, proc_prov)], unit="bytes_per_second"
                        ),
                    ),
                )
            )
        edges.append(
            TopologyEdge(
                edge_id=f"pcie:{switch_id}:{node_id}",
                source=switch_id,
                target=node_id,
                kind=EdgeKind.PCIE,
                directed=False,
                full_duplex=True,
                sharing_group=f"pcie-{switch_domain}",
                contention_domain=f"pcie-{switch_domain}",
                health=HealthState.HEALTHY,
                facts=(
                    observed("pcie_generation", [(generation, proc_prov)]),
                    observed("pcie_width", [(width, proc_prov)]),
                    observed("theoretical_bandwidth", [(None, proc_prov)], unit="bytes_per_second"),
                ),
            )
        )

    edges.extend(_gpu_topology_edges(host_id, gpu_rows, captured_at))

    network_prov = _provenance("socket/sysfs-network", "sysfs", captured_at)
    ib_devices = {candidate.name for candidate in Path("/sys/class/infiniband").glob("*")}
    for _, ifname in socket.if_nameindex():
        if ifname == "lo":
            continue
        node_id = f"{host_id}/nic/{ifname}"
        speed_mbps = _parse_int(_read_text(Path("/sys/class/net") / ifname / "speed"))
        carrier = _parse_int(_read_text(Path("/sys/class/net") / ifname / "carrier"))
        nic_device = Path("/sys/class/net") / ifname / "device"
        nic_pci_bus = nic_device.resolve().name if nic_device.exists() else None
        nic_numa = _parse_int(_read_text(nic_device / "numa_node"))
        transport = "infiniband" if ifname in ib_devices or ifname.startswith("ib") else "ethernet"
        nodes.append(
            TopologyNode(
                node_id=node_id,
                kind=NodeKind.NIC,
                host_id=host_id,
                health=HealthState.HEALTHY if carrier == 1 else HealthState.UNKNOWN,
                facts=(
                    observed("interface_name", [(ifname, network_prov)]),
                    observed("pci_bus_id", [(nic_pci_bus, network_prov)]),
                    observed("numa_node", [(nic_numa, network_prov)]),
                    observed(
                        "link_speed", [(speed_mbps, network_prov)], unit="megabits_per_second"
                    ),
                    observed("transport", [(transport, network_prov)]),
                    observed(
                        "active_port",
                        [(carrier == 1 if carrier is not None else None, network_prov)],
                    ),
                    observed("roce_capable", [(None, network_prov)]),
                    observed(
                        "rdma_capable",
                        [(True if transport == "infiniband" else None, network_prov)],
                    ),
                ),
            )
        )
        rail_id = f"{host_id}/rail/{ifname}"
        nodes.append(
            TopologyNode(
                node_id=rail_id,
                kind=NodeKind.NETWORK_RAIL,
                host_id=host_id,
                health=HealthState.HEALTHY if carrier == 1 else HealthState.UNKNOWN,
                facts=(
                    observed("name", [(ifname, network_prov)]),
                    observed("transport", [(transport, network_prov)]),
                ),
            )
        )
        edges.extend(
            (
                TopologyEdge(
                    edge_id=f"contains:{host_id}:{node_id}",
                    source=host_id,
                    target=node_id,
                    kind=EdgeKind.CONTAINS,
                    directed=True,
                    health=HealthState.HEALTHY,
                ),
                TopologyEdge(
                    edge_id=f"nic-network:{node_id}:{rail_id}",
                    source=node_id,
                    target=rail_id,
                    kind=EdgeKind.NIC_NETWORK,
                    directed=False,
                    full_duplex=True,
                    sharing_group=ifname,
                    contention_domain=ifname,
                    health=HealthState.HEALTHY if carrier == 1 else HealthState.UNKNOWN,
                    facts=(
                        observed(
                            "theoretical_bandwidth",
                            [
                                (
                                    speed_mbps * 125_000 if speed_mbps is not None else None,
                                    network_prov,
                                )
                            ],
                            unit="bytes_per_second",
                        ),
                        observed(
                            "measured_bandwidth", [(None, network_prov)], unit="bytes_per_second"
                        ),
                        observed("latency", [(None, network_prov)], unit="microseconds"),
                    ),
                ),
            )
        )

    gpu_numa_by_id = {
        edge.target: _parse_int(edge.source.rsplit("/", 1)[-1])
        for edge in edges
        if edge.kind is EdgeKind.CPU_GPU
    }
    nic_nodes = [node for node in nodes if node.kind is NodeKind.NIC]
    for gpu_id, gpu_numa in gpu_numa_by_id.items():
        if gpu_numa is None:
            continue
        for nic in nic_nodes:
            nic_numa_fact = nic.fact("numa_node")
            if (
                nic_numa_fact is None
                or nic_numa_fact.state is not FactState.KNOWN
                or nic_numa_fact.value != gpu_numa
            ):
                continue
            edges.append(
                TopologyEdge(
                    edge_id=f"gpu-nic:{gpu_id}:{nic.node_id}",
                    source=gpu_id,
                    target=nic.node_id,
                    kind=EdgeKind.GPU_NIC,
                    directed=False,
                    full_duplex=True,
                    sharing_group=f"numa-{gpu_numa}",
                    contention_domain=f"numa-pcie-{gpu_numa}",
                    health=HealthState.HEALTHY,
                    facts=(
                        observed(
                            "measured_bandwidth", [(None, network_prov)], unit="bytes_per_second"
                        ),
                        observed("latency", [(None, network_prov)], unit="microseconds"),
                        observed("gpudirect_rdma", [(None, network_prov)]),
                    ),
                )
            )

    in_container, restrictions, host_visible = _container_state()
    visibility_prov = _provenance("process-environment", "environment", captured_at)
    visibility = Visibility(
        in_container=in_container,
        host_devices_visible=host_visible,
        visible_gpu_ids=tuple(visible_gpu_ids),
        restrictions=restrictions,
        facts=(
            observed("container_visible_topology", [(in_container, visibility_prov)]),
            observed("host_device_inventory_complete", [(host_visible, visibility_prov)]),
        ),
    )
    warnings: list[str] = []
    if not gpu_rows:
        warnings.append("No NVIDIA GPUs were visible; no GPU capabilities were inferred.")
    if in_container and host_visible is None:
        warnings.append("The host-visible topology is unknown from this container.")
    if not ib_devices:
        warnings.append("No InfiniBand device was visible; RDMA capability remains unknown.")
    return finalize_graph(
        schema_version="sloforge.fabric.topology/v1",
        topology_id=host_id,
        captured_at=captured_at,
        nodes=tuple(nodes),
        edges=tuple(edges),
        visibility=visibility,
        software=_software(captured_at),
        warnings=tuple(warnings),
    )


def discover_topology(*, topology_id: str | None = None) -> CanonicalTopologyGraph:
    """Discover and finalize the current host into the canonical Fabric IR."""
    from sloforge.fabric.topology.conversion import to_canonical_topology

    return to_canonical_topology(discover_topology_records(topology_id=topology_id))


def save_topology(path: Path, graph: CanonicalTopologyGraph) -> None:
    save_topology_graph(path, graph)


def load_topology(path: Path) -> CanonicalTopologyGraph:
    return load_topology_graph(path)


def save_discovery_records(path: Path, graph: DiscoveryTopologyGraph) -> None:
    write_json(path, graph.model_dump(mode="json"))


def load_discovery_records(path: Path) -> DiscoveryTopologyGraph:
    return DiscoveryTopologyGraph.model_validate_json(path.read_text(encoding="utf-8"))
