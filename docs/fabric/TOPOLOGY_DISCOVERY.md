# Topology discovery

Fabric discovery builds an evidence graph, then normalizes it into the canonical
compiler graph. This two-stage design preserves contradictions that a single
"best value" inventory would hide.

## Current-host sources

`discover_topology_records` uses structured files and bounded vendor commands:

- Python/platform, `/proc`, `sysctl`, sysfs CPU/socket/NUMA/PCI/network facts,
  cgroup memory and device-visibility metadata;
- `nvidia-smi --query-gpu` for product, UUID, VRAM, PCI address, driver,
  compute capability, MIG, ECC, clocks, and power when available;
- `nvidia-smi topo -m` for GPU locality edges when available;
- Linux network interface/sysfs facts and bounded `ethtool`/InfiniBand metadata;
- installed package metadata for relevant runtime versions.

Command absence or a non-zero result yields an explicit unknown observation. It
does not create a synthetic GPU, NIC capability, or GPUDirect assertion. All
subprocesses have timeouts.

Each `ObservedFact` is `known`, `unknown`, or `conflict`. Every observation stores
source, source kind, capture time, confidence, and command. Conflicting values
remain unresolved until conversion policy can make a safe decision. Container
visibility and host visibility are separate facts; `CUDA_VISIBLE_DEVICES` and
`NVIDIA_VISIBLE_DEVICES` restrictions are recorded.

## Canonicalization

`to_canonical_topology` creates typed nodes and edges only from sufficient
evidence. Canonical edges retain theoretical and measured curves, sharing and
contention domains, health, directionality, confidence, timestamp, environment,
discovery source, and measurement source. The canonical hash is used by both
`FabricProfile` and `PhysicalExecutionPlan` to prevent stale-profile compilation.

## Deterministic fixtures

Fixture builders cover a single-GPU workstation, eight-GPU NVLink host,
multi-NUMA PCIe host, two-node InfiniBand cluster, RoCE cluster, MIG host,
degraded topology, container-limited visibility, and conflicting sources. They
are labeled `synthetic` and are used for CPU-only compiler and simulator tests.
The flagship topology in `artifacts/fabric-demo/topology.json` has two hosts,
eight synthetic GPUs per host, two NUMA domains, NVLink groups, PCIe switches,
NIC locality, and two rails.

## What discovery does not prove

A detected RDMA device is not proof that GPUDirect RDMA works end to end. An
NVLink topology edge is not a bandwidth measurement. Interface speed does not
establish usable application throughput. The benchmark pass supplies those
curves. DCGM, NVML library bindings, hwloc, ibverbs, NCCL topology dumps, and
Kubernetes device metadata are represented as provenance categories and adapter
inputs, but were not hardware-exercised on the current Apple Silicon machine.

Privileged inspection and host mutation are never enabled by discovery. Safe
feature detection is distinct from fault injection.

