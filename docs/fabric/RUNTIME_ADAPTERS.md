# Physical runtime adapters

Fabric adapters lower an already validated physical plan into offline artifacts.
They do not deploy, mutate a cluster, or claim that metadata is enforced when a
target has no corresponding scheduling primitive.

## Common contract

`FabricAdapterContext` binds plan, topology, model/revision, image, runtime and
version, namespace, resource names, resource bounds, gang scheduler, and cloud
metadata policy. Validation confirms every rank's host, GPU, NUMA, NIC, and rail
exists in the supplied topology. Runtime versions fail closed outside validated
ranges.

All exporters write canonical `physical-plan.json`, `topology.json`, SHA-256
artifact records, capability flags, and validation results. The result always
sets `deployed: false`.

## Targets

- Local emits bounded launch groups, explicit CPU affinity, expected GPU UUID,
  NIC interface, environment, startup timeout, shutdown grace, and queue bound.
- Docker Compose emits GPU device IDs, CPU/memory/PID limits, init, health and
  shutdown behavior, read-only mounts, dropped capabilities, and no-new-privileges.
- Kubernetes emits node affinity, anti-affinity, topology spread, requested GPU
  and optional RDMA extended resources, probes, zero-unavailable rolling update,
  physical-plan annotations, and a fail-on-mismatch binding policy. Exact GPU UUID
  selection still requires a compatible device plugin/runtime.
- NVIDIA Dynamo emits an offline `DynamoGraphDeployment` for a validated vLLM or
  SGLang backend. The tagged v1.2-v1.3 contract is
  `nvidia.com/v1beta1`, with `spec.components`, a `podTemplate` container named
  `main`, and Grove/LWS multi-node metadata. The operator, not SLOForge, injects
  worker probes on the `system` port (9090). SLOForge compiles metadata and
  components; Dynamo remains the runtime mechanism.
- Direct vLLM process groups use DP=1. Expert parallelism is emitted only when
  physical EP equals `TP_SIZE * rendered DP_SIZE`; direct prefill/decode uses
  native `--kv-transfer-config` with the reviewed `NixlConnector`. The Dynamo
  wrapper alone owns `--disaggregation-mode`.
- Direct SGLang process groups use canonical TP/PP/DP/EP flags and explicit
  disaggregation backend/IB arguments. Expert parallelism never implicitly
  enables DP attention.
- Modal and Truss preserve their existing exporters. Physical data is advisory
  metadata only and requires `allow_advisory_cloud_metadata=true`; it is never
  presented as enforced rank placement.

## Validated API surface

`deploy/fabric/validated-versions.json` records the exact offline validation date,
version ranges, fields, and official documentation sources. The reviewed ranges
are Kubernetes `>=1.31,<1.37` (reference documentation 1.36), vLLM
`>=0.26.0,<0.27.0` (locked 0.26.0), SGLang `>=0.5.2,<0.6.0` (locked 0.5.2;
0.5.15.post1 also reviewed), and NVIDIA Dynamo `>=1.2.0,<1.4.0` (tagged 1.3
schema). Modal 1.5.3 and Truss 0.18.24 were locally inspected. This is
compatibility evidence for generated artifacts, not vendor endorsement or live
runtime validation.

Direct vLLM and SGLang lowerings reject any replica that spans hosts because
they do not synthesize rendezvous. Dynamo DGD is the supported multi-node engine
path. Generic Kubernetes export also rejects multi-node plans: that renderer
does not emit a PodGroup, LeaderWorkerSet, or other atomic gang contract.
Role-local independent SGLang replicas through Dynamo are rejected rather than
conflating SGLang DP/DP-attention with physical replicas. The vLLM transfer
adapter supports only the reviewed NIXL connector; it does not infer another
connector.

## Safety

Adapters reject unsupported gang scheduling, unenforceable cloud placement,
invalid disaggregation roles, missing RDMA resources, and incompatible runtime
versions. `runtimeVersionOverride` is deliberately absent from the DGD subset
because it is not in the tagged 1.3 API. The subset was schema-validated offline,
not admitted by a live operator. No adapter creates paid resources. External
deployment mutation is a separate explicitly authorized operation controlled by
SLOForge's budget and mutation environment variables.
