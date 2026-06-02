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
  SGLang backend. SLOForge compiles metadata and components; Dynamo remains the
  runtime mechanism.
- vLLM lowering uses TP/PP/DP, expert-parallel, and NIXL connector flags only in
  its validated version range.
- SGLang lowering uses TP/PP/DP/EP and disaggregation role flags only in its
  validated version range.
- Modal and Truss preserve their existing exporters. Physical data is advisory
  metadata only and requires `allow_advisory_cloud_metadata=true`; it is never
  presented as enforced rank placement.

## Validated API surface

`deploy/fabric/validated-versions.json` records the exact offline validation date,
version ranges, fields, and official documentation sources. At the time recorded
there it covers Kubernetes 1.31+, vLLM 0.12+, SGLang 0.5+, NVIDIA Dynamo 1.x,
installed Modal 1.5.3, and installed Truss 0.18.24. This is compatibility evidence
for generated artifacts, not a statement of vendor endorsement.

## Safety

Adapters reject unsupported gang scheduling, unenforceable cloud placement,
invalid disaggregation roles, missing RDMA resources, and incompatible runtime
versions. No adapter creates paid resources. External deployment mutation is a
separate explicitly authorized operation controlled by SLOForge's budget and
mutation environment variables.

