# Fabric security and operational safety

Fabric processes hardware identity, subprocess output, model metadata, traces,
deployment artifacts, and recovery actions. The primary risks are malicious
inputs, command/path injection, secret leakage, excessive resource use, unsafe
host mutation, and an incorrect recovery action.

## Trust boundaries

- JSON/YAML/model configs, topology fixtures, profiles, traces, and runtime
  output are untrusted at ingestion.
- Hardware/vendor commands are local executable dependencies, not trusted data.
- Generated Kubernetes/Dynamo/Modal/Truss files cross into external control
  planes only when an operator invokes a separate deployment action.
- Autopsy evidence can contain workload/request metadata and must be access-
  controlled like production telemetry.

## Implemented controls

Strict Pydantic and Rust `deny_unknown_fields` models reject ambiguous input.
Canonical hashes bind plans and reports to evidence. Graphs reject duplicate,
missing, cyclic, and self references. Relative artifact paths are checked against
root escape. WarmPath verifies size and SHA-256 before and after atomic writes.

Subprocess invocations use argument arrays, no shell interpolation, stdin closed
where appropriate, timeouts, byte limits, bounded operation/event/trial counts,
process groups, and cleanup. ForgeCI stores stdout/stderr separately and redacts
sensitive environment values. Generated Docker and Kubernetes artifacts drop
capabilities, avoid service-account tokens, use read-only mounts where possible,
declare resource limits, and provide graceful termination.

Discovery is read-only and records unknown capability. It never changes clocks,
networking, or GPU state. Privileged probes, network faults, GPU clock changes,
and external deployment mutation use separate false-by-default opt-ins:

```text
SLOFORGE_ALLOW_PRIVILEGED_PROBES
SLOFORGE_ALLOW_NETWORK_FAULTS
SLOFORGE_ALLOW_GPU_CLOCK_CHANGES
SLOFORGE_ALLOW_EXTERNAL_DEPLOYMENT_MUTATION
```

Paid resources additionally require `SLOFORGE_GPU_BUDGET_USD`. Offline exporters
always return `deployed: false`.

Recovery requires simulation, confidence, shadow, canary, promotion criteria,
bounded drain, and rollback. Started streams are preserved and not retried after
output begins. External actions require authorization in both plan and executor.

## Deployment guidance

Do not embed secrets in a benchmark matrix or generated environment metadata.
Use runtime secret stores and mark ForgeCI variables sensitive. Treat traces and
model paths as sensitive. Pin images by digest in production. Verify generated
YAML against cluster admission policy and keep the worker runtime responsible for
fail-on-mismatch GPU/NUMA/NIC binding.

Run privileged hardware tools in a constrained diagnostic environment. Never
enable host-wide traffic control or clock mutation on a shared production node.
If optional fault tooling creates namespaces or rules, register cleanup before
injection and validate cleanup afterward.

## Residual risks

The repository does not implement authentication, tenant isolation, artifact
encryption, signing, remote attestation, or a production deployment approver.
SHA-256 detects change but does not establish who produced an artifact. External
runtime commands and model files can execute code outside SLOForge's parser.
Physical metadata may reveal fleet topology. Counterfactual simulation can be
wrong under unmodeled behavior; recovery safety gates reduce but do not eliminate
that risk.

