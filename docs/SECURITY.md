# Security model

SLOForge is infrastructure software that executes profilers and emits deployment code. Its principal risks are untrusted specifications, network/backend behavior, model supply chain, resource exhaustion and accidental paid deployment. The local demo is not an internet-facing security boundary.

## Trust boundaries

| Boundary | Trusted input | Untrusted or failure-prone input |
|---|---|---|
| Compiler | repository schemas and code | user IR, trace, profile and extension values |
| Profiler | explicit engine/device selection | packages, model repositories, engine subprocesses |
| Simulator | validated scenario schema | request/fault volume and durations |
| Gateway | static operator config | client JSON, headers, slow clients, backend SSE |
| Exporters | validated plan | target strings and generated runtime dependencies |
| Reports | indexed paths plus hashes | stale/tampered artifacts |

## Implemented controls

- Strict models reject unknown fields, non-finite numbers and invalid cross-field combinations.
- Namespaced extension keys reduce accidental collision; extension values remain plain JSON and are never evaluated.
- Canonical SHA-256 and atomic writes protect evidence integrity against accidental mutation.
- Request bodies, SSE events, admission, backend concurrency, traces, simulator events and subprocess time are bounded.
- Backend URLs come from static configuration and must be HTTP(S); request data cannot choose an arbitrary upstream URL.
- SLO-only metadata is removed before forwarding to engines.
- Retry is forbidden after stream output begins.
- Client drop releases owned permits and cancels ongoing stream work.
- Health checks, timeouts and circuit breakers contain failing backends.
- Exporters refuse non-empty output directories and never invoke cloud deployment.
- Paid cloud work requires an explicit external command and `SLOFORGE_GPU_BUDGET_USD`; normal generation is offline.
- Environment evidence uses an allowlist and hashes host identity in the canonical bundle.

## Deployment requirements

The gateway intentionally does not implement authentication, tenant billing or TLS. Put it behind a trusted ingress/service mesh that provides TLS, authentication, request-size policy, network ACLs and rate limits. Bind health, metrics and mock-admin endpoints to private interfaces. Never ship the mock backend's `/admin/fault` endpoint in a public production service.

Backend URLs are operator-controlled but still create an SSRF-capable configuration surface. Validate plans in a trusted pipeline, restrict egress and disallow link-local/cloud-metadata ranges at the network layer. Do not compile or serve arbitrary user-submitted plan files.

## Model and dependency supply chain

Model IDs are paired with revision, checksum and license metadata in the IR, but runtime adapters must actually verify downloaded content against the recorded digest. A mutable model tag is insufficient. Private repository tokens belong in provider secret stores, never plans, generated YAML or logs.

Python and Rust dependencies are lock-file pinned. Generated Modal/Truss images use pinned package versions, but an operator should scan image layers, review transitive licenses and signatures, and rebuild when security advisories land. Generated deployment code is source and must be reviewed before execution.

## Data handling

Prompts and outputs may contain sensitive data. Current traces can store request token counts, classes and IDs; mock gateway responses are synthetic. Production adapters should redact prompt text, hash or pseudonymize tenant/request identifiers, minimize retained raw output and apply retention policy to profiles/traces. Never place prompt, request ID or tenant in Prometheus labels.

Trace context from a client is validated structurally but remains untrusted correlation data. Do not use it for authorization. Structured backend errors are truncated, yet deployments should review whether upstream messages can reveal paths or package details.

## Denial of service

Global admission protects in-process work but a reverse proxy is still required for connection limits, header/body deadlines and per-principal quotas. Slow consumers exert backpressure and hold capacity until timeout; choose `request_timeout_ms`, queue timeout and capacity for the deployment. SSE event bounds prevent unbounded parser memory but do not replace bandwidth controls.

The simulator's maximum event count bounds hostile scenario expansion. Profile duration/cost limits and subprocess timeouts bound experiments. GPU OOM remains possible because static feasibility is conservative estimation, not formal proof.

## Cloud cost safety

Generation and static validation must never create resources. An operator-run cloud job should parse `SLOFORGE_GPU_BUDGET_USD` as a hard total ceiling, reserve at least 15%, estimate every job, write a spend ledger, terminate idle instances and stop before projected spend exceeds the remaining budget. The current repository does not autonomously provision Modal or Baseten resources.

## Reporting vulnerabilities

Report suspected vulnerabilities privately to the repository maintainers rather than opening a public issue containing exploit details or secrets. Include affected revision, reproduction with synthetic data and impact. Rotate any credential that appears in an artifact; removing it from Git history does not invalidate the secret.
