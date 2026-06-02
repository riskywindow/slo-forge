# Fabric demo script

The extension demo is CPU-only, deterministic, and non-privileged. It simulates
hardware; it does not silently claim measured GPUs or networks.

## Run

```bash
make bootstrap
make fabric-check
make fabric-demo
```

The demo creates a two-host, 16-GPU synthetic topology with two NUMA domains per
host, NVLink groups, PCIe locality, NICs, and two rails. It profiles synthetic
fabric curves, builds a synthetic MoE model graph, compiles topology-aware and
topology-unaware physical plans, and replays 12 bursty requests spanning short
interactive, long-context, and batch classes with three priorities.

It injects a network resource-rate fault and a rank-specific GPU slowdown. The
degraded output is captured as canonical Autopsy evidence, compared with the
healthy run, diagnosed, and replayed under seven counterfactual repairs. Recovery
then exercises simulation validation, replacement build, shadow, canary,
promotion, started-stream-preserving drain, and completion.

## Inspect the evidence

```bash
jq '{healthy,degraded,restored,ground_truth_faults,diagnosis,
     counterfactuals_evaluated,selected_counterfactual,recovery_final_state}' \
  artifacts/fabric-demo/manifest.json

jq '.[] | {sequence,at_ms,event,detail,evidence_uri}' \
  artifacts/fabric-demo/timeline.json

jq '{top_hypothesis,top_three,confidence,warnings}' \
  artifacts/fabric-demo/autopsy/diagnosis.json
```

Open `reports/fabric-demo/fabric-demo.html` or read the Markdown report. Import
`artifacts/fabric-demo/traces/degraded.perfetto.json` into Perfetto/Chrome trace
viewer. Prometheus exposition is under `artifacts/fabric-demo/metrics/`, and the
OpenTelemetry-shaped trace is `artifacts/fabric-demo/traces/otel.json`.

The report is derived from the manifest and raw simulator artifacts. Validate its
integrity by recalculating each SHA-256 listed in the manifest.

## Component demos

```bash
make autopsy-demo
make forgeci-demo
make warmpath-demo
make extension-evaluation
```

Autopsy writes a separate scoped run. ForgeCI creates and bisects a local fixture
repository. WarmPath profiles deterministic local snapshot files, compiles tier
placement, simulates startup, and executes with checksum verification.

## Hardware-ready workflow

```bash
sloforge fabric discover --output artifacts/fabric/topology.json
sloforge fabric benchmark \
  --topology artifacts/fabric/topology.json \
  --suite host-memory --measured \
  --output artifacts/fabric/measurements
```

GPU/collective suites require an explicitly installed adapter and matching
hardware. If it is absent, stop at the unavailable result; do not reuse synthetic
curves as measurements. Preserve the discovered topology and raw profile with any
hardware-backed report.

