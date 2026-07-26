# SLOForge Helix BranchFabric Characterization Report

## Evidence scope

- Run ID: `a53a739f0488359750d4f70f48f8753a`
- Combined canonical trace corpus: `90ae8353fe3010f0eb006e138ce99fd2ae6c6d60483082a87bddfbe1ae76549e`
- Verified evidence artifacts: 8
- Workload evidence: deterministic synthetic coding-agent fixture
- Timing evidence: measured host CPU monotonic/process clocks where recorded
- GPU status: unavailable; no compatible NVIDIA/CUDA Helix GPU was recorded

Controlled matrix points are design sweeps, not empirical production distributions. Every number below remains linked to its immutable source artifact in the companion requirements JSON.

## Measured branch state

| Characteristic | Result |
|---|---|
| Branch fanout | p50 4 count; p95 UNKNOWN; p99 UNKNOWN; max 4 count |
| Shared root bytes per fork | p50 2213 bytes; p95 UNKNOWN; p99 UNKNOWN; max 2213 bytes |
| Private suffix bytes per fork | p50 455 bytes; p95 UNKNOWN; p99 UNKNOWN; max 455 bytes |
| Branch lifetime | p50 422.585 milliseconds; p95 UNKNOWN; p99 UNKNOWN; max 423.181 milliseconds |
| Divergence rate | UNKNOWN |

## Copy-on-write and metadata

- Physical model-state amplification: p50 1 ratio; p95 UNKNOWN; p99 UNKNOWN; max 1 ratio.
- Hardware page-fault rate: UNKNOWN.
- Page-size decision: UNKNOWN. The trace records logical whole-file and logical-page COW, not a GPU/OS page-fault sweep.
- Current-software metadata throughput summaries: UNKNOWN.
- Outstanding metadata queue depth: UNKNOWN.

## Transform, transport, and transactions

- Explicit transform sequences retained: 1.
- STATE_SEND payload bytes: p50 1769.5 bytes; p95 UNKNOWN; p99 UNKNOWN; max 3444 bytes.
- Observed send fanout: p50 1 count; p95 UNKNOWN; p99 UNKNOWN; max 1 count.
- Multicast opportunity bytes: UNKNOWN.
- Commit rate in the bounded Continuum trace: p50 10.0244 events_per_second; p95 UNKNOWN; p99 UNKNOWN; max 10.0244 events_per_second.
- Abort rate in the same observation window: p50 0 events_per_second; p95 UNKNOWN; p99 UNKNOWN; max 0 events_per_second.

## Amdahl bounds

These are independent, non-additive upper bounds from directly recorded spans. They are not hardware speedup forecasts; simulated transport timing remains labeled.

| Operation | Maximum speedup if recorded primitive becomes free |
|---|---|
| none | UNKNOWN |

## Hardware conclusions supported by this run

- REQUIRED/HIGH_VALUE/OPTIONAL ISA recommendations: 0.
- SOFTWARE_ONLY classifications: 6.
- NOT_JUSTIFIED for the current bounded corpus: 10.
- Unresolved pending stronger measurements: 14.

No hardware primitive is promoted from this run alone. Unobserved quantization, compression, encryption, and multicast operations are negative findings for this bounded corpus; observed operations remain unresolved until target sensitivity, Amdahl leverage, and strongest-software-baseline evidence are all available.

## Queue, memory, bandwidth, and latency requirements

Observed concurrency establishes only lower bounds for queue depth. Recommended and pathological depths remain UNKNOWN. The controlled matrix covers 8--128 branches, but does not execute those fanouts, so capacity fields remain UNKNOWN rather than extrapolated. GPU interface targets are UNAVAILABLE on this host; all other bandwidth and maximum tolerable latency targets remain UNKNOWN without sensitivity experiments.

## Limitations

- The workload corpus is a deterministic synthetic CPU coding-agent fixture, not a production distribution.
- Model-state byte sizes are simulated; environment and host-operation timings are measured on the recorded CPU host.
- No compatible NVIDIA GPU, HBM, PCIe GPU link, multi-GPU fabric, or multi-node network was measured.
- Unknown values are not replaced with matrix sweep points, analytical projections, zeroes, or targets.
- NOT_JUSTIFIED ISA labels mean absent from this bounded corpus and must be revisited with broader real workloads.
- Metadata microbenchmark throughput is service capacity and is not reported as workload operations-per-second demand.

## Evidence artifacts

| Included in trace corpus | SHA-256 | Artifact |
|---|---|---|
| yes | `aa8c884f7d1e53ef824ceefc74528ea058c02de31ea6ab86db0d1271bd80fa7b` | `attempts/continuum_state/attempt-000/continuum-trace/state-operation-trace-v1.jsonl` |
| no | `6dc8a14d822287f9ac7e06f86737db62860d60d53ee7b29749180a91bfdd91c4` | `attempts/metadata/attempt-000/metadata-study.json` |
| yes | `1647d7ecad74914851b864a770633334c9c526066ed19fe0f5c19e8c53446dfc` | `attempts/vertical_trace/attempt-000/vertical-trace/branch-workload-trace-v1.jsonl` |
| no | `e63b8e1f6ef2a57999fffba6a5437b5131ae9008bf738cd50f119199ed34dd71` | `attempts/vertical_trace/attempt-000/vertical-trace/sharing-analysis.json` |
| yes | `fb2be4a476d0508e737f2ed568d4e4cd06ed64d5289d4ba2f16a18f86598f4eb` | `attempts/vertical_trace/attempt-000/vertical-trace/state-operation-trace-v1.jsonl` |
| no | `001bd96a163d7bfe51141c1885a9af56bae96e0a352dbe97516ff551761c8632` | `inputs/characterization.yaml` |
| no | `c7b2c3b6aee95211385a5e62a6d0a98ea5c0b39aaabfb7750046cc51fe72f238` | `inputs/hardware-capability.json` |
| no | `8985e4f61a072d8d4d31275b9f52e831ff69b22dbd3a02aa5248b6da2bb88a7e` | `run-manifest.json` |
