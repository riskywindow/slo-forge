# BranchFabric execution hardware adversarial review

Review status: **NO-BUILD AFFIRMED; FINAL GATE TRACEABILITY FINDING OPEN**

This is an independent combined GPU-performance, networking, FPGA/DPU
architecture, adversarial hardware-justification, and Modal/Baseten hiring
review. It reviewed the current main workspace at commit
`9c18719b78336e415fafc5b2da30dab5c9e59c81`. The gate result had an uncommitted
wording-only change at review time, so its observed SHA is a workspace snapshot,
not a committed identity. No source, raw measurement, gate input, gate logic,
gate result, or existing report was changed by this reviewer.

## Executive verdict

No BranchFabric hardware primitive is justified. The only evidence-backed
placement is CPU software.

The new fanout study is useful because it executes 8-, 16-, and 32-way branch
groups for two controlled reference workloads, three seeds, and matched
software implementations. Its strongest result is a software result:
shared-root lazy COW is 102.28x to 237.71x faster at branch readiness than
eagerly copying every serialized reference-state segment. After applying that
software baseline, readiness is only 0.0020% to 0.0060% of the measured local
branch lifecycle. Making it free would improve that local lifecycle by at most
1.000060x. The measured state is a deterministic CPU reference fixture, not a
real transformer allocation, RSS, GPU memory, HBM, or CXL memory.

The new reclamation campaign does execute Continuum fork, pause, incremental
checkpoint, layout conversion, local-file transfer, ownership commit, resume,
stale-source rejection, and corruption rejection in one local transaction. It
does not execute production serving traffic or reclaim a GPU. The serving spike
is a constructed Helix scheduler request whose evidence references use all-zero
hash placeholders; `slo_restored` is read from the compiled plan. Physical GPU
capacity and GPU bytes reclaimed are exactly zero in every trial. The optimized
four-worker software path reduces median interruption by 12.29% and empirical
p99 interruption by 11.43%, below the corresponding hardware headroom gate and
without a hardware prediction or confidence bound.

No CUDA GPU, multi-GPU fabric, PCIe GPU path, NVLink, RDMA, physical test NIC,
FPGA, DPU, accelerator toolchain, or target interface was measured. Every
BranchFabric authorization variable is absent, both budgets are effectively
zero, and target-specific RTL, FPGA builds, external resources, and billable
GPU use are unauthorized. `hardware/branchfabric` and `sim/branchfabric` are
absent, as they should be.

## Evidence checked

| Evidence | SHA-256 | Independent result |
|---|---|---|
| `execution/fanout/raw-samples.jsonl` | `af616379666cc4d9043be355976016cc06a57a87b07caa8b60b2ad69d901b810` | 144 raw records; 108 measured |
| `execution/fanout/summary.json` | `51768b2b164caacb65d0de4ca3307737c4eda8c83cef030a89dd2597fcffebc4` | medians, matched speedups, and ideal-zero-cost bounds reproduce |
| `execution/fanout/fixture-evidence.jsonl` | `cdc1b580d2127d11e264f264406a08969bda7447ac6f15b49f77eb663a6d802c` | 6 reference fixtures |
| `execution/fanout/manifest.json` | `a55415259fd5ef81fecef2deff2d289e1daae2e2cb6ca4fd098658bee2708f9f` | all three member hashes and sizes match |
| `execution/reclamation/raw/trials.jsonl` | `5cd3dd64e0e0ff01286f8cb7b893ae0ac690c6a923a0e069102db8eb093a340d` | 36 trials; all declared transaction checks pass |
| `execution/reclamation/campaign.json` | `0ca238f362feef351679e8426d6a763300949f7202d738b913287931e1a453c6` | descriptive percentiles and software speedup reproduce |
| `execution/reclamation/environment.json` | `e39cfada6e71977c1926c89cc974eeee1611ba788d199c532d7059acc4966dbe` | CPU-reference only; no GPU or network fabric |
| `execution/reclamation/MANIFEST.json` | `c6a6357aaaabccde974ce07baad87f5911f83bdcc553679201342b7d28806ecf` | all three member hashes match |

Paths in this table are below `artifacts/branchfabric/`.

The focused acceptance command passed 14 tests:

```sh
uv run --locked pytest -q \
  tests/python/test_branchfabric_fanout_vertical.py \
  tests/python/test_branchfabric_reclamation.py \
  tests/python/test_branchfabric_gates.py
```

## GPU-performance assessment

- Available host evidence is Apple M4 Pro CPU timing. The integrated Apple GPU
  was not exercised, and it is not CUDA evidence.
- GPU authorization and GPU budget are absent. No GPU resource may be created
  or used for this phase.
- KV/HBM allocation, copy engines, launch overhead, GPU utilization,
  interference, PCIe, peer-to-peer, and NVLink are all unmeasured.
- The fanout fixture reports 532,200 to 1,030,592 serialized bytes per root, but
  those totals are deterministic content bytes, not device allocation.
- Reclamation moves only 54,496 to 54,528 reported plaintext bytes across eight
  local-file operations and reclaims zero GPU bytes. It cannot size or select a
  GPU copy/transform kernel.

GPU placement: **REJECTED_UNMEASURED_UNAUTHORIZED**.

## Networking and multicast assessment

The fanout experiment creates local Python branch views and performs no
network transfer. Its identical new-suffix opportunity is only 163 bytes for
the coding fixture and 4,259 bytes for the reasoning fixture. The shared root
is referenced in one process; it is not physically delivered to 8, 16, or 32
destinations.

The reclamation transaction invokes eight transfers to one shared local
`FileContentStore` through a local temporary-file spool. That is one host and
one destination store, not one-to-many delivery to workers or NIC endpoints.
No packet, wire-byte, route, congestion, retransmission, RDMA, NIC, or topology
measurement exists.

Multicast or DPU placement: **REJECTED_NO_PHYSICAL_MULTIDESTINATION_EVIDENCE**.

## FPGA/DPU architecture assessment

No credible target envelope can be derived. Byte rate, operation rate,
hardware-facing concurrency, queue depth, working set, latency target,
bandwidth target, fault interface, host/device interface, resource estimate,
clock target, and power target remain unknown. The local fanout queue is
serialized at concurrency one. Reclamation reaches host-thread concurrency
four and queue occupancy eight, but that describes the Python/local-filesystem
implementation, not an accelerator command queue.

The current local-transfer median is dominated by tiny-file persistence and
content-store overhead: a typical serial transfer is about 6.8 KiB and roughly
26 ms. Mapping that path to an FPGA or DPU would accelerate a test-harness
storage choice rather than a measured GPU/network bottleneck. A stronger future
software baseline should batch or avoid repeated per-chunk spool `fsync`, reuse
content-addressed chunks across the shared destination, and then measure a real
fabric.

FPGA placement: **REJECTED_UNAUTHORIZED_UNSIZED**.

DPU/SmartNIC placement: **REJECTED_UNAUTHORIZED_NO_NETWORK_PATH**.

CXL/HBM shared-root placement: **REJECTED_NO_PHYSICAL_ALLOCATION_PRESSURE**.

CPU shared-root/lazy-COW software: **SELECTED_FOR_CURRENT_EVIDENCE**.

## Gate assessment

The observed preliminary gate result remains fail-closed:

- outcome: `FAIL_NO_BUILD`
- passing candidates: none
- hardware implementation allowed: false
- functional model or cycle simulator allowed: false
- required action: `TERMINATE_HARDWARE_PATH`

That disposition is correct. The new CPU-reference artifacts cannot satisfy
the mandatory real-evidence or platform-feasibility gates. They also provide no
hardware service curve and no confidence-backed system-level hardware
headroom. High software speedup against eager copying is not hardware headroom.

The gate snapshot is still marked `preliminary`, and neither new evidence path
appears in its gate input or result. Before final no-build publication, the
final gate artifact must cite and classify both new campaigns, preserve their
non-eligibility, and regenerate the fail-closed result. This is traceability,
not a reason to change the outcome.

## Findings

### High severity

#### BF-HA-H1 — Mandatory real-evidence and platform gates remain impossible

All accelerator authorizations are absent, both budgets are zero by policy,
and no compatible target hardware or toolchain was measured. There is no legal
or technical route to a target-specific implementation. The required action is
no-build.

#### BF-HA-H2 — High fanout exposes a solved software problem, not residual hardware value

The shared-root lazy-COW baseline removes 88.89%, 94.12%, and 96.97% of the
fixture's eager at-ready serialized-byte accounting at fanouts 8, 16, and 32.
Its readiness speedup is 102.28x to 237.71x. Yet optimized readiness contributes
at most 0.006006% of the local lifecycle, yielding a maximum ideal-free bound
of 1.000060x. There is no evidence that software COW is insufficient or that
the reported serialized savings release GPU memory or increase trajectories.

#### BF-HA-H3 — The reclamation campaign is not the required physical serving/GPU transaction

The state lifecycle is causally executed, but the traffic spike is a synthetic
scheduler input, its scheduler evidence hashes are zeros, SLO restoration is a
plan field rather than an observed serving latency, and reclaimed GPU capacity
is zero. The campaign is valuable integration evidence but cannot satisfy the
real-workload, actual-GPU, or end-to-end Helix hardware gates.

#### BF-HA-H4 — The current gate snapshot omits the new campaigns

Neither `execution/fanout` nor `execution/reclamation` is referenced by the
preliminary gate input/result. A final result must bind them as CPU-reference,
hardware-ineligible evidence. Leaving them outside the final gate would violate
the requirement that every gate reference the raw evidence, even though their
inclusion cannot produce a PASS.

### Medium severity

#### BF-HA-M1 — Measurement-producer revision is not internally reproducible

The fanout summary calls baseline commit `46955be...` its
`repository_revision`, but `fanout_vertical.py` does not exist at that commit.
The reclamation artifacts also record the semantic baseline commit without a
separate producer revision or dirty-state policy; `reclamation.py` likewise
does not exist at the recorded baseline. Publication commits contain both code
and artifacts, which helps, but the artifact manifests do not distinguish
semantic baseline, generator revision, and publication revision.

#### BF-HA-M2 — Reclamation transport is not the strongest relevant transfer baseline

The optimized path changes worker count from one to four while preserving a
local-file transport that creates, writes, `fsync`s, renames, rereads, hashes,
and publishes every small chunk. Eight branches target one content store.
Per-operation transfer durations increase under parallel contention while the
migration wall time improves through overlap. Batch transfer,
content-address-aware reuse, and persistence amortization remain untested, so
this path cannot be used as a hardware comparison baseline.

#### BF-HA-M3 — Reclamation tails and effect uncertainty are descriptive only

There are 18 observations per baseline, formed from three seeds, two
repetitions, and three tracing levels. The campaign pools tracing levels,
reports p99 as the maximum, and provides no paired effect interval or lower
confidence bound for the 1.1401x median interruption speedup. Inclusive
operation fractions sum overlapping thread time and COW/append observations;
they are not critical-path fractions. They must not be inserted into a hardware
gate as confidence-backed headroom.

### Validated behavior

- Raw hashes, manifest sizes, trial counts, matched software effects, fanout
  headroom arithmetic, reclamation percentiles, and transaction checks
  reproduce.
- Three seeds, randomized order, raw samples, bounded queues, and tracing
  controls are present.
- Every reclamation trial restores its modeled SLO, rejects stale-source and
  corrupted-state faults, completes its transactions, and reports no hidden
  fallback.
- The artifacts explicitly label CPU-reference/synthetic scope and make no
  hardware claim.
- No target-specific RTL or fake accelerator directory was created.

## Required adversarial answer

**Would this hardware have been selected from the measured workload if the
BranchFabric name and prior architecture proposal did not already exist?**

**No.** A neutral architect would select the measured shared-root/lazy-COW and
bounded-parallel CPU software, then request real model, GPU, physical-network,
and serving-transaction measurements. They would not select an FPGA, DPU,
multicast engine, GPU kernel, metadata unit, or CXL/HBM store from these data.

## Modal/Baseten hiring answer

**Does this demonstrate real hardware-software co-design, or is it a synthetic
accel attached to a software project without end-to-end relevance?**

It does **not yet demonstrate real hardware-software co-design**, because no
production inference state, physical GPU/fabric behavior, target architecture,
or end-to-end accelerator benefit was measured. It is also **not a synthetic
accelerator attached to the project**: the gate correctly prevented one from
being built. The work demonstrates strong negative-result discipline,
transactional software integration, and measurement engineering. That is a
positive systems-hiring signal, but it would not meet a hardware co-design bar
if represented as a completed accelerator result.

## Final reviewer decision

`REJECT_ALL_BRANCHFABRIC_HARDWARE; RETAIN_OPTIMIZED_CPU_SOFTWARE; PUBLISH_VERIFIED_NO_BUILD`

