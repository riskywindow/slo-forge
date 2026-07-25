# BranchFabric Characterization Systems and Hardware Review

Review status: **BLOCKING FINDINGS**

Review scope: Helix architecture, Continuum state lifecycle, GPU performance,
networking, storage and memory, instrumentation overhead, FPGA placement, and
SmartNIC/DPU placement. This review is an evidence audit, not a hardware design
proposal. No raw artifact was modified.

## Executive verdict

The conservative hardware conclusion is supported: this corpus does not justify
an FPGA, SmartNIC/DPU, CXL engine, hardware metadata unit, multicast primitive,
or frozen BranchFabric ISA. GPU, multi-GPU, and multi-node measurements are
correctly marked unavailable in
`artifacts/branchfabric/hardware/status-seed-20260809.json`, the roofline report
does not invent missing ceilings, and the placement report refuses fabricated
numeric device scores.

The characterization completion claim is not yet supported. The CPU run
validates one synthetic coding-agent vertical slice plus a separate Continuum
fixture; it does not execute the declared experiment matrix, it does not contain
a single end-to-end flagship transaction, and its current Amdahl migration
bounds do not isolate the migration transaction. The instrumentation-overhead
result also does not time the actual full canonical trace path. These gaps must
be fixed or the final handoff must remain explicitly incomplete.

## Helix architecture review

### High severity

#### HELIX-H1: the experiment matrix is validated, not executed

`run_characterization` expands the matrix only to enforce the
`max_experiments` bound, then starts one fixed set of stages
(`python/sloforge/helix/characterization/workflow.py:145`). The CPU run manifest
records 855 cases, but the requirements compiler contains one coding-agent
workload and one observed branch group
(`artifacts/branchfabric/requirements/branchfabric_requirements.json`). The
reasoning, verification-heavy, capacity-reclamation, and highly-divergent
workload classes are controlled declarations, not executed trace corpora.

Impact: branch fanout, prefix/suffix, divergence, branch survival, pressure,
fault, migration, and physical-layout distributions required by the completion
contract do not exist as empirical sweeps. `matrix_case_count` must not be
reported or interpreted as an experiment count.

Required resolution: execute a bounded representative subset for every required
CPU workload class, preserve a result or explicit failure/skip artifact for each
selected cell, and bind those traces into requirements. If that work is deferred,
state that the characterization phase is incomplete.

#### HELIX-H2: the flagship is not the required complete transaction

The flagship artifact says `single_transaction: false`, `source_commit: null`,
and marks PRODUCTION, BRANCH_READY, and EVALUATE untraced; MIGRATE and RESUME are
from a separate Continuum lifecycle
(`artifacts/branchfabric/characterization/flagship/flagship-manifest.json:605`).
The Helix vertical run itself says migration is absent
(`artifacts/branchfabric/characterization/vertical-seed-41-final/vertical-run.json`).

Impact: the required production-through-promotion waterfall, capacity
reclamation, pause/migrate/resume behavior, and per-stage resource accounting do
not exist for one causally connected coding-agent transaction. The linked
manifest is honest, but it is not a completed flagship run.

Required resolution: trace the missing boundaries and one causally connected
pause/checkpoint/migrate/resume path, or leave the flagship and completion status
explicitly incomplete. Do not splice the independent traces into a false
critical path.

#### HELIX-H3: measurement producer source identity is not reproducible

The trace manifest has no measurement-source commit and embeds the baseline
software manifest whose `source_commit` is the pre-instrumentation baseline
`d6f77c8...`; the flagship source commit is null. The overhead artifact records
`f1ebbf6...`, showing that an exact producer revision was available, but it is
not bound into the vertical/Continuum trace manifests.

Impact: a consumer cannot identify the instrumentation code that produced a
trace. Baseline commit, measured Helix semantic commit, and trace-producer commit
are different concepts and must be recorded separately.

Required resolution: add and verify a producer source revision (and dirty-state
policy) in every trace/run manifest and flagship manifest. Regenerate the corpus
after the schema-compatible fix.

### Medium severity

#### HELIX-M1: generic model state is omitted from `model_state_bytes`

The state trace records the generic `model` segment, but `_MODEL_SEGMENTS` omits
`StateSegment.MODEL` (`python/sloforge/helix/characterization/workload_analysis.py:770`).
Consequently the workload analysis reports 13,300 generic-model bytes while its
`model_state_bytes` summary is zero. The design brief discloses this at
`docs/branchfabric/CHARACTERIZATION_DESIGN_BRIEF.md:31`, but disclosure does not
make the aggregate semantically correct.

Required resolution: include generic model bytes in a clearly named aggregate,
while keeping unavailable KV/recurrent/sampler decomposition separate and
unknown.

#### HELIX-M2: bottleneck ranking cannot identify the full transaction bottleneck

The seed-41 stage unions total about 408 ms, while about 10.34 s is arithmetically
unattributed (`artifacts/branchfabric/analysis/bottleneck-decomposition.json`).
The artifact correctly warns that inclusive stage unions are not an exclusive
critical path, but the design brief still treats reward as dominating the
observed lifecycle and infers state metadata/COW is secondary
(`docs/branchfabric/CHARACTERIZATION_DESIGN_BRIEF.md:95`).

Required resolution: retain “reward is the largest traced stage union,” but do
not call it the end-to-end bottleneck until the unattributed interval and
exclusive critical path are resolved.

### Low severity / validated behavior

- Branch DAG ancestry, outcome balance, immediate token/action divergence, and
  shared-root lifetime are derived from exact synthetic trajectories and clearly
  limited to one four-branch group.
- Shared timing spans for the combined model/environment fork are deduplicated;
  the workload waterfall exposes inclusive union, envelope, and an explicit null
  exclusive critical path. This handling is correct.
- Model CAS sharing and eagerly materialized environment workspaces are kept
  separate in both the analysis and design brief.

## Continuum state review

### High severity

#### CONTINUUM-H1: the reported migration Amdahl bounds do not isolate migration

`analyze_run_amdahl` assigns every event in any `continuum-trace` file to the
`migration_latency` objective and uses the envelope of the entire state stream
(`python/sloforge/helix/characterization/workflow.py:407`). In the harness, the
measured snapshot, publish, fork, COW, append, delta, checkpoint, reshard, and
real in-process transfer occur on `characterization-branch-a`. A separate
pre-copy transaction starts later at
`python/sloforge/continuum/characterization/harness.py:829` and exposes only its
transport wrappers and ownership commit in the canonical stream.

Impact: the published free-reshard, free-delta, free-COW, and free-reclaim
“migration” bounds combine operations from a different lifecycle with a
denominator spanning both lifecycles. They are not migration critical-path
bounds. The same implementation labels the vertical state-event envelope a
`full_helix_transaction`, even though it is not the 10.7-second end-to-end run.

Required resolution: introduce explicit objective boundaries/transaction IDs,
derive the denominator from the matching causal transaction, and include only
dependent operations. Rename any partial objective to “traced state-lifecycle
envelope.” Regenerate Amdahl, bottleneck, placement, design-brief, and negative
finding claims.

#### CONTINUUM-H2: the trace loses observed repack/checksum work and pipeline dependencies

The measured conversion declares the operation chain
`STATE_RESHARD`, `STATE_REPACK`, `STATE_CHECKSUM` and times
`direct_convert_capture` (`python/sloforge/continuum/characterization/harness.py:788`).
It emits only `STATE_RESHARD`; the lifecycle adapter retains only scalar
attributes, so the tuple-valued `operation_chain` is discarded
(`python/sloforge/helix/characterization/trace/adapters.py:135`). The state event
also has no dependency on its snapshot or subsequent send. The transform report
therefore sees a one-stage chain and the design brief says no repack chain occurs
(`docs/branchfabric/CHARACTERIZATION_DESIGN_BRIEF.md:59`).

Impact: transform frequency, integrity work, common sequences, fusion analysis,
and negative ISA classifications are biased by missing instrumentation. A
combined inclusive span may be valid, but every semantic operation must be
represented with a shared timing-span identity and “do not sum” attribution, or
the combined event must have a schema-level operation-chain representation.

Required resolution: preserve the full chain and causal dependencies without
double-counting timing. Regenerate transform, integrity, Amdahl, requirements,
and `WHAT_NOT_TO_BUILD.md`.

### Medium severity

#### CONTINUUM-M1: migration internals are not a complete state-lifecycle trace

The pre-copy call performs capture/delta/import/activation/resume semantics, but
the canonical migration portion contains only three send/receive pairs and a
commit. Pre-copy convergence, final delta, recopy bytes, dirty-rate time series,
conversion, destination activation, and resume latency are not attributed to the
migration transaction.

Required resolution: instrument those existing Continuum boundaries with the
migration transaction ID and preserve simulated link timing separately from
real host work.

### Low severity / validated behavior

- Simulated transport timing is consistently classified `SIMULATED_HARDWARE`;
  in-process host copies remain `HARDWARE_BACKED_REAL` host timing.
- Logical Continuum COW is not described as an OS or GPU page fault.
- Model migration state and Helix filesystem environment state are not
  conflated.

## GPU performance review

### High severity

#### GPU-H1: generated hardware commands cannot produce hardware measurements

The hardware-status artifact generates `sloforge helix characterize run
--hardware gpu` as the future GPU, multi-GPU, and multi-node command
(`python/sloforge/helix/characterization/hardware_studies.py:465`). The
orchestrator unconditionally writes `SKIPPED_UNAVAILABLE` for every hardware
stage, including when capability evidence says compatible hardware exists
(`python/sloforge/helix/characterization/orchestration.py:831` and
`python/sloforge/helix/characterization/orchestration.py:965`). There is no GPU
worker implementation.

Worse, on a machine with `nvidia-smi`, the Make target runs that skip-only
workflow and then records `status=exercised` and `gpu_results_claimed=true`
without inspecting exercised hardware evidence (`Makefile:231`).

Impact: the promised exact future commands are nonfunctional, and an accessible
GPU host can produce a false hardware-backed completion marker.

Required resolution: fail closed unless a hash-verified exercised-hardware
artifact exists and contains the required operations. Until a real runner is
implemented, future commands must say “not implemented” and the Make target must
record `unavailable/unexercised`, never `exercised`.

### Low severity / validated behavior

- Current GPU, multi-GPU, NVLink, CUDA, and Helix runtime results are honestly
  unavailable; no Apple integrated-GPU metric is relabeled as CUDA evidence.
- HBM, PCIe, GPU copy engine, kernel, and interference values remain null rather
  than analytically fabricated.

## Networking review

### Medium severity

#### NETWORK-M1: zero multicast opportunity is only a bounded-corpus absence

The conservative detector is correct for the four fanout-1 sends, but the corpus
does not execute multi-destination transfers. This supports `NOT_JUSTIFIED` only
for the current fixture, not a production negative finding. The design brief
mostly uses the right scope; requirements and final summaries must preserve that
wording.

#### NETWORK-M2: no physical transport requirement is derivable

There is one 3.4-KiB in-process copy and three simulated-link sends. NIC source
load, link bytes, protocol overhead, topology, destination conversion,
retransmission, overlap, and burst windows are absent. Keeping all NIC bandwidth
requirements unknown is correct; do not convert the 883.5-MB/s host microbaseline
into a device or network target.

### Low severity / validated behavior

- Send and receive copies are not double-counted as source payload in the
  transport analysis.
- Event-envelope hashes are not used as payload identities for multicast.
- The SmartNIC/DPU no-build decision follows from the physical-network evidence.

## Storage and memory review

### High severity

#### STORAGE-H1: memory/page/queue “requirements” are mostly unexecuted matrix declarations

The requirements file correctly leaves HBM/DDR/CXL/capacity values unknown, but
the declared 8--256 branch rows are not measured capacity points. No workload
class has simultaneous state residency, eviction, dirty-log, transform-buffer,
or network-buffer capacity measurements. Queue minimum depth 1 is merely the
maximum hard-coded/serialized concurrency observed in the fixture.

Impact: this is an explicit unknown, not a completed memory, page-size, or queue
requirement derivation. The design brief correctly recommends no sizing, so the
phase cannot simultaneously claim those completion-contract items are complete.

Required resolution: either execute capacity/concurrency sensitivity studies or
mark the characterization incomplete. Do not promote depth 1 or allocation-only
4 KiB to hardware requirements.

### Medium severity

#### STORAGE-M1: known host/CAS locations become `unknown` in Helix canonical events

The Helix lifecycle records source/destination representations such as
`reference_runtime_memory` and `content_addressed_checkpoint`, but
`CanonicalLifecycleRecorder._state_event` does not map source/destination
locations and hard-codes empty dependencies/concurrency 1
(`python/sloforge/helix/characterization/trace/adapters.py:220`). This contributes
to an all-unknown roofline even for host-resident data.

Required resolution: record the known host/CAS/storage tiers and explicit causal
dependencies while leaving genuinely unavailable physical counters unknown.

### Low severity / validated behavior

- The page projection is labeled analytical and does not claim a universal page
  size.
- CAS content deduplication, live workspace physical bytes, and filesystem block
  sharing are explicitly separated.
- The roofline report correctly refuses a dominant-resource classification when
  counters and ceilings are incomplete.

## Instrumentation-overhead review

### High severity

#### OVERHEAD-H1: the overhead study does not time the actual full trace path

The overhead trial passes only `InMemoryLifecycleRecorder` to the lifecycle
harness. The actual vertical full trace tees raw recording with
`CanonicalLifecycleRecorder`, which performs strict model construction and
bounded-buffer admission on the hot path. Full/minimal overhead trials do not
include that canonical adapter, and persistence/adaptation is explicitly outside
the timed run. In addition, the lifecycle emitter does not distinguish minimal
from full fields/events; only the later canonical buffer has minimal filtering.

Impact: the reported -0.126% full-trace wall change measures wrapper plus raw
in-memory recording, not the deployed full BranchWorkloadTrace and
StateOperationTrace hot path. It cannot close the instrumentation-distortion
requirement. CPU time does show a roughly 3.5% median increase, which should
remain reported but is also for the incomplete path.

Required resolution: measure disabled/minimal/full using the exact recorder
topology used in `run_vertical_trace`, with persistence reported separately.
Include attempted/accepted/filtered/dropped counts per trial and retain semantic
equivalence checks.

### Medium severity

#### OVERHEAD-M1: each per-level series includes every level's warmup samples

`_series` filters measured samples by trace level but collects warmups with only
`if trial.warmup` (`python/sloforge/helix/characterization/overhead.py:256`). Each
per-level series therefore declares three warmups and repeats the same three
values even though the schedule has one warmup for that level. Post-warmup
statistics are unaffected, but provenance, selector sample count, and the warmup
record are wrong.

Required resolution: filter warmups by `trial.trace_level is level`, correct the
artifact evidence sample selector/count, add a regression test, and regenerate
the overhead artifact.

#### OVERHEAD-M2: semantic equivalence is a projection, not complete artifact equivalence

The semantic digest excludes several hashes/latency fields and only selects part
of the demo summary. This is useful but does not verify exact branch/checkpoint,
trajectory, promotion, and transaction artifacts across levels.

Required resolution: compare a documented semantic artifact manifest in
addition to the summary projection.

## FPGA review

### High severity

No FPGA-specific overreach was found. The high-severity Amdahl and transform
issues above must nevertheless be fixed before using the numeric 1.02349x bound
as an FPGA gate.

### Low severity / validated behavior

- `docs/branchfabric/CHARACTERIZATION_DESIGN_BRIEF.md:159` correctly recommends
  no FPGA target and requires device-scale byte-dominant traces before
  reconsideration.
- No RTL, HLS, FPGA simulator, bitstream, or hardware driver file was found.
- The placement artifact declines invented numeric suitability scoring.

## SmartNIC/DPU review

### High severity

No DPU feature overreach was found in the recommendation itself. `GPU-H1` also
applies to the generated multi-node command: it cannot produce the evidence the
document promises.

### Low severity / validated behavior

- `docs/branchfabric/CHARACTERIZATION_DESIGN_BRIEF.md:165` correctly recommends
  no SmartNIC/DPU implementation.
- The recommendation requires physical source-NIC saturation, CPU involvement,
  identical payload identity, destination transforms, and end-to-end leverage
  before reconsideration.

## Cross-cutting requirements and acceptance review

### High severity

#### REQUIREMENTS-H1: machine-readable ISA disposition contradicts the reviewed documents

The compiler deliberately emits `software_only_operations=()`
(`python/sloforge/helix/characterization/workflow.py:1616`), while
`docs/branchfabric/WHAT_NOT_TO_BUILD.md:188` classifies allocation, map, publish,
COW, append, hash/checksum, transaction, retry, reclaim, and free as “SHOULD
REMAIN SOFTWARE.” The hardware placement report similarly recommends CPU
software for many operations.

Impact: the machine-readable handoff has zero software-only operations and
eleven unresolved operations, so it does not satisfy the required identification
of software-only operations and does not agree with the human design decision.

Required resolution: reconcile classifications through one evidence-gated
source of truth. If insufficient evidence prevents `SOFTWARE_ONLY`, the human
documents must say unresolved; if the evidence supports software-only, encode it
with artifact references in the requirements JSON.

#### ACCEPTANCE-H1: clean-room acceptance does not regenerate the claimed characterization

`branchfabric-clean-room-test` regenerates the fixed CPU orchestration,
requirements, and short generated report only (`Makefile:256`). It does not
regenerate or validate the direct three-seed Helix/Continuum corpus, workload DAG
reports, COW study, transform/transport analyses, Amdahl/roofline/placement,
software baselines, hardware-status artifact, flagship manifest, design brief,
or `WHAT_NOT_TO_BUILD.md`. Those artifacts are currently untracked workspace
outputs. The mandated `BRANCHFABRIC_CHARACTERIZATION_FINAL_REPORT.md` was absent
at review time.

Impact: the acceptance command can pass while the principal reported artifacts
are stale, missing, or hash-inconsistent. It is insufficient for the requested
clean-state reproduction contract.

Required resolution: add a bounded clean-state workflow that regenerates every
claim-bearing derived artifact from preserved raw inputs (or reruns the raw CPU
experiments where required), verifies all hashes and source revisions, rebuilds
the flagship and final report, and fails on stale/missing evidence.

### Medium severity

#### REQUIREMENTS-M1: the requirements report uses one orchestration sample

The requirements/report statistics are derived from the single CPU-reference
run, while the design brief cites direct seeds 41, 73, and 113. Point estimates
presented as p95/p99 from one branch group or one operation remain schema-valid
but do not provide tail evidence.

Required resolution: bind the replicated traces into one requirements corpus or
label single-sample percentile fields as point estimates without tail meaning.

## Focused validation performed

The following read-only validation was run on 2026-08-09:

```text
uv run --locked pytest -q \
  tests/python/test_branchfabric_hardware_studies.py \
  tests/python/test_branchfabric_workload_analysis.py \
  tests/python/test_branchfabric_amdahl.py \
  tests/python/test_branchfabric_overhead.py \
  tests/python/test_branchfabric_requirements.py \
  tests/python/test_branchfabric_transform_analysis.py \
  tests/python/test_branchfabric_transport_analysis.py
```

Result: **34 passed**.

Additional integrity checks produced:

- requirements schema validation: pass;
- all 8 `verified_artifacts` present and SHA-256 matched;
- Helix seed-41 manifest: all 6 artifact hashes matched, zero drops;
- Continuum seed-41 manifest: all 5 artifact hashes matched, zero drops;
- hardware-study schema validation: pass, all three studies `UNAVAILABLE`;
- design brief sections 1 through 30 present;
- repository hardware-source scan: no RTL/HLS source found.

These checks establish schema and artifact integrity. They do not resolve the
semantic and coverage findings above; current unit tests encode the present
behavior, including the invalid future hardware route and incomplete Amdahl
objective selection.

## Required disposition before final handoff

1. Fix or withdraw every high-severity quantitative claim, especially migration
   Amdahl and instrumentation overhead.
2. Make hardware commands fail closed and incapable of claiming an exercised GPU
   run without verified exercised evidence.
3. Execute the required CPU workload classes and a causally complete flagship,
   or keep the phase explicitly incomplete.
4. Regenerate the corpus, requirements, negative findings, design brief, and
   final report from source-identified evidence.
5. Extend clean-room acceptance to every claim-bearing artifact.

Until those conditions are met, the only hardware-architecture conclusion this
review approves is: **do not implement BranchFabric hardware from the current
corpus**.
