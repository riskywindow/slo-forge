# BranchFabric adversarial measurement and hardware-justification review

Review scope: current BranchFabric characterization source, canonical traces,
analysis artifacts, generated requirements, characterization report, design
brief, negative-findings document, and Make workflow. Raw benchmark artifacts
were read but not modified.

## Verdict

**Does the BranchFabric design brief follow from measured Helix workloads, or
are we retrofitting measurements to justify a predetermined hardware project?**

The design brief's principal conclusion follows from the bounded evidence: it
says **not to build BranchFabric hardware yet**, leaves hardware sizes and
targets unknown, and rejects a frozen ISA. That is the opposite of retrofitting
measurements to justify a predetermined device. The brief consistently labels
the workload synthetic, simulated network timing simulated, and unavailable
GPU/network measurements unavailable.

This verdict does **not** validate the characterization as complete. The
present package is a sound trace/instrumentation vertical slice and a useful
negative hardware result, but it is not yet the comprehensive empirical Helix
workload characterization in the completion contract. Presenting it as that
completed phase would be unsupported.

No measured result currently supports an FPGA, SmartNIC/DPU, CXL state store,
hardware page table, multicast engine, or fixed BranchFabric ISA. The most
defensible architecture decision is the one already stated in
`docs/branchfabric/CHARACTERIZATION_DESIGN_BRIEF.md`: retain a software and
measurement reference architecture and gather the missing evidence.

## High-severity findings

### H1. The experiment matrix is validated, not executed

`artifacts/branchfabric/characterization/cpu-reference/run-manifest.json`
records 855 expanded cases and 18 sweep dimensions, but the matrix stage only
loads, expands, hashes, and records those cases. The next stage invokes one
fixed `run_vertical_trace` call; it does not iterate cases. See
`python/sloforge/helix/characterization/orchestration.py:1091` and
`python/sloforge/helix/characterization/orchestration.py:1136`.

The generated report correctly admits that declared 8--128 fanouts were not
executed (`reports/branchfabric-characterization/CHARACTERIZATION_REPORT.md:66`),
but this means the required fanout, prefix, suffix, divergence, survival,
workload-class, pressure, migration, fault, and physical-layout studies do not
exist as measurements. A matrix declaration cannot substitute for a workload
corpus.

Required correction: execute a bounded, randomized, restartable selected
subset that covers every required workload class and critical dimension;
preserve per-case raw artifacts and independent repetitions. Continue to mark
controlled cells `SYNTHETIC` and do not infer a production distribution from
their frequencies.

### H2. The flagship run is not the requested full Helix characterization

The analyzed flagship is one fanout-four group. Each branch trajectory has one
token and one action, and all branches diverge immediately. Its waterfall has
no observed `PRODUCTION`, `BRANCH_READY`, `MIGRATE`, `RESUME`, or `EVALUATE`
stage. The exact evidence is
`artifacts/branchfabric/analysis/workload/helix-coding-agent-seed-41.json`.
Migration is measured in a separate tiny Continuum fixture, not in the same
production-to-promotion transaction. The design brief acknowledges this at
`docs/branchfabric/CHARACTERIZATION_DESIGN_BRIEF.md:95` and
`docs/branchfabric/CHARACTERIZATION_DESIGN_BRIEF.md:183`.

This does not satisfy the requested long-context, 32--128-sibling, pruning,
capacity-reclamation, checkpoint, migrate, resume, reward, train, evaluate,
and promote flagship waterfall. It also cannot characterize late divergence,
long-tail branch lifetime, or state growth with generated tokens.

Required correction: run a representative full lifecycle at a feasible but
declared scale, with a single causally linked waterfall. If 32--128 branches or
long context are infeasible on this host, preserve that as an unavailable
measurement rather than calling the fanout-four one-token fixture the full
flagship.

### H3. The machine-readable ISA disposition is incomplete and disagrees with the documents

The compiler enumerates 24 candidate operations at
`python/sloforge/helix/characterization/workflow.py:1293`, but the generated
requirements classify only 18: 11 unresolved, 7 `NOT_JUSTIFIED`, zero
recommended, and zero software-only. `STATE_ALLOC`, `STATE_MAP`,
`STATE_REPACK`, `STATE_HASH`, `STATE_CHECKSUM`, and `STATE_ABORT` are absent
from every output classification list. The selection is controlled by a
hand-written seven-operation `absent_not_justified` set at
`python/sloforge/helix/characterization/workflow.py:1319`; another absent
candidate such as `STATE_REPACK` is silently omitted.

Meanwhile, `docs/branchfabric/WHAT_NOT_TO_BUILD.md:182` supplies a manual
disposition for these and additional trace operations, frequently classifying
them `SHOULD REMAIN SOFTWARE`. This contradicts
`artifacts/branchfabric/requirements/branchfabric_requirements.json`, where
`software_only_operations` is empty, and fails the requirement that every ISA
candidate receive an evidence-backed machine-readable classification.

Required correction: enforce a coverage invariant over the complete candidate
set; derive the document tables from the same machine-readable records. An
unobserved operation should normally be `NOT_JUSTIFIED` for this bounded corpus
or `UNKNOWN`, with the scope explicit. A software-only classification needs a
matched software baseline and end-to-end leverage evidence, not merely an
absent trace event.

### H4. `metadata.operations_per_second` is software capacity, not workload demand

The requirements compiler populates `metadata.operations_per_second` from the
median throughput of every `current_software` metadata microbenchmark across
different operation types and thread counts. See
`python/sloforge/helix/characterization/workflow.py:961` through
`python/sloforge/helix/characterization/workflow.py:987`. These values are
microbenchmark service capacities for heterogeneous operations, not metadata
operations demanded per second by a Helix workload. Pooling them into p50/p95/
p99 is not a meaningful demand distribution.

The generated report softens the label to "Current-software metadata throughput
summaries" (`reports/branchfabric-characterization/CHARACTERIZATION_REPORT.md:29`),
but the JSON field is the hardware-requirements field an architect will
consume. It can be misread as a queue/engine rate requirement.

Required correction: leave workload metadata demand `UNKNOWN` until trace
events or a joined counter stream measures it. Preserve microbenchmark capacity
as a separately named, operation-stratified software-baseline field; never pool
unlike operations into workload-rate percentiles.

### H5. The hardware-requirements statistics are single-fixture point observations

The generated requirements declare one coding-agent experiment and one branch
group. Fanout percentiles therefore repeat the value four with sample count
one; branch-lifetime percentiles cover four correlated siblings from the same
group. The requirements compiler does not aggregate the standalone seed 41,
73, and 113 traces or attach confidence intervals. See
`artifacts/branchfabric/requirements/branchfabric_requirements.json` and
`reports/branchfabric-characterization/CHARACTERIZATION_REPORT.md:18`.

The design brief references repeated seed artifacts, but its main branch,
sharing, divergence, transform, network, and Amdahl conclusions use seed 41 or
one Continuum trace. This is acceptable for conformance and negative evidence,
not for tail requirements or workload distributions.

Required correction: aggregate independent seeds and repetitions at the
experiment/group level, retain within-group correlation, and report confidence
intervals or explicitly label point observations. Do not present p95/p99 from a
single group as empirical tails.

### H6. The completion report and corpus-wide verification are not present in the reviewed tree

`BRANCHFABRIC_CHARACTERIZATION_FINAL_REPORT.md` was absent during this review.
The generated report at
`reports/branchfabric-characterization/CHARACTERIZATION_REPORT.md` covers the
small orchestrated CPU run, not the separate three-seed traces, valid overhead
study, software baselines, environment study, active experiment queue,
workload-DAG analysis, or replication artifacts. The required comprehensive
statistical analysis and exact experiment counts are therefore not in one
verified handoff.

Required correction: create the final report only after the high-severity
measurement gaps are either completed or explicitly declared inapplicable/
unavailable. Bind every quantitative claim to an immutable hash and one corpus
manifest, including negative results and unavailable hardware paths.

## Medium-severity findings

### M1. Requirements provenance conflates workload provenance and timing provenance

Canonical events correctly carry independent `workload_provenance` and
`timing_measurement_class`. The requirements evidence object retains only an
`evidence_class`, usually `SYNTHETIC`. Consequently, a host-timed reshard and a
deterministic simulated transfer are distinguishable only through selected
rationale text, not a stable machine-readable timing field.

Correction: add timing-measurement classes to requirement evidence or bind a
typed timing provenance record. Keep `SYNTHETIC` workload semantics independent
from `HARDWARE_BACKED_REAL`, `SIMULATED_HARDWARE`, and not-measured analytical
timing.

### M2. The design brief's evidence set is not covered by its own manifest

The generated requirements verify eight artifacts and all eight hashes matched
the files under
`artifacts/branchfabric/characterization/cpu-reference`. The broader design
brief additionally relies on standalone workload, overhead, environment,
software-baseline, Amdahl, roofline, hardware-status, COW, transform, and
transport artifacts. Those claims cite paths but are not tied together by a
design-brief manifest, and the files are not in the generated requirements'
`verified_artifacts` set.

Correction: generate a corpus/report manifest containing each cited artifact,
size, SHA-256, evidence/timing class, seed, repetition, source commit, and role.
Add a report verifier that rejects missing or changed citations.

### M3. Several documents combine distinct runs without a common run identity

The generated CPU-reference report uses the seed-20260809 orchestrated trace
(for example, 420.639 ms p50 branch lifetime and 2,213 shared bytes per fork),
while the design brief uses the standalone seed-41 trace (384.576 ms and 2,660
total physical fixture bytes). Both are valid observations, but readers can
mistake them for one corpus because `cpu-reference` is also the Amdahl artifact
name and no aggregate run ID is presented at the top of the design brief.

Correction: assign one corpus ID, list constituent run IDs and source commits,
and label each table by run/seed. Explain cross-run differences instead of
silently selecting one.

### M4. Some `SHOULD REMAIN SOFTWARE` findings are stronger than the demand evidence

`docs/branchfabric/WHAT_NOT_TO_BUILD.md` classifies dirty tracking and
environment snapshot acceleration `SHOULD REMAIN SOFTWARE`, even though dirty
updates are a faithful isolated micro-operation with no workload rate and the
environment study does not exercise physical filesystem COW, database
snapshotting, or process reconstruction. `NOT CURRENTLY JUSTIFIED` is the safer
classification until workload demand and matched acceleration opportunities
are measured. The same caution applies to hardware hashing: the host baseline
is useful, but no end-to-end hash fraction was measured.

Correction: reserve `SHOULD REMAIN SOFTWARE` for operations with representative
demand plus a strong matched baseline and negligible end-to-end leverage;
otherwise use `NOT CURRENTLY JUSTIFIED` or `UNCERTAIN`.

### M5. Amdahl objective names can overstate the measured boundary

The design documents call one objective the "full Helix transaction," but the
single trace lacks production, branch-ready, migration/resume, and evaluation,
and the migration objective mixes real host operation timings with simulated
transport timing. The documents disclose both limitations, so the arithmetic
is not hidden; the objective name is the remaining risk.

Correction: rename it to the exact bounded lifecycle span and keep real-host
and simulated-transport bounds separate. Do not use it as an upper bound for a
production-to-promotion learning transaction.

### M6. "Physical allocation" is model-store accounting, not observed memory allocation

The 1.0 amplification and 75% sharing figures are useful logical/content-store
accounting for a simulated model-state fixture. They are not RSS, allocator,
filesystem-block, HBM-page, or device-memory measurements. The design brief
usually says this, but phrases such as "physical allocation" can escape that
qualification when copied.

Correction: call the result `modeled physical/content-addressed allocation` in
all summaries and reserve unqualified physical bytes for counters from the
actual memory/storage owner.

### M7. Replace-oriented Make targets weaken raw-artifact immutability

`make branchfabric-characterization-cpu`, `make branchfabric-requirements`, and
`make branchfabric-report` write fixed paths with `--replace`. The trace
manifests make changes detectable, but a subsequent run can remove the only
copy at that path. At review time, most final characterization artifacts were
also untracked by Git.

Correction: publish immutable run-ID/source-commit directories, make the
friendly `cpu-reference` path an index or pointer, and never replace the only
raw evidence copy. Reports may be regenerated; raw samples should be append-only.

## Low-severity findings

### L1. Report artifact references are ambiguous outside their run directory

The report's evidence table prints paths such as
`attempts/vertical_trace/...`, while the requirements JSON's verified-artifact
records contain the resolving `file_path`. A standalone report reader has to
infer the base directory.

Correction: print both the corpus-relative reference and the repo-relative
resolved path, or state the evidence root immediately above the table.

### L2. Hardware-negative language should remain corpus-scoped in summaries

The detailed documents reliably use qualifiers such as "current corpus" and
"not currently justified." Preserve those qualifiers in the final executive
summary. Zero observed multicast opportunities at fanout one is evidence
against building multicast from this corpus, not evidence that production
Helix will never benefit.

## Validated strengths

- The baseline/hardware capability evidence is honest: no CUDA, multi-GPU,
  RDMA, or multi-node result is claimed, and no paid provisioning occurred.
- The seed-41 Helix and Continuum trace manifests record 78/78 and 18/18
  accepted events respectively, with zero drops and zero filtering.
- Workload provenance and timing classes are separate in canonical trace
  events, and simulated-network events remain labeled `SIMULATED_HARDWARE`.
- The transport analyzer does not infer multicast from fanout-one traffic and
  explicitly reports the missing payload identity/topology counters.
- The page-size document reports the 4/16/64 KiB allocation tie and declines a
  universal hardware page-size recommendation.
- The roofline reports unknown dominance instead of filling absent hardware
  counters with estimates.
- All eight artifact hashes in the generated requirements resolved and matched
  under the CPU-reference run.
- Focused requirements/workflow/workload/transport/transform/overhead/hardware
  tests passed: 34 tests, zero failures.

## Release recommendation

Do not begin BranchFabric implementation. The current negative design brief is
safe to use as a gate against premature hardware work, but do not label this
characterization phase complete until H1--H6 are resolved or explicitly
accepted as unavailable scope in the final completion audit. In particular,
no page size, queue depth beyond an observed lower bound of one, memory tier,
bandwidth target, latency target, or hardware primitive should be frozen from
the present corpus.
