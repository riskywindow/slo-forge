# Instrumentation-Overhead Review

## Verdict

The instrumentation-overhead completion condition is **not yet closed**.

The authoritative primary artifact
`artifacts/branchfabric/overhead/helix-cpu-3seed-2rep-v4/instrumentation-overhead.json`
(SHA-256 `0eac7c0173f0311a4f9b461892ba6e8d86ce8f99d9f8ec225f7a945205a0fe12`)
was produced from source commit
`f1ebbf621241ca27a40645c04f95bd421079eba8`. At that commit the overhead
harness timed only `InMemoryLifecycleRecorder`. It did not time the canonical
Pydantic adapter, canonical event hash, bounded-buffer lock, or event sealing
used by the vertical trace runner. Its reported full-versus-disabled wall-time
effect therefore does not measure the actual BranchWorkloadTrace and
StateOperationTrace hot path.

The current working tree contains the right first correction: the overhead
harness tees raw events into `CanonicalLifecycleRecorder` and a
`BoundedTraceBuffer`, matching the recorder topology in `run_vertical_trace`.
It also fixes the per-level warmup filter and places warmups before measured
runs. Those corrections have not yet produced a clean, committed replacement
artifact. The existing `-0.126%` paired wall-time point estimate and `+3.5%`
full-mode CPU-time median difference must remain labeled as measurements of the
old in-memory-only path.

This review made no source or raw-artifact changes. The focused trace and
overhead tests passed: 13 tests in 0.87 seconds.

## Evidence inspected

- Hot-path lifecycle creation and hashing:
  `python/sloforge/helix/characterization/lifecycle/harness.py`, especially
  `_Emitter.emit`, `_install_wrappers`, and `run_characterized_cpu_demo`.
- Raw recorder:
  `python/sloforge/helix/characterization/lifecycle/recorder.py`.
- Canonical adaptation, validation, sealing, and locking:
  `python/sloforge/helix/characterization/trace/adapters.py`,
  `trace/canonical.py`, and `trace/buffer.py`.
- Post-run persistence:
  `python/sloforge/helix/characterization/runner.py` and `trace/io.py`.
- Resource-counter behavior:
  `python/sloforge/helix/characterization/resources.py`.
- Overhead schedule, trial schema, analysis, and metric-availability claims:
  `python/sloforge/helix/characterization/overhead.py`.
- Primary raw samples, SHA-256
  `2c0698061964021a1472c3000acb2ff811d4dd2e906d76359ab7c2d0efec696e`.
- Independent replication:
  `artifacts/branchfabric/replication/overhead/` and
  `docs/branchfabric/reviews/INDEPENDENT_REPLICATION.md`.

## High-severity findings

### IOH-H1: the authoritative artifact does not measure the canonical hot path

At source commit `f1ebbf6`, `_run_trial` passed one
`InMemoryLifecycleRecorder` to `run_characterized_cpu_demo`. In contrast,
`run_vertical_trace` tees every lifecycle mapping to both the in-memory raw
recorder and `CanonicalLifecycleRecorder` (`runner.py:262-280`). The canonical
recorder constructs a full immutable Pydantic event synchronously
(`trace/adapters.py:152-255`), then the bounded buffer takes a global lock and
seals the event (`trace/buffer.py:107-132`). Sealing performs a model dump,
canonical JSON encoding, and SHA-256 (`trace/canonical.py:41-65`). These costs
were absent from all primary and replication overhead trials.

The old primary artifact contains six measured samples per level. It reports:

| Metric | Disabled | Full | Interpretation |
|---|---:|---:|---|
| wall-time median | 11.211 s | 11.221 s | in-memory observer only |
| paired median wall change | - | -0.126% | inside measured noise; not a speedup |
| CPU-time median | 2.030 s | 2.101 s | about +3.5%, in-memory observer only |

The two-pair independent replication reverses the wall-effect sign to +1.34%
and observes about +6.5% CPU time. It confirms that the wall penalty is
unresolved even for the incomplete path.

The working-tree tee correction is structurally appropriate, but no existing
artifact contains its new `canonical_event_count` fields. Until a replacement
artifact is produced, report and design-brief claims must not describe the
primary result as "full trace overhead."

Required resolution:

1. Commit the exact canonical-tee implementation and tests.
2. Run from a clean tree whose recorded revision identifies both workload and
   instrumentation code.
3. Regenerate disabled/minimal/full samples with the same three seeds and at
   least the current two repetitions per seed.
4. Preserve attempted, accepted, filtered, dropped, and buffered event counts
   for each trial, not only accepted and dropped counts.
5. Replace downstream overhead claims and retain the old artifact as explicitly
   invalid for canonical-path overhead.

### IOH-H2: required memory, storage, throughput, and branch-latency overheads are not comparable

The artifact schema retains several values per enabled trial, but it only
derives statistical summaries for wall time and CPU time. More importantly,
the remaining values do not have a tracing-disabled comparator:

- `ResourceSampler.start` returns immediately for `TraceLevel.DISABLED`
  (`resources.py:587-603`). Every disabled primary trial has zero samples and
  null RSS/VMS/I/O. Minimal/full trials have 109-114 samples. Memory overhead
  therefore cannot be computed from this study, and the sampler's own work is
  folded into enabled-mode wall/CPU effects without independent attribution.
- `workload_artifact_bytes` is measured before the resource trace is written
  and contains the normal Helix demo output. It does not contain canonical
  JSONL, Perfetto, Parquet, manifest, or raw-lifecycle trace output.
  `process_write_bytes_delta` is null in every primary trial on this host.
  Consequently the current `storage_writes: AVAILABLE` declaration is false
  for trace persistence overhead.
- `branch_readiness_ns` is the duration copied from the first `BRANCH_FORK`
  lifecycle event. All four fork events refer to one combined model-and-
  environment group-fork span and explicitly say
  `shared_span_do_not_sum_across_branch_events`. This is group-fork duration,
  not fork-to-branch-ready latency. Disabled trials have no corresponding
  value.
- Rollout throughput is computed only from enabled wrapper events. Disabled
  trials have neither generated-token nor rollout-duration values, so no
  tracing overhead can be calculated. The fixture also emits policy-decision
  tokens rather than model-server token latency.
- CPU utilization is stored per trial as CPU time divided by wall time, but no
  paired summary is emitted. Peak memory, storage values, branch-group fork,
  and rollout throughput likewise have no summary, confidence interval, noise
  floor, or effect size.

The availability map correctly declares TTFT, per-token latency, GPU
utilization, and GPU-backed behavior unavailable. It should apply the same
standard to the non-comparable fields above.

Required resolution:

1. Use an identical external measurement sidecar for all three trace levels to
   obtain comparable RSS/VMS and I/O observations. Separately run a sidecar
   off/on control to measure the counter collector's own perturbation.
2. Add level-independent outer stage probes for group-fork, rollout, reward,
   training, and full transaction boundaries. Keep these probes identical in
   disabled/minimal/full trials. Rename the current value
   `combined_branch_group_fork_ns`; do not call it branch readiness.
3. Time trace finalization separately: buffer drain, BranchWorkload JSONL,
   StateOperation JSONL, Perfetto, optional Parquet, manifest, bytes written,
   and peak finalize RSS. Report both hot-path and finalize overhead rather than
   merging them.
4. Emit paired effects and confidence intervals for every actually comparable
   metric. Mark unsupported metrics `UNAVAILABLE` rather than merely retaining
   null trial fields.

### IOH-H3: the only macro workload has 78 attempted events and cannot validate the synchronous design at target rates

The vertical demo produces 61 branch and 17 state observations in full mode.
It does not exercise concurrent rollout producers, high fanout, sustained event
rates, sampling, buffer pressure, or a million-event corpus. On the current hot
path, one event incurs raw event canonicalization and SHA-256 in `_Emitter`, an
in-memory recorder lock and copy, canonical Pydantic construction, a global
buffer lock, another canonical model dump/JSON encoding/SHA-256, and retention
until the run ends. Minimal filtering occurs only after canonical Pydantic
construction, so filtered events still pay most adapter cost.

The buffer is bounded and overflow is observable, which is correct. However,
the evidence does not establish that its synchronous validation and sealing
preserve scheduling or stay low overhead at the fanouts and event rates that
motivate BranchFabric.

Required resolution:

1. Add a deterministic recorder replay benchmark using event-size and
   operation mixes from the preserved raw trace.
2. Sweep 1, 2, 4, and 8 producer threads where supported; minimal/full levels;
   deterministic stride sampling; and capacities that cover both no-overflow
   and controlled-overflow cases.
3. Preserve enqueue latency p50/p95/p99/max, events/s, CPU time, peak RSS,
   attempted/accepted/filtered/dropped counts, and drain/finalize time for at
   least 10,000 events. Use a larger case only if it remains within the declared
   time and memory budget.
4. Run one highest-feasible controlled Helix fanout macro trial. If the
   synchronous path materially distorts execution, move canonicalization and
   sealing behind a bounded producer ring and quantify the information-loss and
   overflow policy.

## Medium-severity findings

### IOH-M1: the primary warmups neither precede measurement nor belong to their reported series

In the primary raw schedule, the full, minimal, and disabled warmup trials occur
at order indices 4, 5, and 18. Measured trials therefore run before their
declared warmup. At commit `f1ebbf6`, `_series` also selected all three warmups
for every level while selecting measured samples by level. The derived measured
medians did not include warmups, but the warmup policy, provenance sample count,
and claim that measurements followed warmup are incorrect.

The working tree now schedules the warmup block first and filters warmups by
level. The new regression test passes. Regenerate the artifact before treating
the correction as measurement evidence.

### IOH-M2: minimal-mode information loss and buffer accounting are incomplete

Lifecycle minimal mode still constructs and emits high-level rollout events,
while `BoundedTraceBuffer` excludes `BranchOperationType.ROLLOUT` from its
minimal set (`trace/buffer.py:20-50`). The canonical event is constructed before
that exclusion. Thus "minimal" means a mixture of wrapper omission and
post-construction filtering, not a uniformly cheap event level.

The working-tree trial schema records canonical accepted events and drops, but
not attempted events, filtered events, event-type counts, or high-water
occupancy. It therefore cannot quantify minimal-mode information loss or prove
accounting equality.

Required resolution: preserve the complete `TraceBufferStats` per trial; list
operation types retained and omitted at each level; and report CPU/storage
savings against the corresponding lost fields and events.

### IOH-M3: semantic equivalence is a partial summary projection

`_semantic_digest` intentionally selects eleven summary fields and removes
latency/duration values plus several artifact hashes. That is a useful guard,
but it does not prove equality of all trajectory, checkpoint, environment,
reward, trainer, transaction, and promotion artifacts. A tracing bug that
changes an omitted output can pass `semantic_equivalence_verified`.

Required resolution: create a semantic artifact manifest from every output
whose contents should be identical across trace levels, normalize only declared
timestamps and trace-only fields, and compare its hash in addition to the
current summary projection.

### IOH-M4: the wall-time effect is below the noise floor and has no paired confidence interval

The primary disabled relative MAD is about 0.67% and p95 absolute deviation is
about 3.33%, while the paired median full effect is -0.126%. The independent
replication changes the sign. The study reports per-level bootstrap median
intervals, but not a confidence interval for paired relative differences or a
minimum detectable effect. Trials also share one long-lived Python process;
enabled peak RSS rises with run order, and disabled RSS is unobserved.

Required resolution: keep the current conclusion "wall overhead unresolved";
add a paired bootstrap interval and declared resolution threshold; record
background load for every level; and prefer one fresh bounded subprocess per
trial so caches, GC, and retained process state do not accumulate across the
randomized schedule.

### IOH-M5: a new run can misidentify modified instrumentation as commit `f1ebbf6`

`.sloforge-source-commit` currently contains
`f1ebbf621241ca27a40645c04f95bd421079eba8`, while the instrumentation source is
modified and repository `HEAD` is different. `demo._source_commit` prefers that
marker (`demo.py:125-132`), and `_source_commit_from_trial` copies it into the
overhead artifact. Running the corrected harness in this state would label new
measurements with the old revision even though the measured instrumentation
implementation changed.

Required resolution: do not regenerate while the tree is dirty. Commit the
instrumentation, remove the temporary marker or update it to the exact measured
commit, and assert marker/HEAD equality before every overhead trial. If workload
and instrumentation revisions are intentionally different, record both fields
explicitly.

## Validated behavior

- The run schedule is deterministically randomized from an explicit seed.
- Paired wall samples are sorted by `(seed, repetition)` before effect-size
  computation.
- Source-revision disagreement and semantic-digest disagreement fail the study.
- Raw samples, outliers, schedule positions, and warmup flags are preserved.
- Trace and resource buffers have explicit capacities, timeout-bounded shutdown,
  and observable drop accounting.
- Event ordering and event-content hashes are tested.
- Canonical JSONL persistence is batched and streaming after collection.
- The report does not turn a negative point estimate into a speedup claim.
- GPU, PCIe, NVLink, and model-server token metrics are honestly unavailable on
  the measured host.

## Minimal remediation sequence

1. Commit the canonical-tee, warmup, and complete buffer-accounting changes.
2. Add clean-revision guards and separate workload/instrumentation revisions.
3. Add identical cross-level stage/resource probes and a separate finalization
   timing record.
4. Add the bounded deterministic recorder replay; do not redesign the buffer
   unless that evidence shows material distortion.
5. Regenerate primary and replication overhead artifacts from clean states.
6. Update the design brief, methodology, final report, and requirements evidence
   to cite only the replacement artifact. Preserve old raw artifacts with a
   machine-readable invalidation reason.

## Focused acceptance commands

Run before measurement:

```bash
git diff --quiet
test ! -e .sloforge-source-commit || \
  test "$(tr -d '\n' < .sloforge-source-commit)" = "$(git rev-parse HEAD)"
```

Run the focused implementation checks:

```bash
uv run --locked pytest -q \
  tests/python/test_branchfabric_overhead.py \
  tests/python/test_branchfabric_trace_adapters.py \
  tests/python/test_branchfabric_trace_io.py \
  tests/python/test_branchfabric_resources.py
uv run --locked ruff check \
  python/sloforge/helix/characterization/overhead.py \
  python/sloforge/helix/characterization/trace \
  tests/python/test_branchfabric_overhead.py
uv run --locked mypy python/sloforge/helix/characterization
```

Regenerate the bounded orchestrated study and then run trace integrity checks:

```bash
make branchfabric-characterization-cpu
make branchfabric-trace-check
```

The replacement overhead artifact is acceptable only if:

- every measured trial records the clean instrumentation revision;
- disabled/minimal/full semantic artifact manifests match;
- every trial balances attempted = accepted + filtered + dropped;
- no unreported buffer or resource samples are lost;
- stage and resource metrics have a level-independent baseline;
- persistence time and bytes are reported separately from hot-path overhead;
- paired intervals show whether the effect is resolved or below the experiment's
  noise floor.
