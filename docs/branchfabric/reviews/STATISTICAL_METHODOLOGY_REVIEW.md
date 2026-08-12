# BranchFabric Statistical Methodology and Measurement-Bias Review

Review status: **BLOCKING FINDINGS**

Review scope: preserved CPU raw samples, cross-seed aggregation, instrumentation
overhead, environment and metadata studies, software baselines, Amdahl bounds,
and the machine-readable requirements. This review did not alter raw data. It
evaluates whether the current observations support workload distributions or
hardware requirements; it does not review hardware implementation.

## Verdict

The statistical utilities are generally careful: raw values are retained,
Hyndman--Fan type 7 percentiles are named, bootstrap seeds are explicit,
outliers are not silently removed, and synthetic workload provenance is usually
prominent. The current corpus is useful as a deterministic instrumentation
validation and as evidence for the conservative conclusion **do not build
BranchFabric hardware yet**.

It is not yet a statistically complete Helix characterization. The independent
experimental unit is frequently a single synthetic run, while output fields
present p95/p99 values, observed concurrency, or event rates as if they were
workload distributions or requirements. The workload matrix is controlled
design space with no production weights. No current result supports a
production tail, queue-depth requirement, page-size requirement, or hardware
bandwidth/latency target.

This verdict is based on the following artifact snapshot:

| Artifact | SHA-256 |
|---|---|
| `artifacts/branchfabric/overhead/helix-cpu-3seed-2rep-v4/overhead-raw-samples.json` | `2c0698061964021a1472c3000acb2ff811d4dd2e906d76359ab7c2d0efec696e` |
| `artifacts/branchfabric/overhead/helix-cpu-3seed-2rep-v4/instrumentation-overhead.json` | `0eac7c0173f0311a4f9b461892ba6e8d86ce8f99d9f8ec225f7a945205a0fe12` |
| `artifacts/branchfabric/analysis/cpu-characterization-raw-aggregate.json` | `e55967de3d42710ac40171250c5df1e45b553dd24719c3d072cd3b86a824d197` |
| `artifacts/branchfabric/analysis/cpu-characterization-statistics.json` | `714bef5e91bd3c53f02b66aae50425177aff47bb2f2ad3ceec40f2fdf928d028` |
| `artifacts/branchfabric/analysis/amdahl/cpu-reference.json` | `22eb97b1d19bf31751c700ef7b8f762ac205dc8ef760e737fdb6966df768b9a7` |
| `artifacts/branchfabric/environment/seed-20260809/environment-study.json` | `a8e67a84e2ab57d18219dcf93c01df9d66f92fc3a0a27072940af4f77c695cb6` |
| `artifacts/branchfabric/metadata/seed-20260809/metadata-study.json` | `d46a31b60147dc86427295de4c7de12c9078df44034aafdd41e518cfd1a1db8f` |
| `artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json` | `017b7385ba31cf913582f85f79b20e71976c3e193ab1524c844c4b434621bb9b` |
| `artifacts/branchfabric/requirements/branchfabric_requirements.json` | `c2177db84c6c895cc1c8acf12d33f5ad77c245938817714fffd03a0cd17722a5` |

The reviewed source `HEAD` was
`28d97052021870a174d328bd70c4e88dcc23f6d9`. Uncommitted fixes present during
review are not accepted evidence until the affected artifacts are regenerated
and the tests below pass.

## High-severity findings

### STAT-H1: overhead warmups are assigned to the wrong series and do not precede measurement

The v4 overhead schedule randomized warmups together with measurements. The
full and minimal warmups ran at order 4 and 5, and the disabled warmup ran at
order 18; measured trials already ran at orders 0 through 3. A warmup executed
after a measured trial cannot warm that trial.

The derived per-level series then copied all three warmups into every trace
level. Each level consequently reports `warmup_count=3` and provenance
`sample_count=9`, even though its selector identifies one warmup plus six
measurements. The six post-warmup measurement values themselves were selected
correctly, so their medians are numerically unchanged; the warmup policy and
provenance are not correct.

The uncommitted source seen during review filters warmups by level and makes the
schedule warmup-first. That is the right direction, but v4 remains invalid for
warmup/noise-floor acceptance until a new raw artifact is produced.

Required resolution:

1. Execute every declared warmup before every measured trial.
2. Retain randomized order within warmup and measurement phases.
3. Filter series warmups by trace level and make selector counts exact.
4. Regenerate overhead raw samples using the exact canonical recorder topology.
5. Preserve v4 as invalidated evidence rather than overwriting it.

Acceptance tests:

```text
uv run --locked pytest -q tests/python/test_branchfabric_overhead.py
```

Add assertions that, for every level, all warmup `order_index` values are less
than the global minimum measured `order_index`, and that
`provenance.sample_count == level_warmups + level_measurements`. The regenerated
artifact must report zero canonical drops and semantic equivalence.

### STAT-H2: workload/timing evidence axes are collapsed in statistical and requirement provenance

The methodology requires workload provenance and timing provenance to remain
independent. Canonical events do this correctly. `ArtifactEvidence` in the
statistics layer and `EvidenceReference` in the requirements layer retain only
one `evidence_class` field. In the CPU aggregate, every raw series has
`evidence_class=SYNTHETIC`, while the document-level timing field says
`HARDWARE_BACKED_REAL except explicitly named simulated series`. The two
simulated migration series are distinguishable by their names, not a typed
per-series timing field.

This is a machine-readable mixing hazard. `SYNTHETIC` describes the fixture;
`HARDWARE_BACKED_REAL` or `SIMULATED_HARDWARE` describes how a timing was
obtained. One cannot substitute for the other, and rationale text is not a
stable discriminator for hardware architecture tools.

Required resolution: add independent `workload_evidence_class` and
`timing_measurement_class` fields to every statistics/requirements evidence
record. Reject aggregation across timing classes unless the result is explicitly
stratified. Regenerate statistics and requirements.

Acceptance tests:

```text
uv run --locked pytest -q \
  tests/python/test_branchfabric_statistics.py \
  tests/python/test_branchfabric_requirements.py
```

Add a negative test proving that a host-timed series and a simulated-transport
series cannot enter one unstratified distribution, even when their workload
class is the same.

### STAT-H3: p95/p99 hardware-requirement fields are mostly single-fixture point observations

The reviewed requirements artifact contains 115 `AVAILABLE` numeric entries
whose evidence sample count is at most three. Examples include:

- branch fanout p50/p95/p99/max from one branch group (`n=1`);
- COW amplification p50/p95/p99/max from one sharing analysis (`n=1`);
- transaction commit and abort-rate p50/p95/p99/max from one observation
  window (`n=1`);
- queue minimum depths from one serialized trace for several operations
  (`n=1`);
- operation size, fanout, and concurrency tails from one to four fixture
  events.

Type 7 interpolation is mathematically defined for these samples, but calling
the result `AVAILABLE` in a hardware-requirements distribution gives it tail
meaning it does not possess. In particular, observed maximum concurrency one
is a lower bound from a serialized fixture, not the minimum command-queue depth
needed by Helix.

Required resolution:

- Keep point observations available as point observations.
- Mark workload p95/p99 and hardware sizing targets `UNKNOWN` until the
  independent experiment count satisfies a declared tail-eligibility policy.
- Rename observed concurrency one to an `observed_lower_bound`, or keep queue
  minimum/recommended/pathological depth unknown.
- Never turn an analytical matrix projection into empirical tail evidence.

Acceptance tests:

```text
uv run --locked pytest -q tests/python/test_branchfabric_requirements.py
```

Add a compiler invariant that forbids p95/p99 hardware-sizing fields from being
`AVAILABLE` when their evidence has fewer than the declared number of
independent experimental units. A single-group fixture must compile as a point
observation plus unknown tail, not four identical percentile claims.

### STAT-H4: sibling branches are treated as independent samples in uncertainty estimates

`branch_lifetime_ns` and `fork_to_first_private_ns` contain 12 values, but they
come from four sibling branches in each of only three seed runs. Siblings share
the same parent, fork span, reward/training schedule, host conditions, and much
of the lifecycle. They are not 12 independent experimental replicates.

The branch-lifetime group means make the clustering visible:

| Seed/group | Four-sibling mean |
|---|---:|
| 41 | 384.764 ms |
| 73 | 408.682 ms |
| 113 | 405.893 ms |

The ordinary sample bootstrap in
`cpu-characterization-statistics.json` resamples the 12 branches individually
and reports a median interval of about 385.165--408.221 ms. That procedure does
not preserve branch-group correlation. Descriptive branch-event percentiles are
fine if labeled as such; inference about workload/run variability has an
effective independent run count of three.

Required resolution: record an observation key containing run ID, seed,
repetition, branch group, and branch. Use cluster bootstrap at the run/branch-
group level for group-generalized uncertainty, or publish descriptive event
distributions without an inferential CI. Report both the number of events and
the number of independent clusters.

Acceptance tests:

```text
uv run --locked pytest -q tests/python/test_branchfabric_statistics.py
```

Add a clustered synthetic test where within-group values are identical and
prove that the cluster interval is based on groups, not inflated event count.
Requirements must expose `sample_count` and `independent_cluster_count` for
branch distributions.

### STAT-H5: the controlled matrix has no sampling weights and cannot establish a workload distribution

The declared matrix spans 855 one-factor controlled cells. In the reviewed run
it was validated rather than executed. An analytical evaluation of all cells,
if added, is still a sensitivity/design-space study: each cell was selected by
the experimenter and has no production occurrence probability. Counting cells
would overweight dimensions with more enumerated levels and manufacture a
workload distribution.

The measured realistic slice remains one deterministic fanout-four, immediate-
divergence coding-agent group. Pure reasoning, verification-heavy, capacity
reclamation, late divergence, long context, and high fanout have no empirical
frequency in the current corpus.

Required resolution: keep controlled matrix outputs stratified by factor and
set `distribution_claim=false`. Do not pool cells into workload percentiles.
Obtain workload-distribution evidence from real/replayed trace sampling with a
documented sampling frame; controlled sweeps may then estimate conditional
response surfaces.

Acceptance tests:

- The matrix artifact must report separately: declared cells, analytically
  evaluated cells, actually executed cells, and failed/skipped cells.
- A schema test must reject a production-distribution claim for controlled
  matrix frequencies.
- Requirements must not use the 855 cell count as an experiment or workload
  sample count.

## Medium-severity findings

### STAT-M1: the three-seed CPU aggregate does not preserve observation identity or randomized run order

The aggregate contains one Helix timing per seed and one Continuum timing per
seed (`n=3` each), with no discarded warmup and no repeated trial per seed. Its
global `source_artifacts` list is hashed, but each value lacks a row-level
mapping to run ID, seed, repetition, execution order, machine state, and source
artifact. `ArtifactEvidence.seed=20260809` is the analysis seed, not the three
measurement seeds.

This prevents a consumer from performing seed-stratified analysis or checking
time/order drift without reconstructing the aggregate manually. Per-run
background load and hardware-clock/thermal state are also absent.

Required resolution: represent observations as keyed rows rather than bare
vectors, preserve realized run order, and record per-trial source revision,
machine/topology manifest, clocks/thermal state when available, and background
load. Add repetitions or explicitly classify the three-seed timings as
exploratory point observations.

### STAT-M2: small-sample tail estimates are reported uniformly without an eligibility policy

Independent sample counts are three for most cross-seed operations, three per
environment cell, six paired measurements per overhead level, and seven per
metadata/software-baseline cell. P95 and p99 at those sizes interpolate near
the maximum; they do not characterize rare latency tails. The methodology
acknowledges this limitation but the schemas and reports do not prevent those
numbers from flowing into hardware requirements.

Required resolution: define a tail-eligibility policy based on independent
units and the intended quantile. Below that threshold, report median, range,
ECDF, and explicit `TAIL_INSUFFICIENT` status. Do not hide the raw type 7 value,
but do not use it for sizing or SLO claims.

### STAT-M3: overhead effect estimates have no interval on the paired difference

For v4, disabled and full wall medians are 11.2111 s and 11.2208 s. Across six
matched pairs, the median difference is -14.14 ms (-0.126%), the mean difference
is +46.04 ms, and paired Cohen's dz is 0.332. The point estimates disagree in
sign and the disabled series contains one retained low outlier. The result is
consistent with an effect inside the measured noise floor, but no confidence
interval or randomization interval is reported for the paired overhead itself.

Required resolution: after correcting STAT-H1, report the full paired
difference vector and a deterministic paired bootstrap or exact sign/
randomization interval. Interpret it as “no resolved wall-time effect at this
sample size,” not a speedup and not a proven zero-overhead bound. CPU-time,
memory, storage, and lifecycle metrics need their own paired analyses rather
than inference from wall time.

### STAT-M4: the statistical methodology is not applied uniformly to claim-bearing studies

The generic statistics module supports bootstrap intervals, ECDFs, histograms,
noise floors, MAD flags, and paired effects. The CPU aggregate uses summaries,
ECDFs, histograms, and median intervals but does not emit noise/outlier reports.
Metadata emits seven-sample p95/p99 summaries and “none removed,” but no
confidence interval or flag record. Environment preserves three samples per
cell but has no standard derived statistical artifact. Software baselines emit
relative MAD and no intervals/effect sizes for selected winners.

Required resolution: create one common analysis pass over every performance
series. It must preserve raw samples and emit applicable descriptive,
uncertainty, noise, and outlier records, plus an explicit reason when a method is
inapplicable. Selection of a “best” software baseline must account for
uncertainty and multiple candidate comparison, not only the smallest median.

### STAT-M5: paired-effect APIs cannot verify that pairs are semantically matched

`paired_effect_size` checks metric, unit, and vector length. `RawSampleSeries`
does not carry per-observation keys, so the function cannot verify seed,
repetition, workload digest, hardware state, or treatment pairing. The overhead
caller currently sorts by seed/repetition and appears matched, but the API would
accept unrelated equal-length vectors.

Required resolution: add typed observation keys and require exact key equality
before paired analysis. Preserve the realized randomized order separately from
the matched comparison order.

### STAT-M6: Amdahl results are single-run sensitivity points, not stable acceleration bounds

Every Amdahl result in the reviewed artifact has `sample_count=1`. Its p50,
p95, and p99 fractions are therefore identical. More importantly, the two
objective denominators do not correspond to the requested complete learning
transaction and isolated migration path, as detailed in
`SYSTEMS_HARDWARE_REVIEW.md` (`CONTINUUM-H1`). The arithmetic is deterministic,
but sampling uncertainty and objective-boundary uncertainty dominate it.

Required resolution: rename objectives to the exact observed causal envelope,
preserve per-run fractions across independent repetitions, and publish
uncertainty on the free-operation speedup. Until then, values such as 1.00345x
and 1.02349x are single-fixture sensitivity points and must not be used as
general upper bounds for Helix or migration.

## Low-severity findings and validated behavior

- The exact ECDF implementation and type 7 percentile implementation are
  internally consistent for the reviewed samples.
- Bootstrap resampling is deterministically seeded and bounded; reproducibility
  is good. Percentile bootstrap coverage at `n=3` is weak by design, not an
  implementation error.
- The disabled wall-time MAD detector flags one observation and removes zero
  samples. Preserving it is correct.
- Metadata and software-baseline schedules are warmup-first and randomized
  within phase. Environment warmups precede its randomized measurement phase.
- Environment fixtures, metadata access streams, fanout microbenchmarks, and
  modeled transfers are clearly described as controlled synthetic inputs.
- Roofline and hardware bandwidth targets correctly remain unknown when the
  counters and ceilings are absent.
- The absence of compatible GPU/multi-node hardware is honestly represented;
  no statistical method can replace those missing measurements.

## Release acceptance criteria

The following are required before calling the characterization statistically
complete:

1. Regenerate a warmup-correct, exact-recorder instrumentation-overhead study
   and invalidate v4 for final quantitative use.
2. Split workload and timing provenance in all statistics and requirements
   evidence records.
3. Bind independent seed/repetition/run/group keys to every observation.
4. Use clustered uncertainty for sibling-derived metrics.
5. Gate p95/p99 and queue/page/bandwidth/latency requirements on adequate
   independent evidence; otherwise emit explicit unknowns.
6. Keep controlled analytical and synthetic matrix cells out of empirical
   workload distributions.
7. Apply the common statistics pipeline to environment, metadata, software
   baselines, lifecycle, and overhead series.
8. Regenerate requirements and reports from one hash-verified corpus manifest.

Minimum focused acceptance command:

```text
uv run --locked pytest -q \
  tests/python/test_branchfabric_statistics.py \
  tests/python/test_branchfabric_overhead.py \
  tests/python/test_branchfabric_requirements.py \
  tests/python/test_branchfabric_workflow.py
```

Then rerun the clean CPU characterization, requirements compiler, report
generator, and clean-room test. A verifier must fail if any claim-bearing
artifact hash differs from the corpus manifest or if any hardware-sizing tail
is derived from an ineligible sample count.

## Final statistical disposition

Approve the corpus for trace-schema validation, deterministic fixture
comparison, software microbaseline exploration, and a conservative no-build
decision. Do **not** approve it as an empirical production workload model or as
the basis for fixed BranchFabric page, queue, memory, bandwidth, latency, or ISA
requirements until the high-severity findings are resolved.
