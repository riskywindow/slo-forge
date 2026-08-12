# BranchFabric Characterization Methodology

## Measurement principles

The characterization process is evidence first. It preserves raw observations,
keeps workload provenance separate from timer provenance, and derives hardware
requirements only from cited artifacts. A missing measurement remains unknown
or unavailable; it is never filled with a nominal value.

The authoritative statistical implementation is
`sloforge.helix.characterization.analysis.statistics`. The shorter policy
summary in `docs/branchfabric/STATISTICAL_METHODOLOGY.md` is consistent with this
document. Trace field and workload definitions are in
`TRACE_SPECIFICATION.md` and `WORKLOAD_SPECIFICATION.md`.

## Baseline and experiment identity

Before instrumentation changes, record an immutable baseline commit, machine
topology, hardware/software manifests, existing fast-test result, and an
untraced Helix CPU demo. Each later experiment records:

- source experiment and source commit;
- explicit seed and repetition;
- workload and timing evidence classes;
- hardware and software manifests;
- trace level and sampling configuration;
- timeout and realized run order;
- raw artifact path, byte length, SHA-256, and sample selector.

The orchestration run copies immutable inputs, hashes matrix expansion, and
writes append-only attempt checkpoints and status snapshots. Resume does not
delete or overwrite evidence from an interrupted or failed attempt. Stage
workers are bounded by configured timeouts and attempt limits.

## Evidence classification

Use the following classification before analyzing a value:

1. Identify whether the workload was `SYNTHETIC`, `REPLAYED`,
   `HARDWARE_BACKED_REAL`, or `SIMULATED_HARDWARE`.
2. Independently identify how its timing or movement quantity was obtained.
3. Verify that the manifest contains the device, topology, runtime, and clock
   needed for the claim.
4. Preserve the label through aggregation and derived requirements.

A host-timed synthetic fixture supports a claim about that software operation
on the recorded host. It does not support a production workload distribution,
GPU latency, HBM allocation, PCIe bandwidth, or network bandwidth. A simulated
transport delay does not become hardware-backed because capture and checksum
operations in the same trace were host-timed.

## Controlled experiment design

The machine-readable matrix performs one-factor sweeps from a declared base
case. It uses multiple explicit seeds and repetitions and randomizes expanded
case order with an explicit run-order seed. The realized order, including
warmups and interrupted attempts, is retained.

Warmup is declared before execution. It is represented by one of:

- `none` with zero warmup samples;
- `fixed_count` with the declared count;
- `external_phase_marker` when the workload supplies a phase boundary.

Warmup samples remain in the raw series and are excluded only from derived
summaries. The policy and rationale are part of the series. Runs must not decide
after inspecting measurements which early samples to relabel as warmup.

Repeated-trial comparisons use matched seed and repetition where possible.
Changing trace level, software baseline, page size, chunk size, or another
treatment must not silently change the semantic workload. Experiment IDs and
raw selectors make each comparison auditable.

## Instrumentation validation

Instrumentation is accepted only after these checks:

- the same explicit seed produces equivalent Helix semantic output with tracing
  disabled, minimal, and full;
- all compared trials use the same source commit;
- event models and JSON schemas accept every record;
- content hashes and strictly increasing per-trace sequences validate;
- manifest accounting balances attempted, accepted, dropped, and filtered
  events;
- clock source and alignment confidence are present;
- bounded-buffer overflow and resource-sampler drops are visible;
- trace persistence occurs outside the timed hot-path span when that is the
  declared measurement boundary.

The overhead schedule is randomized across disabled, minimal, and full levels.
Its raw record preserves warmup flags, order indices, seeds, repetitions,
semantic digests, source commits, wall time, CPU time, branch readiness,
throughput, sampled memory, resource-sampler loss, and storage-related values
when available. A source-commit change or semantic mismatch invalidates the
cross-level analysis; the raw trials remain preserved and are marked invalid
rather than rewritten.

Full-trace overhead is interpreted against the measured noise floor. A small
signed difference is not called a speedup or slowdown merely because its point
estimate has that sign. Missing TTFT, token-latency, GPU-utilization, or learning
transaction measurements remain explicit limitations.

## Raw sample preservation

Raw artifacts are immutable inputs to analysis. `ArtifactEvidence` records the
experiment, artifact path and SHA-256, evidence class, selector, sample count,
seed, repetition, and `raw_artifact_immutable: true`. `RawSampleSeries` preserves
the original sample order and the warmup/measurement boundary.

Analysis code must not:

- edit, truncate, reorder, or replace raw benchmark files;
- silently delete outliers;
- merge incompatible evidence classes;
- substitute an analytical estimate for a missing counter;
- treat trace-level filtering or dropped events as zero activity.

Corrections produce a new artifact. An invalid analysis is retained with an
explicit invalidation record so later reviewers can reconstruct why it was not
used.

## Descriptive statistics

Every performance series reports at least sample count, minimum, mean, median,
p90, p95, p99, and maximum. Percentiles use Hyndman--Fan type 7 interpolation.
P95 and p99 from small samples remain mathematically defined but must be
presented with their sample count; they are not a substitute for adequate tail
coverage.

Empirical CDFs use every measurement sample. They are exact over unique values
when that fits the declared point limit; otherwise a deterministic rank grid is
used and the output is marked non-exact. Equal-width histograms reject caller
bounds that would omit a sample. Joint and conditional analyses retain both
selectors, such as COW amplification conditional on page size or dirty bytes
conditional on branch age.

The implementation bounds a raw series to one million values, ECDF output to
100,000 points, histograms to 4,096 bins, bootstrap repetitions to 100,000, and
total bootstrap draws to 20 million. Bulk traces remain in streaming formats
rather than being forced into an in-memory sample series.

## Confidence intervals and effect sizes

Confidence intervals use a deterministic percentile bootstrap with an explicit
seed, confidence level, repetition count, and statistic. The default analysis
uses 2,000 bootstrap repetitions where the draw bound permits it. The output
records the observed statistic and Hyndman--Fan type 7 interval endpoints.

Paired comparisons are candidate minus baseline and preserve pair order. They
report:

- mean and median paired difference;
- paired Cohen's dz when variance permits;
- matched-pairs rank-biserial effect when nonzero differences permit;
- median relative change using the absolute baseline magnitude;
- the number of usable relative pairs and zero-baseline pairs.

A zero baseline makes relative change undefined for that pair; it is counted,
not coerced. A confidence interval or effect size does not repair a mismatched
workload, a changed source commit, or a synthetic-to-real evidence mismatch.

## Noise and outliers

Noise-floor analysis uses all post-warmup samples and reports the median,
median absolute deviation, p95 absolute deviation, population standard
deviation, coefficient of variation when defined, and relative deviations when
their denominators are nonzero.

MAD analysis flags sample indices and values using a modified z-score. When MAD
is zero, non-median values are flagged without inventing infinite scores. The
primary report removes zero samples. Outliers are evidence that may indicate
scheduling, thermal state, contention, paging, or a fault. If a sensitivity
analysis excluding a point is scientifically useful, it must be an additional
derived artifact with its own method and provenance, never a replacement for the
primary result.

## Correlation and conditional distributions

Paired quantities may report Pearson correlation and tie-aware Spearman
correlation. Constant-series correlation is `null`, not zero. Pair order and
both evidence references are retained. Correlation does not establish causality
or validate a hardware-placement recommendation.

Important conditional distributions include fanout by prefix length, dirty rate
by branch age, COW amplification by page size, migration bytes by divergence,
and transform sequence by physical layout. If the controlled matrix changes one
factor at a time, cross-factor interactions require dedicated cases; they must
not be inferred from unrelated marginal sweeps.

## State and lifecycle accounting

Model and environment state are analyzed separately. Logical bytes, physical
bytes, content-addressed unique bytes, filesystem workspace bytes, metadata,
transferred bytes, and compressed bytes are distinct accounting domains. A CAS
deduplication result does not prove filesystem block sharing, and simulated KV
bytes do not prove GPU HBM allocation.

Branch lifetime is measured between an observed fork and terminal event.
Divergence requires an observed divergence marker, token/action evidence, or a
declared synthetic projection. COW analysis distinguishes logical writes,
allocation, copied page bytes, page faults, and internal fragmentation. If the
backend exposes whole-file replacement but no physical page faults, page-fault
availability is false rather than an observed count of zero.

State-operation dependencies, shared timing spans, and overlapping transfers
are considered before summing latency. Waterfall stages use the union of
deduplicated time intervals where multiple events share a span. Separate trace
origins cannot be combined into a single observed waterfall.

## Resource and hardware measurements

Machine topology and software versions are captured beside traces. Resource
samplers report their bounded capacity and dropped-sample count. Hardware
counters are used only when the relevant device and compatible tool are present.
Unavailable GPU, multi-GPU, multi-node, NVLink, PCIe, RDMA, or profiler paths
produce explicit status artifacts and exact future commands; they do not produce
placeholder performance values.

Paid resources require the characterization budget environment variable and a
cost estimate before execution. In its absence, the methodology permits CPU
measurements and deterministic simulation only. Synthetic network, CXL, and GPU
state projections remain stratified from hardware-backed data.

## Derived analysis and requirements

Amdahl, roofline-style, placement, and requirement analyses cite hashed source
artifacts and carry sample count and evidence classification. Amdahl bounds use
the measured fraction of the relevant end-to-end path and report 2x, 5x, 10x,
and effectively-free counterfactuals. A zero or unknown fraction is not replaced
with an assumed benefit.

Roofline classification requires measured or explicitly modeled bandwidth and
compute ceilings. Missing GPU, PCIe, or network ceilings leave the corresponding
classification unknown. Software baselines are measured with the same workload
and evidence rules before hardware leverage is claimed.

Every value in the requirements artifact must identify:

- source experiment;
- artifact reference and SHA-256;
- sample count;
- synthetic, replayed, simulated, hardware-backed, unavailable, or unknown
  designation as supported by its schema;
- percentile or confidence information where applicable.

Candidate operations unsupported by the corpus are classified as uncertain,
not currently justified, or software-only. Absence of an operation in a lossy
trace cannot justify rejection until trace completeness is established.

## Review and reproducibility

Before a characterization claim is accepted:

1. Re-run trace schema, integrity, and Rust/Python conformance tests.
2. Reproduce the CPU characterization from declared inputs and seeds.
3. Verify trace manifests and analysis evidence hashes.
4. Confirm instrumentation semantic equivalence and quantify overhead.
5. Search analysis and requirement generators for hard-coded benchmark values.
6. Independently replicate high-leverage measurements.
7. Review negative findings and hardware recommendations against the raw corpus.
8. Confirm that no raw artifact was altered and no unavailable hardware path was
   reported as measured.

Reproducible commands must identify the matrix, run directory, seed, timeout,
and expected outputs. Final reports distinguish executed cells from planned
cells and list all unavailable paths and unresolved uncertainty.

## Known methodological limits

Controlled CPU fixtures establish trace correctness and host-software behavior,
not production distributions. Simulated state sizes establish analytical
sensitivity, not GPU memory requirements. One-factor sweeps do not identify all
interactions. Small sample counts provide weak tail estimates. Local unified
memory does not substitute for discrete GPU HBM and PCIe. In-process or modeled
transport does not substitute for multi-node network measurement.

These limits are part of the evidence, not caveats to remove from downstream
reports. A future hardware-backed run should add evidence without rewriting or
reclassifying the CPU corpus.
