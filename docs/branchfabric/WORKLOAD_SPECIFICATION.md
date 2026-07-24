# Helix Characterization Workload Specification

## Purpose

The characterization workload plan is a controlled experiment design for
measuring Helix and Continuum state behavior. Its machine-readable source is
`benchmarks/branchfabric/characterization.yaml`, schema version
`sloforge.branchfabric.characterization-matrix/v1`.

The matrix explicitly declares `distribution_claim: false`. Its values cover
design parameters and negative cases; their frequency in the matrix is not a
claim about production frequency. No matrix cell becomes representative merely
because it was generated or executed. Empirical workload weights must come from
a separately identified trace corpus.

## Evidence classes

Every experiment and trace uses one explicit evidence class:

| Class | Meaning | Prohibited interpretation |
| --- | --- | --- |
| `SYNTHETIC` | A deterministic fixture, projection, or controlled parameter sweep created for characterization. | Production workload distribution or real GPU-state behavior. |
| `REPLAYED` | Operations driven by a previously captured workload artifact. | A new live workload observation; replay timing must still be classified independently. |
| `HARDWARE_BACKED_REAL` | The declared workload and operation executed on the hardware recorded in its manifest, without modeled replacement for the claimed quantity. | Results for an absent device, topology, transport, or runtime. |
| `SIMULATED_HARDWARE` | Hardware behavior, latency, or movement came from a declared model or simulator. | A hardware-backed measurement. |

Workload provenance and timing provenance are independent. A synthetic Helix
fixture can exercise real software and yield real local CPU elapsed time without
becoming a real production workload. Likewise, a real captured workload replay
can contain simulated network timing. Reports must retain both labels.

The matrix validator prevents a case with simulated GPU state from claiming
`HARDWARE_BACKED_REAL` workload evidence. It also prevents a local CPU run from
being labeled as simulated hardware. Those checks supplement, rather than
replace, artifact review.

## Workload classes

The matrix defines five named classes:

| Class | Intended characterization role | Required evidence boundary |
| --- | --- | --- |
| `coding_agent` | Long-context branching with environment state, tool use, pruning, reward, and learning boundaries. | The current CPU harness is a deterministic fixture that calls the real Helix CPU demo path; its workload remains `SYNTHETIC`, model GPU state remains simulated, and host timing is classified separately. |
| `pure_reasoning` | A high-fanout case with little or no environment state, intended to isolate model-state sharing. | The label is not evidence of this behavior unless the executed case records the corresponding environment/tool parameters and trace. |
| `verification_heavy` | A pruning- and verifier-focused case intended to expose short-lived branches and environment reuse. | Verifier frequency and pruning behavior must be observed in events; the class name alone is insufficient. |
| `capacity_reclamation` | A checkpoint, pause, migration, recovery, or resume case intended to isolate state-lifecycle movement. | Migration and capacity-reclamation claims require the Continuum operation trace or another explicit migration artifact, not a Helix run that did not migrate. |
| `highly_divergent_negative` | A negative-control case in which branch sharing is expected to have less opportunity. | It must remain in summaries even if it weakens a proposed acceleration case. |

In the current matrix, the `workload-class` sweep changes the class field only;
it does not apply hidden class-specific overrides. Therefore class-specific
behavior must be supplied by an executed fixture or by explicit values in other
dimensions. Analysis must not infer tool rate, environment state, divergence,
or branch lifetime from the class label.

## Controlled dimensions

All sweeps are one-factor changes from the declared base case unless a sweep
has an explicit override. The current dimensions are:

| Dimension | Values |
| --- | --- |
| Branch fanout | 1, 2, 4, 8, 16, 32, 64, 128 |
| Prefix tokens | 1,024; 4,096; 8,192; 16,384; 32,768; 65,536; 131,072 |
| Suffix tokens | 16, 64, 256, 1,024, 4,096 |
| Divergence | immediate, early, mid, late, highly shared, highly divergent |
| Branch survival | all complete, aggressive early pruning, reward pruning, random death, long tail |
| Environment size | none, tiny, medium, large, dependency heavy, service heavy |
| Tool behavior | none, frequent cheap, infrequent expensive, build heavy, test heavy, service heavy |
| Policy mode | same policy, cross policy, strict, segmented |
| Resource pressure | idle, moderate serving, heavy serving, rollout saturation, CPU environment saturation, network contention, storage contention |
| Migration | none, branch migration, capacity reclamation, checkpoint/resume, worker-failure recovery |
| Page size | 4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB, 2 MiB |
| Trace level | disabled, minimal, full |
| Fault | none, branch abort, worker failure, transfer retry, storage slowdown |
| Workload class | the five classes above |
| Tensor parallelism | 1, 2, 4, 8 |
| State packing | separate KV, packed KV, interleaved KV |
| Memory tier | host DRAM, local NVMe, simulated CXL |
| Transport | in process, local file, simulated network |

The migration sweep explicitly overrides the workload class to
`capacity_reclamation`. Other dimensions retain the base case unless their
sweep changes them.

## Base case

The base declaration is a synthetic, CPU-local controlled case:

- `coding_agent`, fanout 8;
- 8,192 prefix tokens and 256 suffix tokens;
- mid divergence with reward pruning;
- medium environment and test-heavy tool behavior;
- same-policy siblings under idle pressure;
- no migration or injected fault;
- 64 KiB pages, tensor parallelism 1, separate KV packing;
- `continuum-reference-token-major` runtime on CPU-local host DRAM;
- in-process transport, full tracing, simulated GPU state;
- 120-second per-case timeout and zero estimated dollar cost.

These values are controls, not observed percentiles.

## Expansion, identity, and run order

The current matrix uses seeds 41, 73, and 113 and three repetitions per cell.
The trace-level sweep explicitly uses seven repetitions per level. With the
current values, strict expansion produces 855 experiment cases.

Each case identity includes the matrix ID, sweep, value ordinal, seed,
repetition, and full validated experiment specification. Its ID contains a
SHA-256-derived suffix. Expected outputs reside under:

```text
artifacts/branchfabric/characterization/<run-id>/experiments/<experiment-id>
```

After expansion, case order is shuffled by a local deterministic PRNG using
`run_order_seed: 20260809`. This makes the planned order reproducible while
reducing systematic association between time and sweep value. A runner must
preserve the realized schedule because interruption and resumption can make the
executed subset differ from the plan.

The loader rejects matrices larger than 4 MiB and expansions larger than 100,000
cases. Fanout, token counts, page size, tensor parallelism, timeout, and cost are
bounded by the strict experiment model.

## Divergence projections

The controlled COW projection maps divergence labels to a declared common
suffix length:

- immediate and highly divergent: zero common suffix tokens;
- early: one eighth of the suffix, using integer division;
- mid: one half;
- late: seven eighths;
- highly shared: all but the final suffix token.

This mapping is a synthetic input. It is not an empirical divergence
distribution. Projection output remains `SYNTHETIC` and retains the calibration
artifact, its SHA-256, and every scaling assumption. The current KV projection
models append-only fixture KV and page rounding; it does not measure GPU HBM,
filesystem COW, page-fault latency, or production metadata size.

## Environment fixtures

Environment-state characterization uses deterministic synthetic repositories
against the real Helix environment backend. Fixture categories are:

- tiny: four 256-byte generated files;
- medium: 48 generated files of 2 KiB;
- large: 128 generated files of 8 KiB;
- dependency heavy: 96 1-KiB files plus four dependency lockfiles;
- service heavy: 24 1-KiB files, one lockfile, a 128-row SQLite fixture, and a
  process reconstruction recipe.

The fixture payload is bounded to 4 MiB and 512 files. Environment traces keep
filesystem, database, service, and process-reconstruction metadata separate.
The fixture manifest declares `distribution_claim: false`. Real local operation
timings, hashes, and CAS bytes do not convert the synthetic repository shape
into a production distribution.

## Realistic Helix vertical workload

The coding-agent vertical harness wraps the actual operations invoked by
`sloforge.helix.demo.run_cpu_demo`; it does not implement a separate simplified
control path. The observed boundaries include capture, branch/environment fork,
rollout, reward, training, transaction, promotion, pruning, checkpoint, and
completion when the underlying demo executes them.

This is still a deterministic synthetic CPU fixture. Its output must declare:

- `workload_evidence_class: SYNTHETIC`;
- `timing_measurement_class: HARDWARE_BACKED_REAL` for locally timed host work;
- `simulated_gpu_state: true` for fixture model-state sizes;
- separate limitations when migration, live filesystem COW, GPU execution, or
  another requested behavior was not exercised.

Continuum migration is a separate controlled state-lifecycle workload. Analyses
may compare or compose its evidence only while retaining distinct trace IDs,
time origins, and provenance. They must not present separate Helix and Continuum
runs as one observed end-to-end transaction.

## Execution status is not implied by the matrix

A validated or expanded matrix is a plan. It does not prove that any cell ran.
An executed case requires a terminal orchestration checkpoint and hashed raw
artifacts. A skipped hardware stage is evidence of unavailability, not a zero
latency or zero-byte measurement. An interrupted run keeps its prior attempt
artifacts and resumes append-only; reports must identify the accepted attempt.

No synthetic, replayed, and hardware-backed cells may be pooled into a single
distribution without explicit stratification. Scaling beyond an observed cell
requires a stated model and remains synthetic.
