# ServingSynthBench specification

ServingSynthBench evaluates whether a system can turn an unseen, typed reference package into a correct serving implementation. The CPU profile is a reproducible correctness and orchestration benchmark; it is not a proxy for GPU throughput.

## Evaluation boundary

Task creation and task execution are separate operations. `generate_tasks` derives each task from an explicit seed and writes two boundaries:

- `public/` contains the reference implementation, manifest, contracts, sample generator, public workload, and a SHA-256 commitment to the final holdout.
- `evaluator/hidden_cases.jsonl` contains rare shapes, state reset, sampler tie, and burst-priority cases. A submission does not need this directory to inspect or synthesize the public task.

The evaluator verifies both the public package hash and hidden-case commitment before execution. It rejects overwrites, bounds task count and runtime, and requires at least two distinct run seeds. Generated packages are deliberately small and dependency-free so smoke evaluation works without PyTorch, GPUs, cloud credentials, or external synthesis services.

## CPU smoke protocol

For every selected task and run seed, the runner:

1. loads the generated reference entry points;
2. performs the same configured warmup for every applicable baseline;
3. randomizes baseline and repetition order deterministically;
4. records every request with `time.perf_counter_ns` into append-free JSONL artifacts;
5. independently executes evaluator-only cases;
6. audits sample count, request distribution, fingerprints, timer source, and precision;
7. derives the final report by reading the persisted samples back from disk.

A system is valid only when both public requests and hidden cases match exactly. Unsupported CPU baselines remain explicit `not_applicable` records with reasons. The report never converts missing hardware measurements into simulated speedups.

## Scope

The current profile covers zero-day model support, persistent and dynamic state, workload specialization, correctness traps, and deterministic policy search. GPU kernel, energy, multi-device fabric, and hardware-specialization measurements require their separate hardware-backed profiles. CPU results establish correctness and reproducibility only within the generated shape, state, and workload domains.

