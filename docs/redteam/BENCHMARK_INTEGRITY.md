# Benchmark-integrity adversary

Performance evidence is accepted only when the baseline and candidate manifests describe a fair,
comparable experiment. `audit_benchmark_integrity` reports every detected issue in stable order; a
faster point estimate cannot suppress an integrity failure.

The auditor checks:

- synchronization of timed work and misuse of wall-clock or unsynchronized device timing;
- identical, nonzero warmup policies;
- identical ordered input fingerprints;
- identical cache regimes and cache reset between trials;
- hidden fallback invocation;
- required precision and quality-contract identity/minimum score;
- omission of failures from raw evidence;
- hardware clock changes beyond the declared tolerance;
- CPU affinity changes;
- unallowlisted background-process differences;
- discarded samples, including preferential removal of slow samples; and
- benchmark-definition hash mismatch.

Both manifests retain all raw samples, discarded indices, workload fingerprints, quality identity,
execution controls, and environment controls. The auditor does not rewrite raw evidence. Each issue
becomes a typed benchmark adversarial case, a canonical Genesis counterexample, and a reusable
precondition such as `benchmark audit must not report hidden_fallback`.

The current auditor verifies manifest consistency; it cannot prove that a malicious measurement
process truthfully populated the manifest. Production evidence therefore still requires a sandboxed
independent harness, artifact provenance, hardware fingerprinting, and signed or content-addressed
raw samples.
