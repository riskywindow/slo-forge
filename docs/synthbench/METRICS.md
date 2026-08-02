# ServingSynthBench metrics and evidence

Raw CPU evidence is one canonical JSON object per request. It records task and workload hashes, environment fingerprint, baseline, run seed, repetition and execution ordinal, request identity, latency, TTFT, inter-token intervals, observed and expected tokens, exactness, timer source, and numerical precision.

The runner reloads those artifacts to calculate:

- sample count and exact-request rate;
- median and p95 request latency;
- median TTFT and inter-token interval;
- measured CPU seconds and candidate count;
- hidden-case exact rate and escaped regressions;
- aggregate valid-system rate and exact-request rate.

The integrity auditor requires the complete request-by-repetition Cartesian product. It flags discarded slow samples, duplicates, workload or environment changes, seed mismatches, altered distributions, non-monotonic timing, precision mismatch, and output disagreement. Different warmup policies are prevented structurally by one run-level configuration.

Latency samples are real local clock observations and are noisy. The CPU smoke report does not assert a statistically significant speedup, cost reduction, energy result, or hardware advantage. Hardware reports must retain their own warmup, raw repetitions, confidence intervals, effect size, noise floor, randomized order, manifest, and simulator-prediction error before making those claims.

