# ForgeCI fixture evaluation

This CPU-only evaluation was generated from `artifacts/forgeci/demo/evaluation.json`. It creates and measures
a local four-commit repository; no expected performance samples are embedded in ForgeCI.

| Result | Measured value |
|---|---:|
| First-regression bisection | correct |
| Unique candidate commits evaluated | 3 |
| Inconclusive rate | 0.0% |
| P99 TTFT change | +12.00% |
| P99 TTFT corrected interval | [+11.47%, +12.53%] |
| Noise floor | 0.31% |

Warmups are excluded, trials are repeated, and regression classification requires both
practical significance and a family-wise corrected bootstrap interval excluding zero.
The deterministic fixture is an integration validation, not a claim about external GPU
runtimes or hardware.
