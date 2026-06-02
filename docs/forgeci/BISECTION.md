# ForgeCI bisection

`bisect_regression` finds the first likely regressing commit between an ancestor
`good` and reachable `bad` revision.

## Algorithm

1. Resolve both revisions and verify ancestry.
2. Enumerate the ancestry path in oldest-first order.
3. Clone the repository to an isolated temporary directory with no hardlinks.
4. Benchmark the known-good commit and preserve it as the baseline.
5. Select the midpoint of the current interval, checkout detached, build, run,
   and compare.
6. Move the bad boundary on regression; move the good boundary on unchanged or
   improvement.
7. Retry inconclusive/flaky classifications up to the configured bound. Preserve
   an inconclusive list and caveat rather than coercing it into good or bad.
8. Emit every step, comparison path, repetitions, attempts, confidence, and the
   first likely regression.

Build and benchmark commands use process-group cleanup and timeouts. The source
checkout is never reset. Temporary clones are removed after artifacts have been
copied.

## Confidence

Bisection confidence is reduced by inconclusive decisions and failed endpoints.
It is a workflow confidence score based on classified steps, not a probability
that the source diff is semantically causal. Environment changes within the
range, non-monotonic regressions, and build nondeterminism are recorded caveats.

The local fixture result and its SHA-256 sidecar are in
`artifacts/forgeci/demo/bisect/`. The generated upstream issue body is
`artifacts/forgeci/demo/upstream-issue.md`.

