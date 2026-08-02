# Counterexample minimization

Genesis uses bounded delta debugging for ordered protocol traces. The original trace is executed
first; minimization refuses a trace that does not reproduce the same violated contract. Each trial
removes a deterministic, seed-ordered chunk, renumbers logical event steps, and executes the target
again. A reduction is retained only when the same contract still fails.

The minimizer is bounded by both an evaluation count and a monotonic-clock timeout. It preserves
event order and never manufactures new events. The runner records the number of evaluations and
stores both the original and minimized case. The resulting Genesis counterexample is marked
`minimized` only after the minimized trace has executed successfully against the failure predicate.

Delta debugging finds a locally irreducible trace under its deletion strategy; it does not claim a
globally smallest semantic witness. Equal seeds, inputs, evaluator behavior, and limits give equal
results when the timeout is not exhausted. A timeout leaves the smallest reproducing trace found so
far, rather than discarding valid evidence.

For the unsafe streaming fixture, the initial schedule includes unrelated admissions, decoding,
worker failure, and scheduling events. The minimized schedule is:

```text
admit(r0) -> emit(r0) -> cancel(r0) -> retry(r0)
```

This trace preserves `stream.retry_after_visible_output`: cancellation frees state after committed
output and retry subsequently attempts to use it. The generated reusable precondition is:

```text
retry requires externally_visible == false or retained_idempotent_state == true
```

The precondition and counterexample identity are stored together in the regression corpus so later
synthesis can reject the same transformation family and replay the witness.
