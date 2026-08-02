# Counterexample-guided synthesis

Genesis implements an explicit propose, verify, minimize, learn, and correct loop. Generated transformations and proposal heuristics are untrusted. A `CegisVerifier` independently computes acceptance from a typed candidate and witness.

## Loop

For each bounded candidate:

1. check persisted family/precondition constraints;
2. invoke the independent verifier with a deterministic seed;
3. persist the original typed failure;
4. delta-debug the witness while repeatedly invoking the verifier;
5. persist the minimized counterexample with its original parent identity;
6. ask the verifier adapter to generalize a typed precondition;
7. persist that learned constraint;
8. reject the invalid candidate;
9. suppress later candidates that repeat the learned-invalid family condition;
10. verify a corrected candidate independently.

Counterexamples use the canonical Genesis `Counterexample` IR, including candidate and transformation identity, violated contract, scope, request trace, deterministic reproduction command, environment, expected and observed behavior, minimization status, and parent counterexample. Constraints contain both the canonical `LearnedConstraint` and typed family, parameter, and required-value fields; suppression does not parse or execute arbitrary expressions.

The protocol minimizer uses deterministic delta debugging. A reduction is retained only when a fresh verifier invocation reproduces the same violated contract. Evaluation count and maximum trace length are bounded. The fixture also checks one-event minimality of the final trace.

## Executed cancellation fixture

`run_cancellation_cegis(output, seed=73129)` evaluates a synthesized deadline-bucket batching policy. The first candidate has the highest proposal upside but checks cancellation too early. The independent protocol simulator executes:

```text
admit, schedule, prefill, decode, cancel, emit
```

It observes a committed token after cancellation, rejects the candidate, and reduces the real failing schedule to:

```text
admit, cancel, emit
```

The verifier generalizes the family precondition `batching.cancel_check_before_emit == true`. A second unsafe candidate in the same transformation family is suppressed without another full verifier run. A corrected cross-layer request/serving candidate performs the required immediate pre-commit check and is accepted within this bounded protocol scope. Acceptance is computed by simulation; it is not hard-coded and is not a capsule or production-promotion decision.

Artifacts include:

- canonical original and minimized counterexamples;
- canonical learned constraints;
- a contiguous canonical CEGIS event log;
- an executable reproduction command using the same protocol verifier.

Run:

```bash
PYTHONPATH=python pytest -q tests/python/test_genesis_cegis.py
```

This CEGIS surface currently minimizes bounded request schedules. Tensor, topology, dependency, and resource counterexample IRs exist in the trusted core and can be connected through additional verifier adapters; this module does not claim those adapter-specific minimizers are implemented here.

The cancellation failure, delta-debugging, constraint persistence, repeated-family suppression and corrected candidate are exercised in CPU tests and the local synthesis path. Tensor-value, shape/stride, topology, dependency-version and resource-boundary CEGIS loops remain unexercised in this module.
