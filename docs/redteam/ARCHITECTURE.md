# Genesis executable red-team architecture

The Genesis red team searches for the smallest executable input or schedule that violates a
candidate's declared contracts. It does not trust synthesis output and does not infer acceptance
from a candidate's own report.

The implementation in `python/sloforge/redteam` has five bounded surfaces:

1. The tensor adversary varies shape, stride, dtype, contiguity, boundary dimensions, and encoded
   numerical edge values including infinities and NaNs.
2. The protocol adversary varies request interleavings, cancellation, disconnect, retry, emission,
   and worker failure schedules.
3. The topology adversary injects failed and degraded links into bounded host/device topologies.
4. The resource adversary probes queue, device-memory, host-memory, and process boundaries.
5. The benchmark auditor compares independent run manifests for measurement manipulation.

Every generator requires a seed and explicit case, shape, or event bounds. The runner has a
wall-clock timeout, a finding limit, and separate minimization evaluation and time limits. It uses
no network, subprocess, device, or credential access. Target integration is a narrow protocol with
independent evaluation methods for each executable case type.

When an evaluator reports a violated contract, the runner minimizes the protocol schedule where
applicable and converts the result to the canonical Genesis `Counterexample` IR. The evidence
records the candidate, transformation, declared scope, reproduction command, environment facts,
expected behavior, and observed behavior. A learned precondition references the counterexample.
The regression corpus contains the minimized cases and constraints under a canonical content hash.
Replaying the corpus executes the target again; it does not replay a stored success flag.

`UnsafeStreamingCandidate` is an intentionally unsafe fast-path fixture. Its state machine eagerly
releases request state on cancellation, then permits retry after externally visible output. The
adversary finds this behavior, delta minimization removes irrelevant events, and replay reproduces
the four-event `admit, emit, cancel, retry` violation. The same fixture exposes real stride, route,
and queue-bound defects. It is test evidence, not an accepted deployment candidate.

Run the deterministic demonstration with:

```console
PYTHONPATH=python python -m sloforge.redteam.demo \
  --seed 73129 \
  --output artifacts/genesis/redteam
```

The output contains a canonical report, a hashed regression corpus, individual canonical
counterexamples, and individual learned constraints.

## Trust boundary

Adversarial generators and minimizers propose cases; they are not proof checkers. Canonical Genesis
IR validation and hashing remain in the trusted IR layer. A finding demonstrates the recorded
behavior only within the explicit case and bounded search scope. Absence of a finding is not a
universal proof. Sandbox isolation is required before replacing the local evaluator with generated
native code.
