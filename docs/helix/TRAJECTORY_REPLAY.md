# Trajectory replay

The replay engine compares bounded traces in three modes: exact, causal, and semantic. A trace binds
model, environment, policy, sampler, effect, event, token, and resource identities. Exact comparison
requires identity and value equality in the declared scope; causal and semantic modes emit structured
divergences under explicit tolerances.

Resource and event bounds prevent unbounded replay. An exact identity mismatch is an error, not a
reason to downgrade automatically. Evidence records comparison scope, divergences, assumptions, and
hashes. Joint model/environment exactness is distinguished from transcript equality.

Replay cannot reproduce undeclared external services, hidden nondeterminism, or missing source
artifacts. “Causal” mode compares declared event semantics and causal-parent topology; it does not
establish that one event caused another or that two executions are causally equivalent. A semantically
similar outcome is not bitwise equivalence.

The comparator validates the trace identities and evidence supplied to it. It does not independently
attest that a caller-provided runner restored the referenced model bytes or environment capsule. That
claim requires separate restoration evidence from the model and environment backends. See
[limitations](LIMITATIONS.md).
