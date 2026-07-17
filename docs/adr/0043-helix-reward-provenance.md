# ADR 0043: Preserve component-level reward provenance

## Context

A scalar reward can be corrupted, duplicated, gamed, or derived from hidden answers exposed to the policy.

## Decision

Bind reward components to trajectory, policy, evaluator version, input/output hashes, sandbox result, and
hidden-test boundary. Recompute aggregates, reject duplicate identities, hash source before/after execution,
and gate promotion on separate integrity evidence.

## Consequences

Reward claims are inspectable and hidden expected values remain verifier-only. Provenance does not make
an incorrect verifier correct, and hash integrity is not issuer authentication.
