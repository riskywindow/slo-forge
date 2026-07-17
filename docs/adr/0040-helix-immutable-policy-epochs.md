# ADR 0040: Give every behavior policy an immutable epoch

## Context

A mutable "current policy" label cannot establish which distribution produced an action or whether an
old sample is valid for a new optimizer step.

## Decision

Bind events, tokens, actions, log probabilities, rewards, batches, candidates, and sessions to immutable
policy epoch identities and content lineage. Keep deployment champion routing as a separate mutable pointer.

## Consequences

Silent policy mixing is detectable and rollback retains an exact parent. More versions must be stored,
and epoch labels require independently verified hashes to establish actual weight identity.
