# Policy epoch

A policy epoch is an immutable policy identity plus behavior semantics and lineage. Rollout tokens,
actions, events, rewards, credits, batches, training results, and promotion records name the epoch
that produced or evaluated them. An epoch identifier is not inferred from wall time or a mutable
"current policy" pointer.

`DeterministicPolicy` supplies the local reference distribution. Canonical Helix IR carries the
portable `PolicyEpoch`; staleness models add `PolicyVersion` and `PolicySemantics` for distance and
compatibility checks. Strict trajectories require one behavior epoch. Segmented trajectories make
each transition boundary explicit and may require log-probability or state recomputation evidence.

Epoch identity prevents silent policy mixing, but does not prove that two artifacts contain the same
weights. That claim depends on their content hashes and independently pinned lineage. See
[policy versioning](POLICY_VERSIONING.md), [staleness](STALENESS.md), and ADR 0040.
