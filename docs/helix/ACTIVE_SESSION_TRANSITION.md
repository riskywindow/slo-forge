# Active-session transition

Promotion changes routing for new work without rewriting the policy identity of an in-flight session.
`PolicyRegistry` classifies sessions as direct, recompute, request-boundary, champion-pinned, or
incompatible. Request-boundary and incompatible sessions stay pinned to the champion on which they
started; compatible state reuse still needs Continuum evidence.

Champion routing is updated through a transactional compare-and-swap. A fault after pointer update
rolls the SQLite transaction back, leaving both the prior champion and promotion state intact. This
tests atomic local behavior, not distributed linearizability across multiple registries.

See ADR 0048 and [promotion and rollback](PROMOTION_AND_ROLLBACK.md).
