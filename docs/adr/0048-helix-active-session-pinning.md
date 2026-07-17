# ADR 0048: Preserve active-session policy identity across promotion

## Context

Changing a policy pointer mid-session can invalidate cached state, behavior provenance, and output semantics.

## Decision

Classify active sessions by Continuum compatibility. Route new sessions to the new champion only after
promotion; keep request-boundary or incompatible sessions pinned. Update the champion with transactional
compare-and-swap and retain the rollback parent.

## Consequences

In-flight work is not silently reinterpreted. Old policies and state may live longer. SQLite demonstrates
local atomicity only; a multi-node registry requires linearizable coordination.
