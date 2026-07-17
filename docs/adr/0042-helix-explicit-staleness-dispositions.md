# ADR 0042: Make staleness evidence and disposition explicit

## Context

Update count alone is a weak proxy for policy drift, and silently filtering stale samples hides bias and cost.

## Decision

Report update/time distances and supplied distribution evidence per segment. Configure bounds and choose
reject, truncate, or resample explicitly. Require recomputation evidence where compatibility demands it,
and retain excluded samples in accounting.

## Consequences

Training admission is reproducible and auditable. Threshold choice remains empirical; the reference
implementation does not claim a universally safe staleness bound.
