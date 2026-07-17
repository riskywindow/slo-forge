# ADR 0041: Distinguish strict and segmented trajectory semantics

## Context

Long-running work can cross a policy update. Treating it as single-policy corrupts behavior provenance;
rejecting every transition wastes potentially valid evidence.

## Decision

Strict trajectories require exactly one behavior epoch. Segmented trajectories enumerate dense ranges,
transition boundaries, log-probability sources, and state/sampler compatibility. Each segment receives
an explicit eligibility disposition.

## Consequences

Mixed-policy data cannot pass silently, while proven segments remain usable. Segmented validation is
more complex and cannot recover evidence that was never recorded.
