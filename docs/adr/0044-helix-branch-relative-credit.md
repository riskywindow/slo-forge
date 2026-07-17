# ADR 0044: Assign credit relative to sibling counterfactual branches

## Context

Terminal reward copied to every action ignores which intervention differed and encourages spurious updates.

## Decision

Compare outcomes inside one validated branch group, compute baseline-relative advantages, emit explicit
exclusions and pairwise preferences, and bind credit to branch, trajectory, action, reward, and replay evidence.

## Consequences

The update is structurally tied to controlled alternatives. It is not automatic causal identification;
unrecorded branch differences can still confound the comparison.
