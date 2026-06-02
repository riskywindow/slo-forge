# Autopsy minimization

`minimize_run` implements deterministic delta debugging over a validated
`AutopsyRun` while preserving a caller-supplied diagnosis predicate.

## Reduction order

1. Verify the predicate holds on the source run.
2. Attempt to remove each rank in sorted order.
3. Apply ddmin to remaining events, increasing chunk granularity when no
   complement preserves the predicate.
4. Apply ddmin to event-counter pairs.
5. Repair parent and dependency references after each removal.
6. Revalidate the canonical event graph and predicate.

The result records original/minimized event, rank, and counter counts; removed
IDs; predicate evaluation count; and a SHA-256 over the minimized bundle.

## Predicate design

The predicate must express the regression or diagnosis independently of removed
ground-truth labels. A robust predicate normally rebuilds the comparison and
asserts that the target hypothesis remains above its threshold. A weak predicate
such as "the run parses" can reduce away the causal evidence and is not a valid
performance reproducer.

## Boundaries

The current reducer operates on ranks, events, and counters. It does not yet
rewrite model layers, physical parallelism degrees, or a runtime command. Those
can be represented by an outer reducer that regenerates and validates the run.
All reductions are deterministic, but predicate cost can be high if it performs
counterfactual simulation.

