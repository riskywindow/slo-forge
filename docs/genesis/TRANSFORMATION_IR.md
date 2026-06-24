# Transformation IR v1

The canonical `Transformation` records what a candidate proposes to change and which new obligations it creates. It is a strict, immutable, versioned document; it is not executable generated code.

## Required content

A transformation includes:

- a stable transformation ID and family;
- typed source and target genome patterns with region, node IDs, and structural constraints;
- semantic category and designation;
- preconditions and postconditions;
- bounded expected quality, resource, and performance changes, each with lower/expected/upper values and units;
- affected genome regions;
- proof obligations plus required verifier and benchmark stages;
- rollback strategy and proposal source;
- parent transformations, learned constraints, counterexamples, lineage and extensions.

The supported families cover algebraic rewrite, tensor decomposition, operator fusion, layout, precision, quantization, scheduling, batching, cache policy, state layout, distributed plan, communication, kernel, workflow, recovery, and runtime code patches.

The designation separates semantics-preserving, approximate-within-budget, policy, resource-only, runtime-implementation, and experimental/operator-review changes. Every transformation must declare verification obligations. Approximate changes require an explicit expected quality cost and downstream quality evaluation; floating-point reassociation is not classified as exact.

## Use in synthesis

The IR does not make proposal estimates authoritative. Search uses estimates only for ordering. A candidate records content identity, parent candidates, transformation IDs, a contiguous lifecycle, hard budget, and actual usage. Learned constraints bind an expression to candidate, family, hardware, dependency, or universal-precondition scope and cite the counterexamples that support it.

The cancellation CEGIS fixture demonstrates the intended lifecycle: an unsafe batching policy is proposed, fails an independent request-schedule check, produces a minimized counterexample, teaches the family precondition `cancel_check_before_emit == true`, suppresses a repeated unsafe proposal, and accepts a corrected request/serving candidate within the bounded fixture scope.

## Specialized compilers

The tensor, state, policy, and distributed modules use smaller typed records optimized for their restricted compilers. Those records retain rule IDs, checked preconditions, obligations, quality/resource implications, fallback or rollback information, and deterministic identities. A production integration must translate them to or reference the canonical `Transformation` before capsule construction; a specialized record is not permission to omit canonical lineage or proof obligations.

## Validation and current scope

Python and Rust validate and canonically round-trip the v1 transformation fixture. Schema migration supports only the known alpha aliases. The current repository exercises canonical parsing/conformance and several specialized transformation compilers. It does not expose a universal interpreter that can apply an arbitrary `Transformation` document to any genome, and it does not claim global optimality or semantic correctness from the transformation declaration alone.

