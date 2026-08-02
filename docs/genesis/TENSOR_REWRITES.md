# Tensor rewrite system

Genesis represents a restricted tensor graph with typed nodes, topological inputs, symbolic or static shapes, dtype, layout, optional strides, alias group, state dependency, and a numerical contract. Rewrite exploration is deterministic and bounded by candidate count and depth.

## Built-in transformations

The current rule set contains:

- redundant-cast elimination;
- inverse double-transpose elimination;
- integer and contract-qualified floating add-zero elimination;
- multiply-one elimination;
- approximate floating add reassociation;
- state-update fusion;
- approximate int8 output quantization.

Each rule declares its semantic category, dtype domain, maximum quality cost, signed-zero precondition, verification obligations, and fallback requirement. Exact identity rewrites still retain shape, dtype, alias, NaN/infinity, denormal, hardware, state, cancellation, or rollback obligations as appropriate.

Floating-point add-zero is disabled when signed zero must be preserved. Floating reassociation is always approximate, requires an available quality budget, and records differential, quality, exceptional-value, and determinism obligations. Int8 output quantization requires a nonzero output tolerance and separate quality/calibration evidence.

## Exploration

Graph validation rejects non-topological references, invalid shapes/strides, malformed casts/transposes, mismatched add/multiply domains, duplicate IDs, and missing outputs. Applying a rule records matched nodes, checked preconditions, quality cost and obligations. Breadth-first exploration uses structural SHA-256 keys to eliminate duplicates and stops at the declared depth/candidate bounds.

The engine is an explicit rewrite explorer, not an e-graph implementation. It does not execute a candidate or infer performance. A rewrite candidate must still pass operator, quality, resource, end-to-end and capsule gates.

## Exercised and unexercised scope

Tests exercise deterministic exploration, identity elimination, double transpose, rejection of exact floating reassociation, signed-zero preconditions, quality budgets, structural hashing, and invalid graph rejection. The current whole-run local synthesis fixture does not yet lower arbitrary rewrite candidates into generated runtime code. General decomposition, fusion, layout conversion, precision promotion/demotion, sparsity, constant propagation, shape specialization, solver-backed symbolic constraints, and GPU code generation are not implemented by this module beyond what the listed rules explicitly cover.

