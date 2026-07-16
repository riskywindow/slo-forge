# Semantic compatibility

Compatibility asks whether captured logical state can continue under a destination model/runtime/layout contract. Tensor shape equality is evidence about representation only; it never establishes semantic reuse by itself.

## Classes

1. `EXACT_BITWISE`: the declared environment and representation yield identical bits.
2. `EXACT_SEMANTIC`: logical state is exact while physical layout or implementation differs.
3. `NUMERICALLY_EQUIVALENT`: a declared operator tolerance permits bounded numeric differences.
4. `QUALITY_BOUNDED`: measured loss stays inside an explicit quality budget.
5. `RECOMPUTATION_ASSISTED`: token history/checkpoints regenerate incompatible components.
6. `INCOMPATIBLE`: no known safe plan satisfies the contract.

The typed decision includes supporting reasons, rejected stronger classes, required transformations, recomputation requirements, unsupported state, quality implications, verification obligations, and migration restrictions. `to_canonical_report` converts the internal reasoning result to the shared ABI report.

## Fail-closed rules

- Exact model weights plus a layout-only page/TP/order change can be exact semantic.
- Weight-derived state is invalid when state-producing weights change. Matching shapes do not relax this rule.
- A proven output-head-only change can preserve upstream KV/recurrent state when the dependency graph shows the head does not produce stored state.
- Tokenizer or special-token mapping changes require a verified exact mapping; otherwise reject.
- Positional encoding, RoPE, context scaling, mask/window semantics, attention geometry, recurrent update equations, or affected LoRA/adapter computations require explicit proof, recomputation, or rejection.
- Quantization is not bitwise exact. Quality-bounded classification requires measured evidence and a declared budget.
- Runtime changes are exact only when both adapters identify the same logical semantics and support the required state kinds.

## Demonstrated negative case

The CPU flagship constructs a same-shape model revision with a changed state-producing dependency. Direct reuse is classified `INCOMPATIBLE` with `STATE_PRODUCING_DEPENDENCY_CHANGED`. When complete token history and a valid dependency path are supplied, a distinct `RECOMPUTATION_ASSISTED` plan is emitted. This is a planning and deterministic reference-runtime exercise, not a claim of universal cross-model cache reuse.
