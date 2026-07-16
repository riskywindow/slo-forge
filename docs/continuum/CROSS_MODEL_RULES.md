# Cross-model and cross-version rules

Continuum chooses the smallest safe action: direct reuse, exact conversion, quality-bounded conversion, partial/full recomputation, or rejection.

| Change | Default action | Exception required |
|---|---|---|
| exact weights, different runtime | exact semantic conversion possible | both adapters agree on logical semantics |
| exact weights, page/TP/order change | exact relayout possible | complete shard/page coverage |
| state quantization | numeric/quality class | measured tolerance or quality evidence |
| output-head-only weights | upstream state may be reusable | dependency graph proves head does not produce stored state |
| attention/MLP/norm/state-producing weights | recompute or reject | explicit state adapter proof |
| LoRA/adapter change | recompute affected dependencies | proof stored state is unaffected |
| tokenizer/special tokens | reject | verified exact history mapping |
| RoPE/positional/context scaling | reject direct reuse | verified transformation |
| recurrent/state-space update | reject direct reuse | model-specific conversion and continuation evidence |
| architecture change | recompute or reject | explicit state adapter |

Availability of token history is necessary but not sufficient for recomputation. The dependency graph must establish the required upstream inputs, side-effect policy must allow replay, and bounded continuation verification must pass. Workflow tool calls with non-repeatable effects are never silently replayed.

The flagship's changed state-producer case is deliberately same-shaped. It is rejected for semantic dependency change, demonstrating that Continuum is not a tensor-copy compatibility heuristic.
