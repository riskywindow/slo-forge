# ServingSynthBench task grammar

The typed grammar is defined in `sloforge.synthbench.models`; generation is implemented by `sloforge.synthbench.grammar`. Every value needed for generation is derived from the declared seed with stable SHA-256 sub-seeds.

## Architecture vocabulary

The grammar composes dense attention, sliding-window attention, grouped-query attention, gated MLP, sparse MoE, state-space, recurrent-state, convolutional-state, custom normalization, quantized-state transformation, residual branches, speculative heads, custom samplers, and optional cross-attention. Block parameters are typed and bounded, including window, expert, group, convolution, quantization, and state dimensions.

Each task includes:

- a pure-Python reference model with explicit loader, allocator, prefill, decode, and sampler entry points;
- a Genesis-compatible reference manifest;
- semantic, quality, state, numerical, batching, streaming, and supported-domain contracts;
- an explicit workload with committed expected outputs;
- a deterministic sample generator;
- a typed architecture descriptor and hashes;
- evaluator-only randomized cases and their public commitment.

The generator-side interpreter is independent of the emitted reference source. Tests compare the emitted server behavior with expected outputs derived by that interpreter.

## Generalization controls

Generation is byte-deterministic for the same configuration. Across `len(BlockKind)` tasks, the ordinal construction guarantees coverage of every declared block kind while the rest of each architecture remains seeded. Hidden cases are generated after the public workload and are not placed in the reference package.

The static audit parses submitted Python ASTs and rejects embedded task IDs, generation seeds, hidden commitments, and branches that enumerate multiple hidden prompt lengths. This is a targeted anti-special-casing signal, not a general proof that an implementation contains no benchmark-specific behavior.

