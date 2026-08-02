# Genesis limitations

Genesis reports scoped evidence and does not claim universal model support, universal correctness, global search optimality, or hardware performance without execution.

## Frontend and baseline runtime

- Static inspection is conservative AST recovery. It does not understand arbitrary reflection, dynamic import, opaque native mutation, arbitrary data-dependent control flow, or undeclared custom-operator semantics.
- Optional `torch.export` covers only packages with a declared fixture and an installed compatible PyTorch. Export evidence does not replace semantic, state, batching or streaming contracts.
- The generated baseline runtime uses one conservative worker and serial reference operations inside bounded microbatches. It does not expose an HTTP listener itself, preempt a hung native operator inside Python, migrate state, or implement distributed inference.
- The reference-package interface is deliberately restricted. Unsupported behavior blocks compilation instead of being guessed.

## Synthesis surfaces

- The policy DSL is scalar and loop-free. Bounded equivalence exhaustively enumerates Boolean/integer domains only; floating domains require another verifier.
- The tensor rewriter implements a small explicit rule set and bounded breadth-first exploration, not general equality saturation or arbitrary tensor-program synthesis.
- The accepted CPU runtime lowers a bounded contiguous/paged allocator with exact reservation accounting. The state compiler does not perform real DMA, offload, compression, or live state conversion.
- Distributed synthesis mutates selected fields of an already valid Fabric plan. It does not synthesize arbitrary TP/PP/DP/EP degrees or new topology from scratch in that module.
- The local end-to-end synthesis fixture joins request, serving, and state changes through batching and state-layout transformations. Tensor and Fabric mutations are evaluated separately and are not lowered or admitted into the same arbitrary whole-genome runtime.

## Verification

- Operator property testing is bounded NumPy execution, not a proof for all values, compiler modes or devices. Counterexample values are bounded diagnostic samples, not a universal minimizer.
- Quality evaluation relies on datasets supplied by the caller; dataset independence and secrecy are operational responsibilities.
- The Rust protocol checker is explicit-state and bounded. It abstracts tensor math, packet networks, memory ordering and external runtime internals. Truncated exploration is inconclusive.
- Resource verification is conservative arithmetic over declared demand, not automatic whole-program memory analysis or an allocator proof.
- Performance acceptance uses seeded bootstrap confidence intervals over supplied samples. The harness, not the statistical function, must prevent timer, synchronization, affinity, cache, precision and input-distribution errors.
- A Level 5 claim is specific to its recorded hardware, software, workload and operational scope. No lower-level evidence may be presented as hardware validation.

## Trust and operations

- SHA-256 content addressing detects alteration but does not authenticate signer identity unless a caller pins the expected digest. The implemented capsule is not a public-key signature scheme.
- The sandbox depends on host OS facilities. `sandbox-exec` is deprecated; bubblewrap/user namespaces may be unavailable; portable process and memory limits are not equivalent to cgroups. Unsupported hosts fail closed.
- The local evolution controller exercises shadow/canary gates with supplied observations and preserves stream-to-capsule leases. It is not evidence of external production traffic. External live canary/promotion requires two explicit opt-ins.
- No module presence establishes paid-cloud, GPU, multi-GPU, multi-node or upstream-runtime validation. Those statuses must cite raw artifacts from actual runs.

## Research claims

The implemented cancellation-policy fixture is structurally different from the baseline within this repository and demonstrates real counterexample-guided correction. It is not claimed to be globally novel. Similarly, independent compiler surfaces demonstrate feasibility within declared domains; they do not establish that Genesis will improve every model, workload or machine.
