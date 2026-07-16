# Limitations and non-claims

Continuum's deterministic CPU path exercises a portable state ABI, direct cross-layout conversion, pre-copy deltas, transactional owner cutover, rollback, gateway token acceptance, fork/copy-on-write, and recomputation planning. The following boundaries are intentional and must remain visible:

- The reference runtime's devices and topologies are simulated. Its timing is not a GPU or production-network benchmark.
- No NVIDIA GPU, CUDA, Triton, RDMA, InfiniBand, or multi-node execution was available in the recorded local campaign. There is no generated GPU-kernel performance claim.
- PyTorch, vLLM, and SGLang bindings are version-gated and fail closed, but a complete live session migration through their current public internals was not exercised. vLLM/SGLang public KV-transfer APIs do not expose sampler, guided-decoding, client-delivery, and arbitrary recurrent state as one portable contract.
- Genesis integration loads/inspects a generated runtime and exercises a CPU smoke path; it is not a hardware live-migration result.
- The local SQLite coordinator and replay journal provide crash durability on one node. Distributed deployments require an existing linearizable CAS service such as etcd or Kubernetes Lease.
- Bare `tcp_v1` authenticates every frame and verifies bytes but does not provide confidentiality. The optional AES-256-GCM payload mode is exercised locally and prevents plaintext movement, but it is a pre-shared-key mechanism rather than PKI peer identity; production deployments should still use mTLS or another approved channel policy.
- Content-store deletion is logical/best effort and cannot guarantee physical erasure from SSDs, snapshots, replicas, or backups.
- Lazy migration is legal only for declared sliding-window, layer-sequential, cold, or recomputable segments with explicit availability/stall/failure behavior. Dense full-attention KV is not generally post-copied.
- The exercised quality-bounded backend covers CPU floating dtype narrowing and measures maximum absolute error on the exact migrated state. Quantized packing (including FP8 scales/zero points) is not implemented; FP16-to-FP8 is never called exact, and production quality claims still require model/operator-level evidence.
- Bounded continuation comparison and explicit-state exploration are scoped evidence, not universal semantic/formal proof.
- The stop-and-copy/pre-copy pause comparison is an observed CPU, in-memory-store result on one host; it is not a physical-network or GPU interruption claim. Planner oracle regret is scoped to exhaustive enumeration of the implemented finite candidate set.
- The client-visible exercised guarantee is exactly-once gateway acceptance. End-to-end exactly-once delivery requires durable client acknowledgments.
- Cross-model cache reuse is rejected unless dependency evidence proves unaffected state; arbitrary branch merging is not implemented.

The generated evaluation report is the source of truth for exercised paths, negative results, and metric provenance.
