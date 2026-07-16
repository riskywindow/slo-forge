# Continuum related work

Continuum is complementary to inference runtimes and byte-movement systems. It owns logical state meaning, semantic compatibility, conversion planning, proof obligations, and transactional ownership; a runtime exposes/imports state and a transport moves bytes.

## Runtime KV and disaggregated serving

vLLM KV connectors and disaggregated prefill expose runtime-specific transfer mechanisms. SGLang prefill/decode disaggregation and hierarchical cache paths similarly move/cache runtime KV representations. NVIDIA Dynamo composes serving components and transfer mechanisms. These systems are valuable adapter/transport targets, but their public byte-transfer interfaces do not by themselves define a runtime-independent contract for RNG, recurrent state, guided decoding, workflows, client watermarks, or cross-model validity.

Continuum therefore version-gates integrations and fails closed when a public API cannot export the required logical components. It does not claim official runtime support merely because package discovery or a KV connector exists.

## Movement and memory systems

NVIDIA NIXL, UCX, RDMA, GPUDirect, and a separately deployed BIFROST-like store can provide high-performance memory/chunk movement, placement, or replication. Continuum's `StateTransport` allows those implementations while retaining compatibility and cutover above them. NCCL collective capabilities inform topology-aware transformation/transfer but do not define state semantics.

## Checkpoint and tensor layout systems

PyTorch distributed checkpoint and DTensor provide useful tensor/state-dict and placement mechanisms. General checkpoint systems preserve files or training tensors but do not necessarily preserve an active streaming ownership/token contract. Continuum uses a smaller active-session ABI and explicitly excludes process ephemera.

## VM/process migration and storage snapshots

Pre-copy VM migration, process checkpoint/restore, and storage snapshots motivate dirty tracking and transactional publication. Continuum differs by reconstructing runtime ephemera, compiling semantic tensor relayout, and validating bounded continuation rather than moving an address space.

## Content addressing and transactional protocols

Merkle/content-addressed stores enable deduplication, immutable publication, incremental checkpoints, and COW forks. Continuum scopes hashes by tenant authorization to avoid cross-tenant equality leakage. Its owner-epoch/fencing design follows established lease/CAS patterns and delegates distributed linearizability to etcd or Kubernetes Lease rather than inventing consensus.

## Formal and numerical verification

Explicit-state model checking gives finite counterexamples for cutover/message/fault behavior; randomized canonical-versus-optimized conversion tests establish evidence over declared layout domains. Neither is presented as an unbounded proof. Quality-bounded conversion additionally needs empirical model/operator evidence for the exact deployment domain.

Primary API source locks and compatibility notes live under `adapters/continuum/*`; reproduce any optional integration against the pinned/version-gated source before changing its status to exercised.
