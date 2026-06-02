# WarmPath artifact graph

`ArtifactGraph` is schema `1.0.0` and contains a DAG of typed startup artifacts.
Kinds include container and Python layers, tokenizer/config, model and quantized
weights, adapters, compiled kernels, runtime cache, graph capture, engine and
communication metadata, process/CPU/pinned/GPU state, and readiness metadata.

Each `ArtifactNode` has a safe ID, kind, size, SHA-256, source-relative path,
dependencies, rebuildability, lazy-restore permission, security class, and
compatibility constraints. Validation rejects duplicate IDs, missing
dependencies, cycles, unsafe paths, invalid hashes, and ambiguous order. Stable
topological order is used for profiling, planning, and execution.

Compatibility may constrain architecture, operating system, Python, CUDA,
driver, GPU model/architecture, runtime, communication library, and topology.
Unknown or mismatched required values reject snapshot restore; a rebuild may
remain eligible if the artifact permits it.

Storage tiers include object/remote storage, regional/AZ/peer caches, remote
memory, local NVMe, page cache, host/pinned memory, and GPU HBM. A tier records
capacity, bandwidth, latency, cost, restore failure probability, locality,
encryption/trust policy, and optional local path. `security_allows` prevents a
restricted artifact from entering a tier that lacks the required trust and
encryption properties.

The demo graph at `artifacts/warmpath/input/artifact-graph.json` contains model
config, tokenizer, a 2 MiB synthetic weight file, and a 256 KiB synthetic runtime
cache. Their bytes are deterministic fixture content, not a real model snapshot.

