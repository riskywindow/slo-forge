# Physical state layout

`PhysicalStateLayout` describes where and how logical state is represented without exposing process-local addresses. It is separate from `LogicalStateSchema`, so changing page size, ordering, placement, or sharding does not change the semantic identity of the state.

## Records

- `RuntimeIdentity` binds runtime, runtime version, adapter version, build hash, dependency versions, and target hardware.
- `StateSegment` maps a logical component and logical byte range to a physical range, tensor shape/strides, allocation and storage offsets, page/chunk IDs, version, dirty epoch, checksum, compression, encryption, and location.
- `LayoutDescriptor` represents contiguous, paged, blocked, interleaved, transposed, tiled, or namespaced runtime layouts, including page/block sizes, alignment, padding, ordering, and K/V packing.
- `ShardDescriptor` identifies TP degree, pipeline stage, expert group, replica, rank, source and destination logical slices, ordering, and replication state.
- `PlacementDescriptor` records host/GPU placement, NUMA/NIC affinity, memory tier, rail, and fault domain as symbolic identities.
- `QuantizationDescriptor` records format, scale and zero-point granularity, metadata and accumulation semantics, exactness class, and quality contract.
- `AccessPatternDescriptor` controls migration legality: append-only versus mutable, access order, required-before-resume, streamable-before-use, and recomputability.
- `PageTableDescriptor` maps logical token ranges to pages with versions, ownership, dirty state, and copy-on-write reference counts.

Runtime-specific metadata is allowed only under a namespace-qualified, versioned extension key. Raw pointers, handles, and opaque allocator objects are prohibited.

## Reference layouts

The CPU reference runtime implements two materially different layouts:

| Property | Source A | Destination B |
|---|---|---|
| KV order | token-major | head-major |
| K/V storage | separate | packed |
| logical TP | 4 | 2 |
| page size | 3 tokens | 5 tokens |
| simulated devices | 4 | 2 |

The direct converter reads source shards and writes destination shards in bounded token chunks. The canonical verifier independently decodes the logical K/V arrays and re-encodes them for the destination. Recurrent, sampler, guided-decoding, history, and delivery state remain separately typed and are not hidden in KV bytes.

Changing only physical layout is eligible for exact semantic conversion when shard coverage is complete, page versions are current, and no dtype/quantization semantics change. See [Conversion compiler](CONVERSION_COMPILER.md).
