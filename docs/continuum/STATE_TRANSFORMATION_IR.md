# State Transformation IR

`StateTransformationIR` is a strict, versioned transformation DAG. Each `TransformationOperation` names typed source and destination tensor contracts, preconditions, postconditions, exactness, shape and dtype changes, ownership behavior, target device, estimated cost, temporary memory, streamability, verification obligation, and fallback.

The operation vocabulary covers slice, concatenate, split, reshape, permute, transpose, pad/unpad, interleave/deinterleave, pack/unpack, shard/reshard, replicate/gather/scatter, page remap/coalesce/split, dtype conversion, quantize/dequantize, compression, checksum, encryption, copy/zero-fill, metadata reconstruction, recomputation, send/receive, destination write, and validate.

## Invariants

- Operation IDs and outputs are unique.
- Dependencies reference known operations and the graph is acyclic.
- Shape/dtype contracts are explicit; conversions cannot be implicit.
- A chunk schedule covers valid byte or token ranges without exceeding its bounded buffer budget.
- Ownership-changing operations are separated from byte transformations.
- Lossy operations carry a quality obligation; generated/device code carries an independent verification obligation.

Topological order is deterministic. The compiler can pipeline chunks and layers, fuse compatible operations, overlap transfer, retry idempotent chunks, and track partial completion without materializing an entire canonical KV array.
