# Execution state capsule

An `ExecutionStateCapsule` is the content-addressed, proof-carrying publication unit for active execution state. It contains strict metadata and references exact segment chunks in a tenant-scoped content store; large state need not be embedded in the JSON document.

## Contents

- Identity: schema/API version, capsule/session/model/tokenizer/adapter hashes, source runtime and physical plan, owner epoch, capture time, Git commit, and Continuum version.
- Logical state: the complete `LogicalStateSchema`, dependency graph, output watermarks, exactness requirement, and quality contract.
- Physical state: `PhysicalStateLayout`, segments, page table, chunk references, versions, hashes, compression, and encryption metadata.
- Compatibility constraints: source fingerprint, required destination capabilities, prohibited conversions, recomputation permission, quality budget, and architecture restrictions.
- Transaction binding: transaction ID, lease ID, fencing token, source/proposed epochs, commit and rollback watermarks, dirty-log state, and journal hash.
- Evidence: capture consistency, segment integrity, conversion/continuation verification references, model-check scope, benchmark provenance, and limitations.

## Publication and validation

`publish_capture` writes exact captured segment bytes to a `ContentStore`, publishes the manifest transactionally, builds the capsule, and seals its Merkle-style integrity root. `validate_capsule` recomputes canonical hashes and checks internal cross-references. Loading is strict and bounded.

Validation detects changed manifests or evidence, altered or missing segment references, stale page versions, inconsistent logical/physical IDs, mismatched ownership epochs, and transaction-journal tampering. Chunk reads independently verify plaintext content hashes; a valid capsule is therefore insufficient if a referenced chunk is unavailable or corrupt.

## Capsule families

Complete checkpoints are root capsules. Incremental checkpoints carry ancestry and only changed segments. Fork capsules share immutable chunk references while mutable descendants use copy-on-write. Clone capsules publish independent references. Rollback and migration capsules preserve the transaction and watermark boundary. Recomputation-assisted capsules record which components are regenerated and the dependency evidence authorizing that action.

Capsules are valid only inside their declared compatibility and authorization domain. Possession of a JSON capsule alone does not authorize read, fork, resume, or owner-epoch acquisition.
