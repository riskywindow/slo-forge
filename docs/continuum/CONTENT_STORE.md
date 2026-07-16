# Content-addressed state store

Continuum supplies in-memory and filesystem/SQLite content stores. A chunk reference binds tenant, plaintext SHA-256, plaintext/stored lengths, and compression. A published manifest is immutable and includes ordered references, owner identity, expiry, and ancestry.

## Properties

- Tenant-scoped deduplication; cross-tenant deduplication is disabled by construction/default.
- Configurable bounded chunks, partial reads, concurrent readers, integrity verification, optional zlib compression, reference counting, copy-on-write forks, TTL, bounded garbage collection, and startup orphan cleanup.
- Filesystem publication writes and fsyncs chunk data before committing SQLite metadata. Garbage collection commits metadata removal before unlinking orphaned files so a crash does not leave a live reference to deleted bytes.
- Missing/corrupt chunks fail explicitly. Raw state never appears in structured logs.

`AuthenticatedEncryptedStore` wraps a store with AES-GCM when the optional cryptography dependency is available. Associated data binds tenant, chunk digest, key ID, and format metadata; authorization and replay checks remain separate from encryption. Key material is resolved by a provider and is not placed in capsule plaintext or passed to generated conversion code.

Deletion is best effort: removing references and files cannot guarantee physical erasure from snapshots, SSD remapping, backups, or lower storage layers. Deployment retention and key-destruction policy must account for that limit.
