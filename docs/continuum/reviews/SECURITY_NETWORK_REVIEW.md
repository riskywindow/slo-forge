# Continuum security, privacy, and networking review

Review date: 2026-08-02

Scope: `python/sloforge/continuum/storage`,
`python/sloforge/continuum/transport`, the Continuum security and threat-model
documents, and their focused tests. This is a source and deterministic CPU review;
it is not a penetration test, cryptographic certification, or review of an
unexercised mTLS deployment.

## Result

No open high-severity or reasonable medium-severity findings remain in the
reviewed scope. The focused acceptance suite passed 23 tests. Ruff and strict mypy
passed on all reviewed Python modules and the adversarial review tests.

## Findings fixed

### SEC-NET-001: TCP data frames were not authenticated (high)

Before this review, the setup frame had an HMAC but chunk headers, acknowledgments,
rejections, and completion frames did not. An on-path attacker could replace both
a chunk digest and its payload consistently, defeating the unauthenticated SHA-256
check. Captured frames were also not bound to a transfer ID.

The protocol now authenticates every metadata frame with HMAC-SHA-256, separates
client and server authentication domains, and binds chunk/acknowledgment frames to
the transfer ID. Payload bytes remain intentionally unencrypted; the capability
matrix continues to declare `supports_encryption=false`. An adversarial raw-socket
test alters digest, size, and payload while retaining a captured tag and verifies
that the listener fails closed without storing the substitute or logging secrets.

### SEC-STO-002: File-store metadata could select a non-canonical path (high)

The local store previously joined the SQLite `path` field to its root without
checking that the row matched the content-derived tenant path. Local metadata
tampering could therefore make an authorized read target a different file if its
content identity was also known.

Reads, garbage collection, and the corruption test hook now require the exact
canonical `(tenant, digest)` path and reject resolved paths outside the chunk root.
Reads use a bounded regular-file descriptor and `O_NOFOLLOW` where the platform
provides it. A database path-escape test verifies fail-closed behavior.

### SEC-STO-003: Compressed input could expand without a hard bound (high)

`zlib.decompress` previously allowed corrupted local bytes to allocate output
beyond the 64 MiB logical chunk limit before the digest/size check. Decoding now
caps expansion, rejects incomplete and trailing streams, bounds stored size, and
checks the file size before allocation. The review test uses a reduced deterministic
bound to exercise the expansion rejection without consuming excessive memory.

### SEC-STO-004: Local permissions and abandoned transfer retention (medium)

The filesystem backend relied on process umask for directory and database
permissions, and partial TCP transfers created unpublished chunks without an
expiration. Store roots and chunk directories are now owner-only, the SQLite
database is mode `0600`, and TCP receiver staging writes receive a configurable
bounded TTL (one second through seven days, default one hour). Duplicate puts take
the shortest requested expiration, while reference counts still protect published
capsules from collection.

### SEC-ENC-005: AES-GCM boundary advertised an impossible maximum (medium)

The encrypted wrapper accepted a 64 MiB plaintext even though the 16-byte GCM tag
made its ciphertext exceed the underlying store's 64 MiB chunk bound. The wrapper
now reserves the tag bytes in its validated maximum and continues to fail closed;
no plaintext fallback was introduced.

## Controls verified

- Content keys and manifests are tenant-scoped; equal plaintext in two tenants
  does not share a storage authorization key or manifest reference.
- AES-256-GCM associated data binds tenant, capsule, state version, key identity,
  plaintext digest, plaintext size, and schema. Keys are resolved separately and
  never serialized in references or error messages. The coordinator-provided
  minimum state-version policy rejects stale encrypted references.
- TCP setup enforces a fixed tenant, a bounded HMAC key, transfer size/count,
  per-frame lengths, I/O and whole-transfer deadlines, retry bounds, a bounded
  diagnostic buffer, and a bounded durable completed-transfer replay journal.
- Destination acknowledgments are authenticated and their strict `ChunkRef`
  tenant/digest identities are checked by the sender.
- Content and transport errors contain static classes/reasons, not state bytes,
  keys, tags, or peer-supplied identifiers.
- Cross-tenant deduplication remains disabled by construction. Best-effort deletion
  claims remain scoped: unlink, TTL, and key revocation are not physical secure
  erasure guarantees.

## Residual limitations

- Bare `tcp_v1` is not confidential and does not provide certificate-backed peer
  identity. It must stay on localhost or inside an independently authenticated,
  encrypted channel such as mTLS; this repository does not claim an exercised mTLS
  endpoint.
- A tenant-authorized transport client can stage arbitrary tenant bytes. Semantic
  activation still depends on capsule integrity, compatibility verification,
  ownership CAS, and destination validation outside the byte-transport layer.
- Replay history is bounded and node-local. A distributed deployment must bind the
  replay window to its durable transaction coordinator; the local journal is not
  consensus.
- Python key bytes and plaintext cannot be reliably zeroized. Host compromise,
  swap/crash-dump policy, KMS operation, backup lifecycle, GPU side channels, and
  physical media sanitization remain platform responsibilities.
- Filesystem mode checks reduce accidental exposure but do not defend against a
  process with the same OS identity or a compromised kernel. Concurrent writers
  should use one store service per root; the in-process API serializes threads but
  is not a distributed filesystem transaction protocol.
- Plaintext digests and sizes are sensitive equality metadata. Capsule and report
  authorization/redaction remain required even though storage dedup is tenant
  scoped.

## Acceptance commands

```text
.venv/bin/ruff check python/sloforge/continuum/storage python/sloforge/continuum/transport tests/python/test_continuum_security_review.py
.venv/bin/mypy --strict python/sloforge/continuum/storage python/sloforge/continuum/transport tests/python/test_continuum_security_review.py
PYTHONPATH=python .venv/bin/pytest -q tests/python/test_continuum_storage.py tests/python/test_continuum_security.py tests/python/test_continuum_tcp_transport.py tests/python/test_continuum_transport.py tests/python/test_continuum_security_review.py
```

Observed: Ruff passed, strict mypy passed, and pytest reported `23 passed`.
