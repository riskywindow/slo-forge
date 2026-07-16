# State transport

`StateTransport` moves chunk references between authorized source and destination stores under a deadline. It reports checksums, attempts, bytes on wire, elapsed time, faults, and acknowledgments. It does not evaluate model or state compatibility.

## Implementations

- `InProcessTransport`: exact local copy with integrity checks.
- `LocalFileTransport`: atomic bounded spool through the filesystem.
- `DeterministicSimulatedTransport`: seeded bandwidth/latency, retransmission, loss, corruption, duplication, and deadline behavior for CPU protocol tests.
- `TCPStateTransport`: framed portable TCP transfer with bounded frames, socket deadlines, checksum verification, acknowledgments, retries, cancellation, replay journal, and deterministic fault profiles. An explicitly configured 256-bit key enables AES-GCM payload confidentiality with transfer/chunk/attempt-bound nonces and authenticated metadata; a mismatched key fails closed. Plaintext and encrypted modes never silently fall back to one another.

Capabilities state memory type, streaming, retransmission, checksum, compression/encryption support, and zero-copy/GPUDirect availability. Fabric measurements drive selection only after the compatibility contract is known.

NIXL, BIFROST, UCX, or RDMA can implement this interface without owning semantic decisions. They are optional and are not normal-CI dependencies. The current environment has no exercised RDMA, GPUDirect, or BIFROST path.
