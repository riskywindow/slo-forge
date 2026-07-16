# Continuum threat model

## Scope and security objectives

This threat model covers local and distributed capture, storage, conversion,
transfer, import, pause/resume, fork/clone, and transactional live migration of an
active AI execution. It covers confidentiality and integrity of state, tenant and
session authorization, unique ownership, and preservation of gateway-accepted
output. It does not claim to protect a host after its kernel, hypervisor, firmware,
or physical memory is fully compromised.

Security objectives are:

1. State is disclosed only to an authorized tenant/session operation and approved
   runtime trust boundary.
2. Altered, missing, substituted, stale, or incompatible state fails closed before
   activation.
3. At most one current owner can mutate committed state or have output accepted.
4. A stale owner epoch cannot resume, fork, commit, or emit accepted output.
5. Generated conversion code cannot obtain storage keys or silently weaken the
   exactness and verification contract.
6. Storage, queues, frames, retries, journals, and diagnostic buffers stay bounded
   under malicious input.
7. Claims about delivery, encryption, erasure, and model checking remain scoped to
   what was actually implemented and exercised.

## Assets

- prompt and generated-token history;
- attention KV, recurrent/state-space, convolutional, speculative, sampler,
  guided-decoding, workflow, and tool-continuation state;
- client and gateway acknowledgment watermarks;
- tenant, session, model, tokenizer, adapter, and workflow identities;
- storage and transport encryption/authentication keys;
- ownership leases, fencing tokens, owner epochs, transaction IDs, and journals;
- model weights, adapters, conversion plans, evidence, and benchmark artifacts.

## Actors and assumptions

The tenant and authorized operator are trusted to request in-policy operations.
The durable coordinator and gateway ledger are trusted to apply compare-and-swap,
fencing, and token-commit rules. Storage and transport implementations are trusted
to enforce their declared bounds but their bytes are independently verified.

Runtime adapters, runtime processes, generated conversion kernels, capsule input,
network peers, and imported manifests are potentially faulty or malicious.
Destination readiness is not proof of compatibility. Shape equality is not proof
of semantic state compatibility. Model weights and tokenizers are identified by
verified fingerprints rather than names supplied by a peer.

## Threats and controls

| Threat | Primary controls | Residual risk |
|---|---|---|
| Cross-tenant state read or equality leakage | Tenant-scoped content keys, manifest tenant validation, operation authorization, cross-tenant dedup off by default | Exported hashes and size/timing metadata require access control and redaction |
| Plaintext storage disclosure | Optional fail-closed AES-256-GCM wrapper, separate key provider, explicit key IDs, authenticated tenant/capsule/version AAD | Static local provider is not a production KMS; underlying media and backups need policy |
| Ciphertext or metadata tampering | AEAD authentication, content hashes, capsule/Merkle integrity, strict schemas | A currently authorized old object is valid unless replay policy rejects it |
| Chunk substitution | Tenant/capsule/state-version/key/digest/size AAD, expected manifest identity, destination checksum | Compromise of both key authority and coordinator defeats this boundary |
| Capsule or transaction replay | Non-reusable transaction IDs, owner-epoch and fencing CAS, minimum state-version watermark, bounded durable transfer replay journal | Node-local replay history is not a global distributed authority |
| Unauthorized resume, fork, or clone | Explicit tenant/session access policy, new descendant identity/epochs, coordinator lease acquisition | A compromised coordinator or policy service can authorize abuse |
| Split-brain state mutation | Durable lease, monotonic epochs, fencing tokens, CAS transition journal | A runtime that ignores fencing may mutate private bytes, but gateway and commit path must reject it |
| Duplicate, missing, or stale output | Gateway expected epoch, token sequence, commit watermark, deduplication, gap rejection | Exactly-once client delivery needs acknowledgment-capable client protocol |
| Network eavesdropping | Explicit `tcp_v1_aes256gcm` payload encryption plus production mTLS/channel policy | Bare `tcp_v1` has no confidentiality; pre-shared AES keys do not provide PKI peer identity |
| Network corruption, substitution, truncation, duplication, or reordering | Direction- and transfer-scoped HMAC frames, length bounds, SHA-256 payload identity, acknowledgments, retry/deadline bounds, deterministic fault tests | Repeated disruption causes explicit migration failure, not silent fallback |
| Credential theft from subprocesses | Minimal allowlisted environment, no storage keys for adapters/generated code, no secrets in logs/reports | A fully compromised parent process may access in-memory credentials |
| Malicious adapter metadata | Version gates, strict typed schemas, bounded sizes, namespace isolation, capability checks, compatibility engine | Runtime internals change; each supported version requires review |
| Malicious generated conversion code | Sandboxed/bounded execution where available, no state keys, independent reference comparison and continuation verification | Verification scope is bounded to declared inputs and horizon |
| Resource exhaustion | Bounded chunks, decompression output, headers, queues, transfer bytes, retries, deadlines, replay entries, diagnostics, and staging TTL | Authorized workloads can still consume their configured quota |
| Logs, metrics, or crash dump disclosure | No payload/key logging, opaque identifiers, redaction, bounded static errors, crash policy | Host/platform crash collection needs separate configuration |
| Deletion failure | TTL, reference counting, logical GC, key revocation, documented best-effort semantics | SSD, snapshots, replicas, caches, and backups can retain physical copies |
| Hash oracle or dedup side channel | Cross-tenant dedup off, access-controlled manifests and hashes | Within-tenant dedup intentionally reveals equality to that tenant's authority |

## Runtime adapters and generated code

An adapter may inspect state only after authorization and must publish a
version-scoped capability matrix. It must not serialize raw pointers, file
descriptors, CUDA graph objects, communicators, allocator internals, or secret
environment values. Unsupported operations return typed errors rather than
falling back to another runtime, device, precision, or transport.

Generated conversion code receives only the source chunks and destination buffers
needed for its operation. It does not receive storage keys, coordinator
credentials, or broad filesystem/network access. Its output remains quarantined
until structural validation, integrity checks, optimized-versus-reference
comparison, and exactness-appropriate continuation checks pass. Passing bounded
tests is evidence for the tested domain, not a universal proof.

## Replay and rollback windows

Replay protection is layered:

- the encryption wrapper rejects state below the caller's minimum state version;
- the transport rejects completed transfer IDs within its bounded durable replay
  window;
- the transaction coordinator rejects reused transaction IDs, stale phases,
  fencing tokens, and owner epochs;
- the gateway rejects stale epochs, duplicate token indices, and gaps.

A pre-commit abort can retain the source as the valid owner when it has not been
fenced. After ownership commit or destination-visible progress, activating an old
source is not called rollback: it requires a new fenced recovery transaction,
state reconciliation or recomputation, and a valid token watermark.

## Availability and denial of service

All protocol lengths are checked before allocation. TCP headers are at most 64
KiB, chunks at most 64 MiB, a transfer at most 4096 chunks and 1 GiB, retries at
most 16, replay journal capacity is configured, and listener diagnostics are
bounded. Timeouts apply to connection, frame reads, and the entire transfer.
Cancellation before transfer fails explicitly. A peer cannot request an unbounded
queue or force a hidden transport fallback.

Repeated corruption, partitions, destination OOM, unavailable keys, failed
verification, and coordinator unavailability reduce availability by design: the
system fails or follows an explicitly legal recovery path instead of activating
unverified state.

## Out of scope and honest limitations

- Bare TCP does not provide confidentiality or peer identity equivalent to mTLS.
- AES-GCM protects stored chunks or TCP payloads only when the corresponding
  optional mode is explicitly selected and its separately supplied keys remain secret.
- The bundled local key provider is not a managed rotation, audit, or revocation
  service.
- Local SQLite replay and ownership records do not implement consensus; distributed
  deployment needs an existing CAS coordinator such as etcd or Kubernetes Lease.
- Logical deletion and unlink are not guaranteed physical secure erasure.
- Bounded explicit-state model checking is not a proof for unbounded executions,
  unmodeled runtime behavior, or compromised trusted components.
- Side-channel resistance for GPU kernels, host memory, execution timing, and
  hardware is delegated to the selected platform and is not claimed here.

## Required security tests

Normal CPU CI covers authenticated encryption round trips, key separation,
ciphertext/AAD tampering, chunk substitution, stale state replay, wrong tenant,
missing keys, dependency-unavailable fail-closed behavior, TCP corruption,
truncation, checksums, acknowledgments, retry bounds, durable transfer replay, and
listener shutdown. Transaction and system suites separately cover stale epochs,
unauthorized output, gateway deduplication/gaps, coordinator restart, rollback
windows, cancellation, and bounded protocol exploration.
