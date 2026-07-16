# Client-visible token commit protocol

Each token event includes session ID, owner epoch, token index, token ID, state commit version, and an optional migration transaction ID. The durable gateway ledger stores expected owner epoch, next accepted index, commit watermark, optional client acknowledgment watermark, terminal state, and accepted token identity.

## Acceptance rules

- Reject output from an epoch other than the coordinator/gateway owner epoch.
- Accept only the next token index; report gaps instead of filling them.
- Repeated identical `(epoch, index, token ID)` is deduplicated idempotently.
- Reuse of an index with a different token or state version is a mismatch.
- Switch owner only at the declared final source token boundary.
- Commit terminal state once; cancellation and terminal races are explicit events.

The reference runtime separates emission from acknowledgment. A consistent capture is not allowed while emitted tokens are unresolved. After cutover the destination starts at the next accepted source index, and a stale-source emission is rejected by both adapter fencing and gateway epoch checks.

## Delivery guarantees

The exercised claim is exactly-once **acceptance by the SLOForge gateway ledger**. Exactly-once delivery to a client requires an acknowledgment-capable resumable protocol whose acknowledgment watermark is durable. Without client acknowledgments, network delivery is at least once and the client must deduplicate `(owner epoch, token index)` events. Continuum does not claim stronger external delivery semantics.
