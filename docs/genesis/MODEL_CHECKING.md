# Bounded distributed model checking

`sloforge-genesis-modelcheck` is a self-contained Rust explicit-state checker for serving and rollout protocols. It accepts a strict `genesis.modelcheck.request.v1` JSON document on standard input and emits a strict `genesis.modelcheck.result.v1` document. Input is capped at 1 MiB.

## Modeled behavior

The state includes a bounded request set, queue, workers and failures, committed/pending/emitted tokens, state owners and transfers, champion/previous generations, rollout stage, active/orphaned requests, and controller crash/recovery state. Typed transitions cover admission, start, token commit/emission/drop/duplication, completion, cancellation/release, worker failure/retry/abort, state transfer and partial read, shadow/canary/promotion/rollback, controller crash/restart/recovery, and logical time.

The checker reports these properties independently:

1. no duplicate committed token;
2. no committed token disappears;
3. no unsafe retry after visible output;
4. cancellation eventually releases resources;
5. queues remain bounded;
6. state ownership is unambiguous unless replication is declared;
7. partially transferred state is not read;
8. incompatible state is not migrated;
9. promotion preserves active requests;
10. rollback restores a valid champion;
11. recovery terminates;
12. no bounded deadlock;
13. live requests progress or time out under the declared fairness model.

## Exploration and evidence

Exploration is deterministic breadth first. The seed breaks successor-order ties but does not relax completeness. Bounds cover requests, queue capacity, tokens/request, workers, worker failures, depth, states, and fairness window. Results record model version, seed, bounds, state and transition counts, assumptions, action coverage, each invariant result, and truncation reasons.

If depth or state capacity truncates unexplored successors, unaffected properties are `inconclusive`; the result is not silently passed. `universal_proof` is always false. A failure contains a shortest trace under breadth-first exploration, per-step typed actions, state summaries and fingerprints. The independent replay function rejects altered fingerprints or actions.

## Exercised and unexercised scope

Fixtures exercise a safe protocol plus duplicate/drop delivery, cancellation leak, queue overflow, ambiguous ownership, partial reads, incompatible migration, orphaning promotion, invalid rollback, stalled recovery and trace tampering. JSON Schema generation is available for request and result documents.

The checker proves no property beyond its finite transition model, bounds and assumptions. It does not model packet-level networks, actual thread memory ordering, CUDA/NCCL internals, Byzantine failures, unbounded liveness, arbitrary generated source, or numerical tensor behavior. Protocol changes must be encoded as a `ProtocolSpec`/transition-model change before this checker can cover them.

