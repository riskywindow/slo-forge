# Coordinated capture

`CoordinatedCaptureCoordinator` is a durable SQLite-backed barrier across action/model,
environment, and effect sources. Its phases move from proposal through quiescence and independent
component capture to validation and publication. Journal entries permit retry after process failure.

The request fixes source watermarks, policy epoch, seed, time and software provenance, and bounded
quiescence polling. Sources must agree with the requested boundary. Timeout, mismatch, identity
conflict, or hook failure enters an aborted or failed terminal state; a `CoordinatedBranchPoint` is
visible only after `COMPLETED`.

SQLite provides local durability and transactionality, not distributed consensus. A production
deployment would require a linearizable coordinator and adapter-specific quiescence proofs. See ADR
0037 and [branch point](BRANCH_POINT.md).
