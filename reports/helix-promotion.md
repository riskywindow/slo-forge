# SLOForge Helix: H4 and H7

## H4: not_exercised

Fail-closed policy provenance and bounded staleness semantics.

- Observation: The reference evaluation did not execute a multi-policy asynchronous staleness campaign.
- Limitation: Source-level tests are not counted as executed evaluation evidence.
- Limitation: Training stability under a large asynchronous optimizer was not measured.

## H7: supported_within_scope

Local atomic promotion state machine and evidence preservation.

- Observation: Each local synthetic run rejected an executed below-threshold candidate, promoted a passing candidate after shadow/canary, pinned an incompatible session, and restored the champion during rollback.
- Limitation: No live distributed gateway or multi-region control plane was exercised.
- Artifact: `raw/flagship/seed-41/summary.json`
- Artifact: `raw/flagship/seed-73/summary.json`
- Artifact: `raw/flagship/seed-113/summary.json`
