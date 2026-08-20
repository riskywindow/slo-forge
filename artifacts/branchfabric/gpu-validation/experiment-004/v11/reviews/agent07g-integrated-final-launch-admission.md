# Agent 07g integrated final launch admission

Status: **PASS for exactly one atomic attempt-B reservation and one Modal FunctionCall.**

The production evidence verifier passes against raw config SHA-256 `3e7a45d51e9d3a83ab6a8b6bda7bdd180c52907be975d181123e5ed1f197801f` and canonical config SHA-256 `a4dd7976681fddaada89af0e6c302754669a7b1093479f8648bf6b309cb961c1`. The fresh full `make check`, post-micro manifest, cross-runtime review, packaging review, integrity review, replacement-budget review, and provider preflight are content-addressed and PASS. The provider preflight reports zero active apps, tasks, containers, endpoints, reservations, owned child processes, and profilers; only the two authorized persistent volumes exist.

The authoritative ledger is unchanged at SHA-256 `84175458f5e0b2f54fdc7c1e2c37c7d9b855f4b3771127299bcbeb4ad827f603`, contains zero reservations, has consumed 14,937.513626085978 GPU-seconds, and has 6,662.486373914022 GPU-seconds remaining. The exact 2 × 588-second reservation is 1,176 GPU-seconds. Including the 15% planning margin gives 1,352.4 GPU-seconds and leaves 5,310.086373914022 GPU-seconds under the 21,600 GPU-second authorization. Exact launch cost is $0.816144; the margin projection is $0.9385656, both within the explicit $40 authorization.

The designated root GPU coordinator may atomically create the sole reservation `exp004-v11-integrated-s41-b-reservation`, binding exact attempt B, the canonical config hash, two A100-80GB devices, and 588 seconds. It must then issue exactly one FunctionCall with retries disabled, using `SLOFORGE_GPU_BUDGET_USD=40.0`, the exact reservation ID, and a fresh one-time preflight token.

Any sealed-input change, ledger-prehash change, extra reservation, dirty provider state, or environment mismatch revokes this admission. This artifact authorizes no attempt C, no second attempt-B FunctionCall, and no retry after a FunctionCall is created. Post-return settlement and complete provider cleanup remain mandatory.

This final admission is acyclic: it is not part of the post-micro manifest and does not modify any evidence that the config binds. The reviewer invoked no Modal, GPU, or network resource.
