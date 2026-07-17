# SLOForge Helix: H1 and H2

## H1: partial

State identity and replay contracts, not downstream trajectory fidelity.

- Observation: Joint restoration passed exact identity in 3/3 runs.
- Observation: Non-joint modes passed exact identity in 0 cases.
- Observation: Transcript equality explicitly did not establish hidden model/environment equivalence.
- Limitation: Readiness, prefill work, and divergence under a nondeterministic real model were not measured.
- Limitation: Reference policy and repository are synthetic local CPU fixtures.
- Limitation: The result does not establish behavior for arbitrary models, tools, or production traffic.
- Artifact: `raw/flagship/seed-41/summary.json`
- Artifact: `raw/flagship/seed-73/summary.json`
- Artifact: `raw/flagship/seed-113/summary.json`

## H2: partial

CAS-backed logical copy-on-write isolation and cleanup.

- Observation: All branch workspaces were isolated and cleaned after each run.
- Limitation: No reflink/page-fault or GPU-memory amplification benchmark was run on this host.
- Limitation: Reference policy and repository are synthetic local CPU fixtures.
- Limitation: The result does not establish behavior for arbitrary models, tools, or production traffic.
- Artifact: `raw/flagship/seed-41/summary.json`
- Artifact: `raw/flagship/seed-73/summary.json`
- Artifact: `raw/flagship/seed-113/summary.json`
