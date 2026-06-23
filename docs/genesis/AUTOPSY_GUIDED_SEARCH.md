# Autopsy-guided search

Genesis consumes the existing `sloforge.autopsy.diagnosis/v1` artifact rather than inferring a
mutation surface from log text. A diagnosis is usable only when clock alignment is sufficient and
its confidence clears an explicit threshold. The causal bottleneck maps to a typed whitelist of
genome regions and transformation families; every other region is recursively marked frozen.

`MutationGuard` rejects a proposal when any declared affected region is frozen or any
transformation family is outside the budget. This check precedes compilation and benchmarking.
The budget also records the next ranked bottleneck, evidence identities, estimated verifier cost,
and a counterfactual upside interval when Autopsy actually supplied one. It does not invent an
upside estimate when no counterfactual evidence exists.

The current attribution table covers queueing and capacity, startup, tensor/kernel, fabric,
expert-load, KV-transfer, and worker-failure diagnoses. Physical network and collective diagnoses
freeze request policy and tensor algebra while allowing state, distributed, kernel, and recovery
changes. The comparison artifact accepts observed guided and unguided run summaries and computes
candidate, invalid-candidate, hardware-experiment, time-to-improvement, objective, and diversity
deltas without manufacturing missing measurements.

This is causal search guidance, not correctness evidence. Every guided candidate still traverses
the normal static, differential, property, model-check, resource, performance, capsule, and rollout
gates. A changed bottleneck requires a new diagnosis and mutation budget.
