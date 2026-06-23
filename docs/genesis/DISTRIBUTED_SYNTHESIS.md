# Distributed synthesis

Genesis extends Fabric rather than replacing it. The restricted distributed mutation compiler
accepts an already validated `PhysicalExecutionPlan` and a typed mutation for collective
algorithm/transport/rank order, KV transfer, communication overlap, expert placement, or rank
placement. Rank placement moves the corresponding memory-capacity record with each device.

The compiler rejects missing targets, invalid permutations, duplicate expert ranks, and ranks
outside the plan. It then reparses the candidate through the existing strict Fabric IR validator.
The candidate carries its transformation, seed, parent plan, and an explicit marker that inherited
performance evidence is invalid. It cannot be promoted until the digital twin, bounded protocol
checker, end-to-end benchmark, and capsule gates produce new evidence. Communication protocol
mutations always create a bounded model-check obligation.

This surface does not claim that a structurally valid plan is fast, correct on unmodeled hardware,
or deployable. It deliberately preserves the existing topology fingerprint and limits placement
mutations to devices already present in the source plan. New hardware or parallelism degrees must
go through the full Fabric compiler and profiler rather than this focused mutation path.
