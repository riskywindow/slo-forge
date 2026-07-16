# Checkpoint, fork, and clone

`pause_and_checkpoint` quiesces a reference session at a legal boundary, captures exact state, publishes a capsule, and preserves the delivery cursor. Checkpoints may be full roots or incremental descendants; ancestry digests bind parent capsule and manifest IDs and changed segment IDs.

## Fork

A fork creates distinct descendant session IDs and owner epochs while sharing authorized immutable chunk references. Mutable sampler, workflow, recurrent, delivery, and newly appended KV state diverge through copy-on-write manifests. Each branch has independent fencing and cannot emit under the parent's lease.

The CPU flagship forks the migrated checkpoint into two branches, resumes them under different reference layouts and TP degrees, records shared and newly unique bytes, and preserves ancestry. Those byte counts come from the actual content-store manifests.

## Clone

A clone creates an independent descendant publication and may restore through another compatible runtime/layout. It reports copied bytes separately from logical shared-state size.

Branches can be compared, selected, or discarded. Continuum intentionally provides no generic branch merge: merging generated tokens, workflow side effects, RNG state, or recurrent state requires a model/workflow-specific contract.
