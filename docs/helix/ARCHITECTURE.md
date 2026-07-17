# SLOForge Helix architecture

Status: implemented local reference architecture, 2026-08-03.

Helix is SLOForge's evidence-carrying learning loop. It turns a policy decision into coordinated
model, environment, action, and effect state; creates isolated counterfactual branches; records
policy-versioned trajectories and deterministic reward evidence; constructs provenance-complete
training inputs; evaluates a candidate; and promotes or rejects it without weakening serving SLOs.

```text
PolicyEpoch + coordinated model/environment/effect capture
                          |
                    BranchPoint
                          |
      Continuum state forks + environment COW branches
                          |
           strict or segmented TrajectoryCapsules
                          |
       reward evidence -> branch-relative credit
                          |
   staleness gate -> TrainingBatch -> trainer adapter
                          |
       candidate epoch -> separately configured evaluation gates
                          |
       promotion capsule -> shadow -> canary -> CAS
```

Python owns orchestration, validation, local environment state, rollout/reward/training reference
workers, evaluation, scheduling, security controls, promotion, and reports. Rust owns shared Helix IR
wire types and the deterministic data-plane/simulation work integrated elsewhere in SLOForge. The
primary boundary remains versioned JSON over bounded subprocess stdin/stdout; HTTP/SSE is reserved
for a running data plane.

All externally loaded records are bounded and strict. Hashes bind content and lineage, but are not
signatures. Seeds are explicit. Side effects are typed and fail closed. Serving feasibility is a hard
constraint, not a term that learning value may outweigh. See the [dependency graph](DEPENDENCY_GRAPH.md),
[trust model](TRUST_MODEL.md), and [limitations](LIMITATIONS.md).

## Implemented packages

- `ir`: canonical learning artifacts, schema generation, bounded document loading.
- `environments`, `effects`, `capture`, `branching`, `replay`: coordinated state and counterfactuals.
- `rollouts`, `rewards`, `credit`, `datasets`, `staleness`, `trainers`: evidence-to-update path.
- `experience`, `scheduler`: selective evidence curation and serving-hard resource compilation.
- `promotion`, `transactions`, `security`: durable admission, routing, rollback, and controls.
- `evaluation`, `faults`, `demo`: local CPU evaluation, fault matrix, and flagship integration.

The checked implementation is a deterministic local CPU reference. It does not establish production
sample efficiency, GPU efficiency, arbitrary-environment replay, or safe autonomous deployment.
It is a transactional learning substrate—a set of state, evidence, admission, and rollback
boundaries—not a generic RL library or proof that the bundled optimizer improves arbitrary policies.
