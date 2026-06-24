# Genesis architecture

SLOForge Genesis is implemented as a proof-carrying extension of the existing SLOForge compiler, Fabric planner, Autopsy diagnosis pipeline, and deployment controller. It does not replace those systems. The implemented CPU vertical slice turns a typed reference package into an `InferenceGenome`, generates a conservative runtime, synthesizes request/serving policy candidates, rejects an unsafe candidate with a minimized counterexample, and retains scoped evidence for later capsule and rollout gates.

## Data and control flow

```text
reference package + contracts + explicit seed
                  |
        static zero-day inspection
                  |
      canonical InferenceGenome v1
                  |
       generated baseline runtime
                  |
    typed proposals and transformations
                  |
 candidate lifecycle + bounded budgets
                  |
 independent semantic / quality / resource /
 model-check / performance evidence
                  |
       hash-addressed GenesisCapsule
                  |
 shadow -> canary -> promotion or rollback
```

Python owns reference-package inspection, baseline genome compilation, generated runtime orchestration, policy/tensor/state mutation surfaces, search, statistical verification, Autopsy guidance, and evolution control. Rust owns the second canonical implementation of the Genesis IR and the deterministic bounded explicit-state protocol checker. The boundary remains versioned JSON over files or subprocess standard input/output. The generated baseline runtime has a separate bounded JSON-lines process protocol; it is not a new general RPC system.

## Major components

| Component | Implemented responsibility | Trust status |
| --- | --- | --- |
| `genesis.frontend` | Manifest validation, package hashing, non-importing AST recovery, optional explicit `torch.export` evidence | Validated input; recovered semantics remain obligations |
| `genesis.compiler` | Conservative inspection-to-genome lowering and initialized run creation | Compiler output requires independent verification |
| `genesis.ir` and `sloforge-genesis-ir` | Strict wire types, schemas, migration, canonical JSON and SHA-256 | Trusted boundary |
| `genesis.runtime` | Model-specific bounded reference runtime generation and lifecycle | Generated runtime remains untrusted |
| `genesis.policy_dsl` | Restricted parser, checker, bytecode compiler, interpreter, graph, simplifier, mutation and bounded equivalence | Restricted compiler; accepted policies still need evidence |
| `genesis.tensor_rewrites` | Bounded typed rewrite exploration with numerical preconditions | Rules and candidates require independent evidence |
| `genesis.state_transforms` | State-layout/ownership compilation and trace verification | Restricted compiler and checker |
| `genesis.distributed_synthesis` | Mutations over an already valid Fabric physical plan | Reuses Fabric validation; performance evidence invalidated |
| `genesis.search` | Deterministic proposal portfolio, lifecycle, nine-dimensional budget and Pareto archive | Proposal scores are not acceptance evidence |
| `genesis.verification` | Operator, quality, resource and statistical performance evidence | Independent acceptance inputs |
| `sloforge-genesis-modelcheck` | Bounded explicit-state protocol exploration and replayable counterexamples | Trusted bounded checker |
| `genesis.capsule` and `genesis.sandbox` | Integrity, compatibility, evidence and hostile-code execution boundary | Small trusted computing base |
| `genesis.autopsy_guidance` | Diagnosis-to-mutable-region budget and frozen-region enforcement | Search guidance only |
| `genesis.evolution` | Persisted champion/challenger shadow, canary, promotion and rollback | Promotion authority after capsule validation |

## Candidate authority

Synthesis never authorizes deployment. A proposal can advance only through contiguous lifecycle stages, and failure states are terminal. Static checks, compilation, differential tests, property checks, model checking, simulation, hardware evidence, shadowing, canarying, capsule acceptance, and promotion are distinct stages. Synthetic evidence cannot be relabeled as hardware evidence.

Generated source, policies, kernels, and transformations are outside the trusted computing base. Generated execution uses the fail-closed sandbox described in `TRUST_MODEL.md`. Capsule admission binds artifacts, evidence, contracts, hardware, dependencies, and provenance to a canonical digest. The evolution controller revalidates the capsule immediately before promotion.

## Implemented integration status

The exercised local vertical slice connects static inspection, genome compilation, runtime generation, policy DSL compilation, cancellation CEGIS, sandboxed differential replay, and typed candidate persistence. Tensor rewrites, state transformations, Fabric mutations, statistical verification, the model checker, Autopsy guidance, and evolution are implemented and exercised by focused tests.

The local synthesis fixture does not yet compose every independent mutation surface into one arbitrary whole-genome search run. GPU kernel execution, GPU performance claims, real multi-node Fabric mutation, external live traffic, and paid synthesis are opt-in paths and are not established by the CPU component tests. Their status must be taken from generated evidence and the final report, never inferred from module presence.
