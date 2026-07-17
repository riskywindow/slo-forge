# Helix final adversarial review

Review date: 2026-08-03

The final distributed-systems, state-semantics, numerical, resource-scheduling, security,
agent-environment, causal-credit, benchmark, statistical, and hiring reviews classify Helix as a
transactional learning substrate rather than a generic RL orchestration wrapper. The distinguishing
boundary is evidence-carrying coordinated model/environment/effect capture, explicit state-reuse
dispositions, policy/log-probability/reward/credit lineage, bounded staleness, and proof-gated
promotion with compatibility-aware session routing and rollback. The trainer remains deliberately
small and pluggable.

## Findings closed

- Capture execution ownership is fenced; recovery is explicit; retries cannot duplicate live
  callbacks or resume a session they did not pause.
- Environment, Continuum state, rewards, batches, registries, sessions, and promotion capsules are
  tenant-bound. New-session routing and champion updates are transactional.
- State-reuse reports partition every declared source component and cannot omit or overlap reuse,
  recomputation, conversion, replacement, or rejection dispositions.
- Replay tolerances reject non-finite values; causal replay compares declared semantic digests and
  parent topology; evidence validators recompute match and scope invariants.
- Branch minimization validates witness topology and digests. Observational branches receive zero
  optimization credit and cannot enter controlled sibling preferences.
- PPO clipping gradients, temperature gradients, KL scaling, masks, staleness exclusion, behavior
  probabilities, checkpoint identity, and PEFT label masking were corrected and adversarially
  tested.
- Serving reservation is a hard scheduler constraint. Every scheduler policy is evaluated for
  every seed, raw learning-value observations are seeded and sealed, and small-sample intervals use
  paired Student-t sensitivity summaries.
- Promotion rehashes eight local artifacts, validates Continuum recomputation requirements and
  tenant scope, persists shadow/canary evidence, and leaves incompatible sessions champion-pinned.

No unresolved high-severity or reasonable medium-severity finding remains in the exercised local
reference boundary.

## Claim boundary and hiring verdict

Helix does not establish causal identification, autonomous self-improvement, generalization to an
external benchmark, independent evaluator governance, distributed-consensus correctness, or GPU/
production learning gains. Replay comparison does not itself prove that an external backend
restored bytes; the flagship separately exercises the local Continuum and environment restoration
paths. The coding task, candidate actions, failure-seed search, threshold, and training schedule are
pre-authored and disclosed.

The hiring review returned **Strong Hire** for Staff ML-systems/research infrastructure and
**Hire/Lean Hire** for senior applied RL. It did not find sufficient empirical or identification
evidence for a Principal causal-inference or empirical-RL research claim.
