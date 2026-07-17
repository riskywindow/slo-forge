# Reward evidence

`DeterministicRewardWorker` executes bounded deterministic commands and verifier-only hidden cases in
the existing sandbox. A `RewardRun` binds trajectory and policy epoch, verifier specification,
environment before/after hashes, component input/output hashes, scores, pass flags, termination,
return code, bounded excerpts, and total score.

Hidden expected values are hashed into verifier specification evidence and are not placed in the
policy-visible sandbox. Component scores retain their source version and sum exactly to the total.
The source tree is hashed before and after execution, and duplicate reward submissions are rejected.

Command determinism is scoped to the sandbox, source, arguments, seed, and declared dependencies. It
does not make a flawed verifier correct. See [reward integrity](REWARD_INTEGRITY.md) and ADR 0043.
