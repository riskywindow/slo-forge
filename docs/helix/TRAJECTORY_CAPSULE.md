# Trajectory capsule

The canonical `TrajectoryCapsule` and local `ReferenceTrajectory` preserve behavior-policy evidence
for learning. They bind branch and branch-point identities, source state, initial/final environment,
ordered events, tokens, actions, behavior log probabilities, sampler seed/counters, effect-ledger
hash, event-chain head, and terminal status.

Event indices are contiguous and hash chained. Strict mode requires one policy epoch across events,
tokens, and actions. Behavior log probabilities must be finite and non-positive. Missing probabilities,
mixed epochs, broken chains, or changed content fail validation rather than being repaired silently.

This capsule proves integrity of recorded fields, not truth of an unobserved environment. Exact joint
replay additionally requires compatible captured model and environment state. See
[trajectory replay](TRAJECTORY_REPLAY.md) and [staleness](STALENESS.md).
