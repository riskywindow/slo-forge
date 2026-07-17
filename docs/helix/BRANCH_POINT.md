# Branch point

A `BranchPoint` is the atomic publication boundary for counterfactual work. It binds the source
trajectory and policy epoch to model state, environment state, effect state, and their independent
watermarks. The coordinated capture implementation publishes it only after quiescence and cross-source
validation complete.

No component is treated as interchangeable merely because its timestamp looks close. Watermarks are
typed by domain. A mismatch aborts capture without partially publishing a branch point. Repeating an
identical capture is idempotent; reusing an identity for different content is rejected.

The branch point permits later forks; it does not assert that every fork is exact. Each branch carries
a `StateReuseReport` and replay declares exact, causal, or semantic comparison scope. See
[coordinated capture](COORDINATED_CAPTURE.md) and [model-state branching](MODEL_STATE_BRANCHING.md).
