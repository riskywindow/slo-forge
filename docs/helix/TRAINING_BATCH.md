# Training batch

`build_reference_training_batch` joins trajectories, rewards, branch-relative credit, holdout
designation, and explicit staleness distances. Every sample carries trajectory, reward, credit,
state, behavior-policy, and action provenance. The manifest binds ordered samples and split to a
content identity.

Duplicate trajectories or reward submissions, absent rewards, foreign credit, holdout leakage,
mixed strict-policy epochs, negative staleness, action mismatch, and data-hash mismatch are rejected.
Ineligible or stale samples remain visible with their disposition instead of disappearing from
accounting.

The reference builder validates supplied evidence; it does not independently inspect a remote data
lake. See [trainer adapter](TRAINER_ADAPTER.md) and [staleness](STALENESS.md).
