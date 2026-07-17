# Environment branching

The local `EnvironmentBackend` reconstructs a validated capsule into isolated workspaces and creates
copy-on-write logical branches through a content-addressed store. Branch logs and comparisons retain
changed files and storage accounting. Resource limits, path safety, tenant isolation, and cleanup are
hard checks.

Copy-on-write here is primarily a logical/content-store property. The checked CPU evidence does not
measure filesystem reflink behavior, page-fault amplification, container isolation, or remote-service
cloning. Services are reconstructed only from declared descriptors; undeclared remote state remains
outside the branch. See [environment state capsule](ENVIRONMENT_STATE_CAPSULE.md) and ADR 0038.

Each fork starts from the capsule's captured files, declared services, runtime counters, and policy
metadata. A branch seed may replace the environment RNG seed, but it does not mutate or stand in for
the model sampler state; model RNG overrides are separate branch evidence. Cleanup removes the live
workspace, not the immutable source capsule or content-addressed objects. “Reset” therefore means a
new workspace reconstructed from declared capsule state, not a reset of undeclared kernel, network,
credential, or remote-service state.
