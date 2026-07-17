# Environment state capsule

`EnvironmentStateCapsule` is the immutable description from which the local backend reconstructs an
environment branch. It records files and content digests, Git state, dependency locks, runtime and
service descriptors, policies, resource bounds, storage accounting, seed, and a capsule identity.

The content store verifies bytes against their digests. Path normalization rejects absolute paths,
parent traversal, unsafe symlinks, device entries, and paths outside the captured root. Secret-like
material is redacted according to capture policy; production capture requires separate authorization
and redaction evidence. Branch workspaces are tenant-scoped and bounded.

The capsule represents the fields the backend knows how to capture. It is not a VM image and cannot
implicitly preserve kernel state, remote services, credentials, packet timing, or undeclared external
effects. See [environment branching](ENVIRONMENT_BRANCHING.md) and [security](SECURITY.md).
