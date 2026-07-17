# Reward integrity

Security controls represent reward components as deterministic or hidden-test claims with evaluator,
trajectory, tenant, artifact, and hidden-boundary hashes. Aggregates are recomputed from component
score and weight; duplicate component identities, changed submissions, exposed hidden values, and
lineage mismatch fail closed.

Submission registries distinguish an exact duplicate from identity reuse with changed content and
are explicitly bounded. Tool output is sanitized and bound to evidence. Reward artifacts enter
promotion only through separate reward-integrity gates; the candidate cannot self-certify them.

SHA-256 detects change only when an expected digest is independently pinned. The local artifact is
not signed, and sandbox isolation varies by host capability. See [security](SECURITY.md).
