# Helix security

Helix treats production evidence, hidden tests, tools, effects, training artifacts, and promotion as
separate trust boundaries. Strict models accept summaries and digests rather than raw secrets at
authorization boundaries. Unknown fields fail validation.

Implemented controls include tenant-matched production capture grants with expiry; opt-in and
redaction evidence; path and symlink safety; tenant-scoped artifact graphs; retention and deletion
receipts; repository secret scanning; bounded tool-output sanitization; hidden-test isolation;
deterministic reward aggregation; duplicate-submission detection; execution-isolation declarations;
and effect legality checks. Production capture, external effects, and cross-tenant reuse are disabled
unless explicitly authorized. Irreversible speculative effects remain forbidden.

## Threats and responses

- **Cross-tenant disclosure:** tenant IDs are checked at capture, artifact, experience, and execution
  boundaries; cross-tenant reuse is off by default.
- **Reward gaming or leakage:** hidden expected values remain verifier-only during policy execution,
  reward components retain independent hashes, and promotion has a reward-integrity gate. In the
  source-controlled demo they are not secret from the benchmark author, so this is runtime dataflow
  isolation rather than independent test-set governance.
- **Artifact tampering:** canonical identities, payload validation, lineage links, and independently
  pinned expected digests detect changes.
- **Unsafe tools/effects:** outputs are sanitized; external writes require class-specific contracts;
  speculative irreversible or unknown effects are rejected.
- **Partial promotion:** transactional pointer update and rollback-parent CAS fail closed.

SHA-256 is integrity, not identity authentication. SQLite is not distributed consensus. The sandbox
cannot provide stronger isolation than its host primitives, and raw production governance still
depends on operator-controlled storage and access systems. See [trust model](TRUST_MODEL.md), ADR
0049, and [limitations](LIMITATIONS.md).
