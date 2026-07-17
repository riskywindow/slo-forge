# ADR 0049: Fail closed on production evidence, privacy, and external effects

## Context

Online learning can expose tenant data, hidden tests, credentials, and tool effects through capture or training.

## Decision

Disable production capture, cross-tenant reuse, external effects, and raw hidden values by default. Require
tenant-matched grants, consent, redaction evidence, retention/deletion controls, artifact lineage, sandbox
isolation declarations, and bounded sanitized tool output.

## Consequences

Unsafe or incomplete evidence is excluded with reasons rather than accepted opportunistically. Operators
must manage grants and trust anchors; host compromise and malicious external services remain out of scope.
