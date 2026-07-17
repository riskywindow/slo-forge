# ADR 0039: Type and ledger every Helix side effect

## Context

Counterfactual execution is unsafe if a replay can repeat an email, charge, deletion, or unstable read.

## Decision

Classify effects as pure, read-only, idempotent, compensatable, irreversible, or unknown. Require
class-specific metadata, record applications and evidence in a watermark ledger, disable external
writes by default, and prohibit irreversible or unknown real effects during speculation.

## Consequences

The local runtime fails closed and effect state joins coordinated capture. External systems may still
violate their idempotency or compensation contracts; the ledger is not a general distributed transaction.
