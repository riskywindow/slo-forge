# Side-effect system

Helix classifies effects as pure, read-only, idempotent write, compensatable write, irreversible
write, or external unknown. Every effect has a content-derived identity, tenant, operation, target,
and the metadata required by its class. Unstable reads, idempotent writes without keys, and
compensatable writes without compensation recipes are illegal.

`EffectLedger` records application, commitment, compensation, evidence, virtual time, and watermarks.
Speculation cannot run irreversible or unknown real effects. Real external writes are disabled unless
explicitly enabled, and the local rollout worker refuses unsafe effect classes. Commitment and
compensation are distinct state transitions; compensation is not claimed where no recipe exists.

The ledger is an auditable local mechanism, not a distributed transaction manager for arbitrary
third parties. An external system can still violate an idempotency or compensation promise. See
[security](SECURITY.md) and ADR 0039.
