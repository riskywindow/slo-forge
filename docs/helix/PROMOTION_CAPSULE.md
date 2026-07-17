# Promotion capsule

`TrustedPolicyPromotionCapsule` is the closed handoff from learning to deployment admission. It binds
candidate and parent epochs, training and evaluation artifacts, reward/lineage/security evidence,
rollback material, Continuum compatibility, software scope, seed, and content identity.

Validation checks artifact digests and sources, required independent gates, compatibility bindings,
tenant and policy identities, and rollback completeness. The registry consumes validated evidence;
the training process cannot promote by emitting a favorable scalar.

The capsule is hash addressed, not signed. Authenticity requires an expected digest or trust anchor
outside the capsule. See ADR 0047 and [trust model](TRUST_MODEL.md).
