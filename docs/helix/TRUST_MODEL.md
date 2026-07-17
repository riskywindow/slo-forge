# Helix trust model

Helix trusts strict validators, canonicalization, the local coordinator/registry transactions, the
effect legality checker, independently configured gates, and the host mechanisms used by the sandbox
and content store. It does not trust a proposal, candidate policy, reward scalar, artifact filename,
or self-declared `verified` label by itself.

Evidence authority flows from independently pinned digests and configured issuers into bounded,
hash-addressed records. A promotion capsule may summarize authority but cannot manufacture it.
Generated or learned behavior remains untrusted until it passes the applicable validators and gates.

Out of scope are compromised host kernels, stolen signing/access credentials, malicious external
services that violate declared contracts, distributed Byzantine coordination, and correctness of an
incorrect verifier. There is currently no public-key signature layer for Helix capsules.
