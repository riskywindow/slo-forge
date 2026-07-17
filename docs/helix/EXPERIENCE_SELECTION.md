# Experience selection

Helix selects a bounded subset of synthetic or explicitly authorized production evidence. Signals
include failure, verifier and reward disagreement, policy/value uncertainty, novelty, rarity, safety,
recurrence, Autopsy issue, capability regression, branchability, and expected learning value per
cost. Required baselines are seeded random, failure-only, uncertainty-only, and novelty-only.

Consent, authorization artifacts, redaction evidence, tenant identity, privacy class, and effect risk
are hard gates. Exact content fingerprints prevent redundant selection. Budget, capacity, item-count,
minimum-score, and anti-train-all limits are explicit. Every candidate receives a selected or excluded
decision with scores, reasons, prediction uncertainty, assumptions, candidate digest, and artifact
hashes; the complete plan is hash sealed.

Features and value estimates are input claims, not observed gains. Exact fingerprints do not detect
semantic near-duplicates without upstream clustering. See [resource compiler](RESOURCE_COMPILER.md).
