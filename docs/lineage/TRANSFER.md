# Lineage transfer

Lineage transfer seeds a new search with previously accepted transformations that are applicable to the target task. Reuse is proposal initialization only: every `RelatedTransformation` has `requires_reverification=true`.

## Applicability and retrieval

A transformation is eligible only when:

- its outcome is `accepted`;
- target model family and hardware architecture are explicitly applicable;
- target operator and workload sets overlap declared applicability;
- every dependency precondition matches a target dependency;
- no matching candidate/family learned constraint excludes it;
- at least one attached evidence record is passing, fresh, unexpired, and not from the future.

Effective evidence confidence decays exponentially from its declared base confidence, with a configurable 90-day default half-life. Stale, failed, inconclusive, expired, or future-dated evidence contributes zero. Evidence relevance scores model family, hardware, workload overlap, and exact dependency versions.

Task similarity combines model family, operator Jaccard similarity, hardware, workload Jaccard similarity, and topology Jaccard similarity. The final deterministic score combines task similarity, effective evidence confidence, a bounded expected-benefit bonus, and prior transfer history. An improved transfer adds a small relevance-weighted bonus; rejected and negative transfers reduce later ranking but remain visible. SHA-256 of the explicit seed, target task, and transformation provides the final deterministic tie break.

## Search initialization

`initialize_search_from_lineage` fills at most the configured lineage fraction. The fraction is capped at 0.8, preserving at least 20 percent of the population as deterministic unseeded proposals. Missing or inapplicable lineage therefore reduces seeding rather than shrinking the requested population.

Every executed reuse should produce a `TransferRecord` containing target task, transformation, source evidence, retrieval score/rank, seed, outcome, rationale, and time. Recording negative transfer is required for future ranking and analysis; it is not discarded as search noise.

## Exercise status

Tests show related lineage ranking ahead of unrelated lineage, deterministic initialization, preserved unseeded diversity, negative-transfer penalty, learned-constraint exclusion, and stale-evidence exclusion after dependency invalidation. They do not execute a complete empty/unrelated/related/stale-lineage performance campaign, compile the retrieved transformation into a runtime, or measure time-to-improvement. The current transfer module produces a checked search initialization; the main search/compiler must revalidate and persist the eventual outcome.

