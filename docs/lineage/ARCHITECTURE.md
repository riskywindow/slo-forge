# Optimization lineage architecture

Genesis lineage is an embedded, transactional evidence graph for retaining both successful and unsuccessful optimization work. It stores typed facts; it does not trust prior conclusions as current correctness evidence.

## Record model

Schema version `1.0.0` defines strict immutable records for:

- tasks and their model, operator, workload, topology, hardware, dependency, and contract features;
- candidate genomes, parents, disposition, objective evidence, causal bottleneck, and exposed next bottleneck;
- transformations, preconditions, applicability domains, dependencies, expected benefit, proposal source, and outcome;
- passing, failing, or inconclusive evidence with content hash, scope, dependencies, confidence, time bounds, and freshness;
- minimized counterexamples and learned constraints;
- transfer attempts and their seeded, reverified, improved, rejected, or negative-transfer outcomes;
- dependency invalidation events.

All core fields are typed and reject unknown data. Identifiers are immutable primary keys. Candidate parent, transformation, evidence, counterexample, constraint, and transfer references must already exist. A candidate plus its newly produced transformations and evidence can be inserted as one transaction; any conflict rolls the complete bundle back.

## SQLite storage

`LineageStore` uses SQLite with foreign keys enabled, write-ahead logging, `synchronous=FULL`, a bounded busy timeout, and database `user_version=1`. Indexed dependency, evidence-target, and constraint-family tables support the current queries. Every listing, query, snapshot, and invalidation has an explicit bound; SQLite's progress handler enforces a wall-clock query deadline.

The store is append-oriented. Existing identifiers are conflicts rather than updates. Dependency invalidation is the deliberate exception: it atomically changes matching evidence from `fresh` to `stale`, appends the invalidation reference inside the evidence document, and records the event. It never rewrites raw evidence content or deletes negative results.

## Graph and exports

Portable JSON exports contain a complete bounded `LineageSnapshot`. GraphML exports materialize task, candidate, transformation, evidence, counterexample, constraint, transfer, and invalidation nodes. Directed edges represent containment, ancestry, proposal/production, support, invalidation, rejection, constraint generalization, transfer, target, and source-evidence justification.

Both exporters gather one bounded snapshot and write through a same-directory temporary file, flush, `fsync`, and atomic replacement. Record ordering is stable by identifier. GraphML is a view of the transactional records, not the primary database.

## Interfaces and status

The `sloforge lineage` CLI supports filtered transformation queries, explanations with evidence/constraints/counterexamples, dependency invalidation, and JSON/GraphML export. Programmatic transfer retrieval and search initialization live in `sloforge.lineage.transfer`.

Focused tests exercise strict models, persistence across reopen, foreign/reference validation, atomic bundle rollback, bounded invalidation rollback, filtered queries, counterexample/constraint/transfer persistence, and parseable JSON/GraphML exports. The lineage store is exercised locally; replication, concurrent multi-writer service operation, remote databases, automatic ingestion from every Genesis subsystem, and large-scale performance are not established by these tests.

