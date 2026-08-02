# Dependency-aware invalidation

Lineage never silently reuses evidence after a known dependency incompatibility. Invalidation preserves the original result and marks its current applicability stale.

## Dependency selectors

Evidence dependencies are typed as driver, compiler, runtime, library, hardware, model contract, or workload contract and include name, version, and optional content hash. An invalidation event declares kind, name, supported numeric version range, reason, and timezone-aware occurrence time.

The implemented version matcher accepts numeric one-to-three-component versions, exact values, `<`, `<=`, `>`, `>=`, comma conjunctions, `*`, and `x`/`*` component wildcards. Prerelease/build suffixes are ignored for numeric comparison. It is a deliberately small range language, not a complete PEP 440, Cargo, or npm semver implementation.

## Transaction

`invalidate_dependency`:

1. selects fresh evidence with the exact dependency kind and name;
2. enforces the explicit maximum-evidence bound before any durable change;
3. applies the version predicate in the trusted local matcher;
4. inserts one immutable invalidation event;
5. updates every affected evidence record to `stale` and appends the event ID;
6. commits all changes together.

Duplicate event identity, foreign-key failure, timeout, or an affected set beyond the configured maximum rolls the transaction back. No partially stale result is exposed. Stale evidence has effective confidence zero and is excluded from transfer retrieval, but remains queryable and exportable with its invalidation edge.

## Operational use

The CLI exposes explicit invalidation by dependency, kind, version range, reason, database, and maximum affected evidence. Revalidation creates new evidence; invalidation does not mutate old raw samples into a pass, revive a candidate, or automatically rewrite a transformation's applicability declaration.

## Exercise status and limits

Tests exercise selective Triton-version invalidation, confidence dropping to zero, retrieval exclusion, and full rollback when the affected set exceeds its bound. Content-hash-only selection, recursive invalidation through an arbitrary dependency graph, automated driver/compiler watchers, and revalidation job scheduling are not implemented by this store. Model, workload, hardware, or dependency changes have effect only when they are recorded as evidence dependencies and an explicit invalidation event is issued.

