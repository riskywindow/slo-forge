# ADR 0051: Export traces with artifact provenance and claim scope

## Context

An attractive dashboard or aggregate metric can conceal simulated inputs, missing samples, or incompatible clocks.

## Decision

Export canonical static JSON/HTML evidence with artifact paths and hashes, seeds, software/hardware manifests,
raw-sample references, uncertainty where computed, hypothesis status, and limitations. Keep generated metrics
distinguishable from observations and do not fabricate absent GPU or production measurements.

## Consequences

Reviews can trace claims to executed artifacts and reproduce local CPU results. Reports are larger and may
remain inconclusive; provenance does not improve the quality of the underlying experiment by itself.
