# ADR 0021: Keep privileged telemetry optional

- Status: accepted
- Date: 2026-08-01

## Context

DCGM, eBPF, CUPTI, device counters, network namespaces, traffic control, and
clock changes can require privileges or mutate host state.

## Decision

The canonical Autopsy event model accepts those sources, but normal capture and
CI require none of them. Privileged probes and fault injection use separate
false-by-default environment controls. Discovery is read-only. Missing sources
produce warnings and reduced confidence, not invented counters.

## Consequences

CPU CI validates event normalization and causal logic safely. Hardware captures
may be less precise without optional probes. Operators can reason about the exact
evidence gap because source and alignment confidence are retained.

