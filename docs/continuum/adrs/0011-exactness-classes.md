# ADR 0011: Explicit ordered exactness classes

- Status: Accepted
- Date: 2026-08-02

## Context

Bitwise equality, semantic equality, numerical tolerance, measured quality, and recomputation carry different obligations.

## Decision

Use `EXACT_BITWISE`, `EXACT_SEMANTIC`, `NUMERICALLY_EQUIVALENT`, `QUALITY_BOUNDED`, `RECOMPUTATION_ASSISTED`, and `INCOMPATIBLE`. Every conversion declares its achieved class and evidence.

## Consequences

Precision or quantization loss cannot be hidden. A caller may require a stronger class than an available plan and receive rejection.
