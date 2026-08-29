# Experiment 004 Baseline Campaign Plan

Status: **PLAN ONLY — NO CAMPAIGN ARM WAS EXECUTED BY THIS GENERATOR**.

## Arms

A. `KILL_AND_RECOMPUTE`

B. `NAIVE_PRESERVE` — the frozen scientifically valid v10 baseline

C. `OPTIMIZED_PRESERVE` — future v11 software path

## Fixed experimental controls

- λ₁: `12.0` requests/s
- λ₂: `20.0` requests/s
- λ_spike: `15.0` requests/s
- Hardware: the same exact two-A100-80GB topology and runtime pins as v10
- State: the same 16,384-token shared prefix, eight branches, and at least 256-token divergent suffix
- Seeds: `41, 73, 113, 149, 197`; arm order is SHA-256-seeded and position-balanced
- Checks: identical output length, trigger, recovery, restore, first-token, continuation, fresh-allocation, and cleanup gates
- Accounting: identical logical denominator and complete physical pass graph for every arm

Analyze paired per-seed differences; do not report unsupported tail percentiles. A failed arm remains in the audit trail and is never silently retried or replaced.

This plan does not authorize execution, Experiment 005, or hardware implementation.
