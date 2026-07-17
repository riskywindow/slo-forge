# ADR 0046: Optimize learning value only inside serving-hard constraints

## Context

Combining serving latency and learning value in one soft score can sacrifice user traffic for a speculative gain.

## Decision

Reserve and validate serving resources first. Admit learning work only inside fault-adjusted capacity,
budget, privacy, effect, staleness, deadline, branch, and preemption bounds. Compare value-aware scheduling
against dedicated, static, utilization, and FIFO baselines with complete audit accounting.

## Consequences

Predicted learning value cannot override serving feasibility. Utilization may be lower, and results remain
conditional on supplied forecasts and value predictions rather than measured gains.
