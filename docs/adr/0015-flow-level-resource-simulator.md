# ADR 0015: Use a flow-level resource simulator

- Status: accepted
- Date: 2026-08-01

## Context

Physical inference depends on shared GPU, PCIe, NIC, and rail capacity, but a
packet simulator would add detail unsupported by available measurements.

## Decision

Model dependency-graph operations on exclusive or fair-share physical resources
and explicit sharing groups. Use message-size service curves for links and
calibrated durations for compute/startup. Model aggregate rate, availability,
rank, and collective faults. Do not model individual packets.

## Consequences

The simulator captures queueing, contention, barriers, overlap, and stragglers
and runs faster than the fixture wall clock. Packet-level effects must be folded
into calibrated latency/rate/jitter curves, so predictions are only as good as
their profile coverage.

