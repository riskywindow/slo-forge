# ADR 0002: Determinism and evidence are API properties

Status: accepted

Simulation, mock profiling, optimization, control evaluation, and report generation require explicit seeds. Every derived result points to input hashes and raw samples. Timestamps and host metadata may vary, but deterministic payloads are separated from environment envelopes. Offline exporters never deploy resources.
