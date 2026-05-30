# WarmPath local reference benchmark

`local-profile.yaml` is the versioned input template used by the CPU-only WarmPath path. The
benchmark measures local artifact reads and checksum verification after explicit warmups, retains
all raw samples, compiles a constrained storage plan, and materializes it with the local executor.

The template contains no benchmark result values. Reported latency, dispersion, and confidence
intervals are generated from the raw files referenced by the resulting `StartupProfile`.
