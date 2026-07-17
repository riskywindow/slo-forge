# Helix reproducibility

The checked Helix path is deterministic local CPU code. Requests carry explicit seeds; strict models
forbid unknown data; canonical JSON and SHA-256 bind inputs and outputs; bounded loaders reject
oversized documents; and raw/summarized evidence remain connected by artifact paths and hashes.

Primary reproduction surfaces are the Helix-focused pytest files, `sloforge helix demo`,
`sloforge helix evaluate`, scheduler scenarios under `scenarios/helix/resource`, the experience
scenario, and the CPU fault matrix. The evaluation records software/hardware manifests, multiple
seeds, intervals, artifact paths, hypothesis statuses, and limitations. Run `make check` before
interpreting artifacts.

Determinism means identical declared inputs produce identical logical outputs. Wall-clock sandbox
durations, filesystem/SQLite implementation details, and real hardware timing can vary. Checked-in
synthetic/local results must not be relabeled as production measurements. See
[demo script](DEMO_SCRIPT.md) and ADR 0002.
