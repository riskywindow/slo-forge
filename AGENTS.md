# SLOForge contributor rules

- Preserve deterministic behavior: every simulation, optimizer, and demo path accepts an explicit seed.
- Python owns compiler orchestration, modeling, profiling, optimization, control, exporters, and reports.
- Rust owns the gateway data plane, deterministic simulator, load generation, and shared wire types.
- The primary Rust/Python boundary is versioned JSON over subprocess stdin/stdout. HTTP/SSE is used only for the running data plane. Keep both schemas backwards compatible within an IR major version.
- Core queues and subprocesses must be bounded and have timeouts. Never silently switch devices or engines.
- Generated metrics must include provenance to raw samples. Do not commit fabricated measurements.
- Cloud deployment generation is offline by default. Creating paid resources requires `SLOFORGE_GPU_BUDGET_USD` and an explicit deployment command.
- Run targeted tests after edits and `make check` before handoff.
- Use `apply_patch` for source edits. Do not overwrite unrelated user changes.
