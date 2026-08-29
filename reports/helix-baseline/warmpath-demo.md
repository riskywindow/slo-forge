# WarmPath local demonstration

The profile uses measured local reads and checksum verification. The snapshot payload is an explicitly synthetic deterministic fixture.

- Plan: `warmpath-0a2979491e2c58b5` (exhaustive; 243 candidates)
- Predicted p50/p95 readiness: 0.835 / 0.886 ms
- Measured local execution readiness: 4.379 ms
- Restore/checksums: pass
- Deferred non-critical artifacts: 1

All reported values are loaded from `artifacts/warmpath/manifest.json`; raw stage samples are retained under `artifacts/warmpath/profile/raw/`.
