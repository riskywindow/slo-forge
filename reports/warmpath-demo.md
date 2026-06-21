# WarmPath local demonstration

The profile uses measured local reads and checksum verification. The snapshot payload is an explicitly synthetic deterministic fixture.

- Plan: `warmpath-18c7910b4c2e3467` (exhaustive; 243 candidates)
- Predicted p50/p95 readiness: 0.879 / 0.904 ms
- Measured local execution readiness: 3.864 ms
- Restore/checksums: pass
- Deferred non-critical artifacts: 1

All reported values are loaded from `artifacts/warmpath/manifest.json`; raw stage samples are retained under `artifacts/warmpath/profile/raw/`.
