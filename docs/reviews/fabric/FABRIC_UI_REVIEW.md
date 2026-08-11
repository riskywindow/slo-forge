# SLOForge Fabric UI and static-artifact integration review

Review date: 2026-08-01

Scope: `ui/src`, `ui/tests`, and `ui/README.md`. The review did not modify the
Python/Rust core, root manifests, project documentation, or generated demo and
evaluation artifacts.

## Verdict

The Fabric explorer now renders the flagship physical, causal, optimizer, and
recovery evidence from the actual regenerated `artifacts/fabric-demo` bundle.
The served-manifest path verifies every required component against the manifest
SHA-256 before parsing or rendering. It fails visibly on missing, duplicate,
oversized, malformed, digest-mismatched, semantically dangling, or
cross-artifact-inconsistent evidence.

The current checked bundle exercised by the tests contains 40 topology nodes,
88 topology edges, 16 rank bindings, 64 expert assignments, 16 collectives, one
KV route, 16 overlap windows, 97 synthetic-calibrated fabric measurements, 175
optimizer candidates, three Pareto candidates, 27 diagnosis hypotheses, seven
counterfactuals, three recovery actions, 12 recovery audit records, and 11
timeline events. These counts are read from artifacts, not duplicated in UI
source.

## Findings fixed

1. High: the loader test previously added an `optimizer.json` digest to an
   in-memory manifest, masking whether the real served demo was complete. The
   regenerated manifest now contains the real optimizer digest, and tests load
   it unchanged. Missing and duplicate required entries fail before any fetch.
2. High: the CPU demo's `synthetic_calibrated` profile was presented as
   “measured” in chart titles and KV/profile tables. Profile mode is now parsed
   from the canonical extension, cross-checked against
   `manifest.synthetic_hardware`, and displayed as “synthetic calibrated.” No
   hardware measurement claim remains.
3. High: component SHA-256 checks protected bytes independently, but the parser
   did not validate physical or causal relationships after loading. It now
   checks topology edge endpoints; rank-to-host/GPU/NUMA/NIC/rail affinity;
   parallel, replica, expert, collective, and KV rank references; KV edge paths;
   topology/plan/optimizer identities; diagnosis and counterfactual summaries;
   recovery diagnosis/plan identities; Pareto membership and selection; SLO
   attainment derivation; and timeline-to-manifest evidence references.
4. High: fetched manifest and component bodies were unbounded. Every remote JSON
   artifact is now streamed with a 50 MiB cap, including the manifest trust root.
   Declared and actual oversize responses fail before parsing.
5. Medium: the topology SVG showed host/rank placement but omitted explicit
   PCIe, NVLink, GPU-to-NIC, NUMA, and rail relationships. A complete
   artifact-derived edge table now accompanies the compact placement SVG.
6. Medium: `paretoCandidates` compared unrelated physical-plan and candidate ID
   formats, so no transform result was selected. It now uses the unique
   `optimizer_history` select event, which is cross-checked against the Pareto
   frontier.
7. Medium: recovery showed a decorative literal “rollback armed.” It now parses
   and renders the actual abort and rollback criteria, actual failed attempts,
   stream-preservation policy, and external-mutation authorization.
8. Medium: empty evidence arrays could produce plausible blank panels. Required
   flagship inputs now reject empty topology, profile, placement, expert, KV,
   optimizer, diagnosis, counterfactual, recovery, audit, and timeline data.
9. Medium: error-path tests covered only one altered digest. Tests now cover
   missing and duplicate manifest entries, malformed JSON with a matching
   digest, HTTP 404, declared oversize, digest mismatch, dangling physical
   references, inconsistent causal/recovery/optimizer IDs, synthetic-label
   mismatch, and missing required view inputs.

## View coverage

- Physical topology: host/rank SVG plus every typed topology edge, connection,
  contention/sharing domain, curve count, bandwidth, and health state.
- Execution mapping: rank GPU/NUMA/CPU/NIC/rail/fault-domain table, prefill and
  decode groups, parallel degrees, and hot-expert placement.
- Communication: actual collective operations, KV producer/consumer paths,
  compute/communication overlap, calibrated link curves, confidence intervals,
  raw-sample counts, and explicit profile mode.
- Performance: compiler predictions, healthy/degraded/restored replay metrics,
  resource hotspots, and the physical optimizer Pareto frontier with the
  selected candidate marked from optimizer history.
- Autopsy: diagnosed bottleneck, ground-truth faults, diagnosis confidence,
  physical target, supporting/contradicting evidence counts, and every
  counterfactual repair with interval and selection.
- Recovery: actual actions, attempts, state transitions, shadow/canary samples,
  stream preservation, abort/rollback guards, promotion, drain, and final state.
- Timeline: every displayed timestamp, event, detail, and evidence URI comes
  from the hash-bound timeline and references a manifest artifact.

## Trust and residual scope

- The served manifest is the trust root. The UI verifies all referenced
  component bytes but does not sign or independently authenticate the manifest.
- Browser file pickers cannot fetch sibling files. A pre-composed
  `sloforge.fabric.ui-bundle/v1` file receives strict schema and cross-reference
  validation, but byte-level component digests cannot be reconstructed from
  parsed embedded objects. The README directs integrity-sensitive use to the
  served-manifest path.
- The compact SVG focuses on hosts and rank placement; the accompanying table is
  the complete physical relationship view. This is intentional and avoids a
  decorative force-directed layout whose geometry would not be evidence.
- The UI shows representative shape-nearest profile rows and explicitly reports
  the displayed and total counts. The complete profile remains in the source
  artifact.
- Static Markdown/HTML reports are generated and validated by the Python report
  pipeline, not re-parsed by this browser UI. This review verified the UI's
  manifest/artifact integration and did not duplicate static-report validation.
- The current Fabric bundle is deterministic synthetic hardware evidence. No GPU
  or multi-node hardware result is inferred by the UI.

## Acceptance evidence

Executed against the regenerated Fabric demo artifacts:

```text
cd ui && npm run typecheck
tsc -b --pretty false: passed

cd ui && npm run lint
eslint .: passed

cd ui && npm test
8 files passed; 28 tests passed

cd ui && npm run build -- --outDir /tmp/sloforge-ui-review-dist-final
TypeScript build and Vite production bundle: passed
```

The loader success test fetches the actual manifest's required files from
`artifacts/fabric-demo`, recomputes all required SHA-256 values in the browser
path, and validates the composed cross-artifact bundle before component tests
render it. The actual manifest optimizer digest was independently observed as
`fad139a788bd9ca116f88bc89defdd92481dd91ba7fa90701f608c5a9320883b`.
