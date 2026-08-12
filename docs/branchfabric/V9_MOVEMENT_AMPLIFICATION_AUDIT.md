# v9 Movement Amplification Audit

This audit is entirely offline and is pinned to immutable v9 raw-artifact hashes.

## Result

The raw 54x ledger is duplicate-free internally but noncanonical and incomplete: it mixes 51x endpoint touches with 3x link traversal while omitting a real redundant 4x restore-import verification. Corrected memory touch is 55x; explicit full read/write/link work is 58x.

| Metric | Bytes | Amplification |
|---|---:|---:|
| Legacy composite (noncanonical) | 57,076,088,832 | 54.0× |
| Corrected memory endpoint touches | 58,133,053,440 | 55.0× |
| External D2H + H2D movement | 3,170,893,824 | 3.0× |
| Full read + write + link work | 61,303,947,264 | 58.0× |
| Conservative avoidable endpoint work | 32,765,902,848 | 31.0× |
| Critical-path read + write + link work | 61,303,947,264 | 58.0× |

The legacy numerator has no duplicate event IDs. Its problem is dimensional: it combines endpoint reads/writes and link traversal into one score, then misses a second restore-time verification. The corrected memory-only metric is 55×. If the requested full-touch definition literally includes bytes read, bytes written, and bytes moved over PCIe, the result is 58×; external movement remains separately reported as 3×.

## Exact 2× D2H explanation

The 2,113,929,216 D2H bytes are three real synchronous CUDA copies: 1,056,964,608 bytes for checkpoint capture, 954,204,160 bytes for validation of the shared root plus branch 0, and 102,760,448 bytes for validation of branches 1-7. The validation subsets are disjoint and sum to one logical state. K and V are already included in the 1,056,964,608-byte denominator. Duplicate D2H accounting is zero.

## Missing physical pass

`Vllm0230RestoreStager.import_group()` calls `state.verify()` after the worker already completed and recorded a full restore verification. The immutable payload is unchanged. This unrecorded verification adds one pinned-host read, one pageable-host write, one aggregate-hash read, and one set of page-hash reads: 4× logical bytes of real endpoint work, 2× checksum work, and one logical payload of transient allocation. The raw import-to-first-allocation gap and RSS drop independently corroborate the source-path reconstruction.

## Physical pass groups

| Label | Stage | Processor | R | W | Link | Temp | Required | Artifact | Fusion |
|---|---|---|---:|---:|---:|---:|---|---|---|
| `capture-source-native-read` | capture | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | true | false | SOFTWARE_FUSIBLE |
| `capture-native-axis-contiguous` | capture | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `capture-unpage-valid-tokens` | transform | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `capture-stack-layers` | transform | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `capture-concatenate-pages` | transform | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `capture-d2h` | d2h | GPU+CPU | 1,056,964,608 | 1,056,964,608 | 1,056,964,608 | 0 | true | false | CANNOT_FUSE |
| `capture-pinned-transport-lifetime` | checkpoint_lifetime | CPU | 0 | 0 | 0 | 1,056,964,608 | true | false | CANNOT_FUSE |
| `capture-integrity-manifest` | integrity | CPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | true | true | SOFTWARE_FUSIBLE |
| `capture-integrity-hash-reads` | integrity | CPU | 2,113,929,216 | 0 | 0 | 0 | true | true | SOFTWARE_FUSIBLE |
| `transport-publish-validation` | publish | CPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `transport-publish-hash-reads` | publish | CPU | 2,113,929,216 | 0 | 0 | 0 | false | true | SOFTWARE_FUSIBLE |
| `restore-transport-validation` | restore_integrity | CPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | true | true | SOFTWARE_FUSIBLE |
| `restore-transport-hash-reads` | restore_integrity | CPU | 2,113,929,216 | 0 | 0 | 0 | true | true | SOFTWARE_FUSIBLE |
| `restore-h2d` | h2d | CPU+GPU | 1,056,964,608 | 1,056,964,608 | 1,056,964,608 | 1,056,964,608 | true | false | CANNOT_FUSE |
| `restore-import-validation` | state_import | CPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `restore-import-hash-reads` | state_import | CPU | 2,113,929,216 | 0 | 0 | 0 | false | true | SOFTWARE_FUSIBLE |
| `restore-zero-native-pages` | state_import | GPU | 0 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `restore-overlay-valid-tokens` | state_import | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 0 | false | true | SOFTWARE_FUSIBLE |
| `restore-native-axis-contiguous` | state_import | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `restore-stack-native-pages` | state_import | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `restore-destination-native-write` | state_import | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 0 | true | false | SOFTWARE_FUSIBLE |
| `validation-destination-native-read` | validation | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | true | false | CANNOT_FUSE |
| `validation-native-axis-contiguous` | validation | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `validation-unpage-valid-tokens` | validation | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `validation-stack-layers` | validation | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `validation-concatenate-pages` | validation | GPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `validation-d2h` | validation | GPU+CPU | 1,056,964,608 | 1,056,964,608 | 1,056,964,608 | 0 | false | true | SOFTWARE_FUSIBLE |
| `validation-expected-page-concatenation` | validation | CPU | 1,056,964,608 | 1,056,964,608 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |
| `validation-host-tensor-compare` | validation | CPU | 2,113,929,216 | 0 | 0 | 0 | false | true | SOFTWARE_FUSIBLE |
| `validation-recapture-host-lifetime` | validation | CPU | 0 | 0 | 0 | 1,056,964,608 | false | true | SOFTWARE_FUSIBLE |

Allocation-lifetime records are semantic bookkeeping and do not contribute movement bytes. The JSON audit contains the required classification for every recorded and inferred segment-level pass.

## Fusible chains

Capture: `READ_NATIVE → CONTIGUOUS → UNPAGE → STACK → CONCAT → D2H → MATERIALIZE/HASH → PUBLISH_VERIFY`. The GPU packing steps are software-fusible; canonical output can be streamed to D2H; hash generation can consume that stream; publish re-verification is removable.

Restore: `VERIFY → H2D → DUPLICATE_VERIFY → ZERO/OVERLAY → CONTIGUOUS → STACK → NATIVE_WRITE → READBACK → REPACK → D2H → EXPECTED_CONCAT → COMPARE`. The duplicate verify is removable. Full v9 pages make zero-fill removable. Packing and native write are software-fusible. The write/readback barrier is mandatory for the current proof, while device-side exact proof could remove validation D2H and both host validation buffers.

No fusion or preservation optimization was implemented in this task.

## Provenance

- `artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/exp004-pilot-naive-s41-v9/rollout/result.json` (`5bc1167139165cc36792186ae607e7a0d971aa8f614f784f5cf93ac5c2d7ae2e`)
- `artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/exp004-pilot-naive-s41-v9/rollout/telemetry/cuda-and-host-operations.json` (`cb0841111a2f80bb6d0227e80445a7b0464444aa807a1a1720a9a29bde9c81f4`)
- `experiments/branchfabric/gpu_reclamation_worker.py`
- `python/sloforge/continuum/adapters/vllm_reclamation.py`
- `python/sloforge/helix/characterization/gpu_reclamation_accounting.py`
