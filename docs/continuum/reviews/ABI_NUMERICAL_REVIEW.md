# Continuum state ABI and numerical-correctness review

Date: 2026-08-02
Scope: portable ABI, compatibility analysis, reference adapters/runtime, CPU layout conversion, optional PyTorch conversion, and Rust/Python conformance.

## Verdict

The reviewed CPU reference path is a genuine logical/physical state ABI rather than a shape-only KV relay. The exercised path preserves attention KV, recurrent state, sampler counter state, guided-decoding state, token history, and client watermarks across the reference token-major TP=4 and head-major TP=2 adapters. The optimized attention converter is independently compared with the canonical converter, and destination continuation is replay-validated.

All high-severity and reasonable medium-severity findings in this scope were patched. The reviewed CPU tests and Rust conformance checks pass. PyTorch and CUDA conversion are implemented as explicit optional paths but are not exercised on this host; no GPU correctness or performance claim is supported by this review.

## Findings and disposition

| Severity | Finding | Disposition |
| --- | --- | --- |
| High | Capsule Merkle roots did not bind identity metadata such as capture timestamp, Git commit, capsule type/parent, or top-level namespaced extensions. Those fields could be altered without changing the capsule ID. | Fixed in Python and Rust. `CapsuleIntegrity` now records identity and extension hashes, and the Merkle root commits both. Identity/extension tampering is rejected in both languages. |
| High | Capsule validation did not require physical component-size references to exactly match the logical component set, nor bind the identity's runtime/physical plan/adapter to the physical and logical records. | Fixed with cross-layer component-set, runtime, physical-plan, adapter, lease-owner, transaction-evidence, and watermark invariants. |
| High | Equal full weight hashes with conflicting derived state-producer/output-head hashes could be accepted. Adapter changes could also be "proven" by dependency evidence that did not name the adapter. | Fixed. Contradictory fingerprints and incomplete adapter/output-head dependency evidence fail closed. |
| High | The converter accepted arbitrary numeric dtype pairs, including lossy integer narrowing or float/integer conversion, and classified every dtype change as numerical equivalence. | Fixed. Safe widening is exact-semantic, float narrowing/widening is explicitly numerical, and arbitrary lossy integer/float conversion requires a separate quality-bounded backend and is rejected by the current compiler. |
| High | Conversion evidence's zero error measured only optimized versus canonical destination output. It could obscure source-to-destination loss during float narrowing. | Fixed. Evidence separately reports direct-versus-canonical error, source-to-destination maximum absolute error, declared exactness, requested tolerance, and whether that tolerance was met. |
| High | Reference snapshot validation allowed a missing or duplicate page-table mapping even when paged attention segments existed. | Fixed. Every paged segment must appear exactly once in a unique page-table entry with matching placement, range, version, dirty epoch, and owner epoch. |
| Medium | Shard rank was not constrained by tensor-parallel degree; the original golden fixture contained invalid TP=1 scalar ranks. | Fixed in Python and Rust. The fixture and cross-language golden hashes were regenerated deterministically. |
| Medium | Runtime changes were exact-semantic without a declared shared logical-state contract. | Fixed. Runtime capabilities publish a versioned logical-state contract, and mismatches are incompatible. |
| Medium | External runtime version parsing accepted arbitrary suffixes through a prefix regex. | Fixed. Only complete stable semantic versions, explicit post releases, and local-build suffixes are accepted; prerelease/dev/suffix smuggling is rejected. |
| Medium | Logical recurrent rows, seed range, head divisibility, and reference layout kind versus ordering/packing were not fully cross-validated at the adapter boundary. | Fixed with fail-closed manifest and layout validation. |
| Medium | Quantization descriptors could label an arbitrary non-identity format bitwise exact. | Fixed. Non-identity quantized representations cannot claim `EXACT_BITWISE`; quality-bounded formats require a typed quality contract. |
| Medium | The direct compiler declared two bounded scratch buffers although its iterator executes one payload buffer at a time. | Fixed to one bounded buffer. The destination allocation remains separately accounted in the memory plan. |

## Compatibility review

The engine now rejects matching-shape state reuse for changed state-producing weights, mismatched tokenizer/special-token/RoPE/attention/recurrent/sampler semantics, unsupported state, runtime logical-contract mismatch, contradictory weight fingerprints, and unproven adapter changes.

The recomputation regression discovered during this review was corrected without weakening fencing or direct-reuse rules. One exercised reference adapter conservatively maps its full-model hash to `output_head_hash`; therefore an attention revision appeared to change both state producer and output head. A state-producer change may still use recomputation when dependency evidence proves the output head is a state sink and explicitly identifies every affected/recomputable state component. When state-producing weights are unchanged, an output-head revision must still explicitly name `output_head` in the dependency evidence.

Numerical equivalence is conditional on the request's declared tolerance and destination verification obligation. Quality-bounded quantization or otherwise lossy dtype conversion requires measured evidence with an artifact reference and sample count. Compatibility classification does not imply that a matching executable quality-bounded converter is present; compilation must still fail if no such backend exists.

## Conversion backend status

| Backend | Status | Device/status evidence |
| --- | --- | --- |
| Trusted canonical NumPy | Implemented and exercised | CPU only; full logical K/V materialization; compared in the focused suite. |
| Vectorized direct NumPy | Implemented and exercised | CPU only; chunked direct relayout/reshard/page remap in both layout directions with deterministic seeds, partial pages, TP changes, empty state, and non-contiguous positive-stride fixtures. |
| Live reference-capture bridge | Implemented and exercised | CPU int32 attention bytes; token-major/separate TP=4 to head-major/packed TP=2; byte-compared with canonical encoding. Non-KV state uses the trusted typed codec and continuation replay. |
| PyTorch explicit-device backend | Implemented but unexercised on this host | PyTorch package probe reports `package_not_installed`. The backend requires explicit `cpu` or `cuda`, rejects negative-stride implicit copies, uses bounded chunk tensors, and returns only after complete physical-byte comparison with canonical CPU output. |
| CUDA through PyTorch | Implemented but unexercised | CUDA is never selected implicitly. Explicit `cuda` fails if unavailable and never falls back to CPU. No CUDA hardware or PyTorch installation was present. |
| Triton/custom CUDA kernel | Not implemented | No generated GPU kernel exists, so there is no generated-kernel verification or performance claim. |
| Quantized quality-bounded converter | Not implemented | The compatibility contract can describe a proven quality budget, but the reviewed KV compiler fails closed for lossy integer/float conversion and has no quantization packing backend. |

The NumPy memory bound covers converter-owned logical chunk arrays; NumPy allocator internals and temporary cast workspaces are not independently observable. The PyTorch evidence similarly reports controlled tensor bytes, not framework allocator fragmentation or kernel workspace. These are scoped bounds, not whole-process peak-memory proofs.

## Runtime adapter status in this scope

- Deterministic reference adapters: exercised across capture, direct attention conversion, import, validation, owner change, and continuation, including recurrent/sampler/guided state.
- PyTorch binding and conversion: implemented, version-gated, package absent, unexercised.
- Genesis binding: its generated runtime loader/smoke path is exercised elsewhere, but schema 1.0.0 does not expose portable active-state export and fails closed for that capability.
- vLLM and SGLang bindings: version/public-API gated byte-movement/configuration hooks only. Packages are absent, and both explicitly reject claims of a complete portable execution-state export.
- GPU and multi-GPU hardware: absent; simulated topology/layout coverage is not hardware validation.

## Remaining limitations

1. The live direct capture bridge currently targets the head-major packed reference destination; the generic NumPy converter supports both directions, but not every live-adapter combination has a byte bridge.
2. The direct converter covers attention K/V. Recurrent, sampler, guided-decoding, delivery, and token state are migrated through typed canonical reference codecs rather than direct tensor fusion.
3. NumPy does not provide the reviewed path with native bfloat16/FP8 semantics. Those formats need an independently verified backend and explicit numeric/quality contracts.
4. The canonical JSON profile is the project-defined sorted compact JSON profile, not a claim of RFC 8785 conformance. Rust/Python parity is covered by the shared capsule fixture plus representative exponent, negative-zero, escaping, and Unicode cases.
5. Bounded tests and deterministic randomized seeds are evidence within their declared shapes and limits, not a universal proof over all tensor dimensions or runtime versions.

## Acceptance evidence

Commands executed from the repository root:

```text
PYTHONPATH=python .venv/bin/pytest -q \
  tests/python/test_continuum_abi_review.py \
  tests/python/test_continuum_ir.py \
  tests/python/test_continuum_compatibility.py \
  tests/python/test_continuum_conversion.py \
  tests/python/test_continuum_runtime.py \
  tests/python/test_continuum_runtime_dirty.py \
  tests/python/test_continuum_external_adapters.py \
  tests/python/test_continuum_capture.py \
  tests/python/test_continuum_migration.py \
  tests/python/test_continuum_flagship.py
# 120 passed

.venv/bin/ruff check tests/python/test_continuum_abi_review.py \
  python/sloforge/continuum/{ir,compatibility,conversion,reference,adapters}
# passed

.venv/bin/mypy --strict python/sloforge/continuum/ir \
  python/sloforge/continuum/compatibility \
  python/sloforge/continuum/conversion \
  python/sloforge/continuum/reference \
  python/sloforge/continuum/adapters
# passed for 30 source files

cargo fmt --all -- --check
cargo clippy -p sloforge-continuum-ir --all-targets -- -D warnings
cargo test -p sloforge-continuum-ir
# 6 conformance/property tests passed; doc tests passed
```

The regenerated canonical whole-capsule hash is `18fbe6c114ec019839072c03213c1bf35efc72d650a668dee2addbb89a7215ee`. The content-addressed capsule ID (Merkle root) is `a87547ccd8c5cab2c660464e7c281588b3ca1a58ec45b656b119f2674fb2dec5`.
