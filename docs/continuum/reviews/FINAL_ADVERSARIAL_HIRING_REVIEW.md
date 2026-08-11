# Continuum final adversarial, benchmark, clean-room, and hiring review

Review date: 2026-08-02

Scope: the integrated Continuum source tree, retained CPU artifacts and reports,
CLI/Make workflow, clean-room script, benchmark methodology, flagship semantics,
model-check evidence, core placeholder search, and the release claims requested in
the completion contract. This is a CPU/source review. No NVIDIA GPU, RDMA,
multi-node, vLLM, SGLang, Genesis, or PyTorch live-session migration was exercised.

## Release verdict

**The reviewed source has no remaining high-severity implementation finding, but
the release claim remains conditional on provenance gates.** The source must first
be committed, then all retained Continuum evidence must be regenerated against
that commit, and the committed-archive clean-room gate and final-report audit must
pass. The previous retained evaluation names the pre-Continuum baseline commit and
cannot be final release evidence for this source revision.

Continuum is materially broader than a KV-transfer wrapper. Its strongest verified
result is a deterministic CPU research substrate with a portable logical/physical
state ABI, direct and canonical layout conversion, non-KV state, durable ownership
and gateway token fencing, pre-copy/final-delta migration, rollback, content-
addressed forks, changed-weight recomputation, and bounded protocol exploration.
External runtimes and GPU execution remain explicitly unexercised.

## High-severity finding dispositions

### ADV-HIGH-001: normal migration failure and restart recovery -- closed in source

The normal pre-copy orchestrator now has phase-aware cleanup. Before commit it
persists abort/rollback, stops dirty tracking, discards destination staging, checks
the authoritative lease and gateway/source watermark, and only resumes a proven
valid source. After commit it never revives the old source and classifies recovery
as `FAILED_AFTER_COMMIT` or `OPERATOR_REQUIRED`.

The durable coordinator now enumerates nonterminal transactions after SQLite
reopen and accepts typed recovery evidence. Forward progress and ownership commit
remain illegal after the transaction deadline, while recovery-only terminalization
is allowed with monotonic journal time. Exact replay is idempotent and altered
recovery evidence conflicts. Focused tests cover pre/post-commit outcomes,
deadline expiry, SQLite reopen, and replay. Runtime-process reconstruction beyond
the reference adapter remains a scoped limitation, not an implicit rollback.

### ADV-HIGH-002: live migration discarded direct conversion output -- closed

Initial import now consumes the direct destination-layout capture. Iterative and
final source deltas are merged into a verified source mirror, converted to the
destination physical layout, and applied as destination-layout deltas. The final
quiesced delta is required to contain real cutover-boundary state. Tests inspect
the destination import and prove packed head-major TP=2 state, including nonempty
final-delta movement, rather than relying on adapter-internal source decoding.

### ADV-HIGH-003: verifier trusted claimed booleans -- closed

CLI verification now requires an exact sibling demo manifest, validates the
capsule and per-event timeline seals, requires legal failed/successful phase
histories, derives owner/layout/TP/page/token/fork/recomputation invariants from
primitive evidence, reproduces the gateway token timeline, and rejects a mismatch
between derived and claimed invariants. It also checks capsule-derived direct-
conversion counts, bytes, hashes, canonical agreement, and temporary-memory
bounds. Missing/bad manifests and invariant, token, conversion, compatibility,
fork, and recomputation tampering are negative-tested. Raw campaign artifacts are
verified through their hash-bound evaluation index rather than this sibling-
manifest command.

### ADV-HIGH-004: capsule validation ignored external chunks -- closed

`continuum capsule validate` now opens the explicit or adjacent filesystem CAS,
checks tenant and canonical CAS URI, refuses an unimplemented transformed-chunk
fallback, reads every bounded chunk through authenticated store metadata, and
verifies chunk and segment digests. Missing stores and corrupted chunks are
negative-tested.

### ADV-HIGH-005: hard-coded conversion throughput was labelled observed -- closed

Planner inputs now derive throughput samples from selected-backend
`ConversionMeasurement` byte counts and elapsed nanoseconds, including sample
count and coefficient of variation, and bind the raw conversion artifact. The
transport curve remains correctly labelled configured synthetic evidence. H4 is
still scoped to exhaustive enumeration of the finite deterministic candidate set,
not observed production regret.

### ADV-HIGH-006: release evidence was not source-reproducible -- open release gate

The exact benchmark command now records `--reset`; the source-side benchmark,
GPU-marker, Docker archive-context, and clean-room scripts are repaired. This item
is closed only when the final source is committed, CPU/demo/model-check evidence
is regenerated with that commit, metric bullets are populated solely from those
new hash-validated artifacts, and `make continuum-clean-room-test` passes from the
committed archive. `docs/reports/CONTINUUM_FINAL_REPORT.md` must bind the final commit, commands,
artifact digests, clean-room result, negative results, and unexercised paths.

## Remaining medium-severity limitations and completion gaps

1. **The fault matrix is mostly a catalogue plus an abstract model.** One real CPU
   system fault (destination crash during validation) is exercised. The Rust model
   checker explores bounded single-fault interleavings. The many typed fault
   definitions are not each end-to-end system campaigns.
2. **External runtime modules are partial bindings.** vLLM, SGLang, and Genesis
   reject portable active-state export where no stable API exists; PyTorch is a
   bounded tensor/RNG and optional CUDA-conversion helper. None is an exercised
   cross-runtime live-migration adapter in this environment.
3. **GPU evidence is intentionally absent.** The opt-in marker and version-gated
   PyTorch CUDA equivalence test now exist, but PyTorch/CUDA was unavailable. There
   is no custom Triton/CUDA kernel or GPU throughput/interruption result.
4. **Distributed coordination remains an interface-level design.** Embedded
   SQLite CAS and fencing are implemented. No etcd or Kubernetes Lease backend was
   exercised or shipped.
5. **Cross-language conformance is capsule-centric.** Rust contains the additional
   wire records, but the shared end-to-end golden fixture is the complete capsule;
   standalone Rust/Python goldens for every secondary IR are not present.
6. **Report regeneration authenticates referenced raw files, not every aggregate.**
   Raw paths and hashes are checked, but an editable summary's confidence intervals
   and hypothesis text are not all independently recomputed by the renderer.
7. **Visualization breadth is below the requested artifact-backed view set.** Raw
   transaction/timeline/layout/conversion/fork data exist; the static HTML remains
   primarily a report table rather than all requested graph and mapping views.
8. **No upstream-ready external runtime/transport patch is present.** Local
   version-gated bindings and the portable TCP implementation exist, but no
   upstream runtime patch has its own acceptance bundle.

Previously reported gaps now closed include: actual changed-weight recomputation
through clone, teacher force, full cutover transaction, activation and bounded
gateway-accepted output; an explicit measured float32-to-float16 quality-bounded
converter; optional TCP AES-256-GCM payload protection with downgrade/wrong-key
failure; a registered honest GPU test/status path; nonempty final-delta movement;
correct reproducibility filename; and real source commit provenance in state
capture rather than the literal `workspace` placeholder. The final source also
exposes and tests checkpoint, pause, transactional resume, and clone CLI workflows.

## Evidence and methodology assessment

- The retained model-check result records 1,659 states, 3,216 transitions, maximum
  depth 22, and all 15 declared invariants passing under explicit one-fault and
  one-token bounds. This is useful bounded evidence, not a universal proof.
- The CPU flagship exercises KV plus recurrent, sampler, guided-decoding, token and
  client-delivery state across token-major separate-K/V TP=4/page=3 and head-major
  packed-K/V TP=2/page=5 reference adapters. It performs a pre-commit failure and
  safe rollback, a successful epoch-1 to epoch-2 migration, stale-source rejection,
  contiguous gateway acceptance, COW fork, unsafe changed-weight rejection, and an
  executed recomputation transaction.
- Direct CPU conversion remains an honest negative benchmark in the retained
  pre-final campaign: it won 0/5 median comparisons. No GPU, RDMA, multi-node, or
  cloud speedup is claimed.
- The quality-bounded backend is a scoped numeric contract: lower-precision float
  conversion is checked against the trusted canonical result and a measured finite
  maximum-absolute-error budget. It is not a claim about universal model quality.
- TCP AES-256-GCM binds transfer, sequence, attempt, digest, and plaintext size as
  authenticated context and refuses encryption-mode downgrade. This protects
  payloads when configured; it is not GPUDirect, TLS PKI, or a production endpoint
  identity system.

## Hiring review answer

**Does Continuum look like a thin KV-transfer wrapper, or does it demonstrate a
genuinely new portable execution-state abstraction with credible compiler,
runtime, transaction, GPU, and fault-tolerance depth?**

It does **not** look like a thin KV-transfer wrapper. The separation of logical and
physical state, recurrent/sampler/guided/client components, dependency-aware model
compatibility, canonical and direct conversion, content-addressed COW state,
durable epoch/fencing protocol, explicit gateway token ledger, crash recovery, and
bounded model checker demonstrate a genuinely broader portable execution-state
abstraction. The CPU compiler/runtime/transaction/fault-tolerance work is a strong
principal-level research and systems artifact.

The correct qualifier is that GPU and external-runtime depth are not yet validated
implementation evidence. The optional CUDA test was unexercised, there is no
custom GPU kernel, and vLLM/SGLang/Genesis active-state migrations did not run.
Those are honest limitations; they prevent a universal-production claim but do not
reduce the implemented CPU substrate to a KV-transfer wrapper.

## Review commands

```text
uv run --locked sloforge continuum --help
uv run --locked sloforge continuum runtime --help
uv run --locked sloforge continuum state --help
uv run --locked sloforge continuum migration --help
uv run --locked sloforge continuum conversion --help

uv run --locked pytest -q \
  tests/python/test_continuum_evaluation.py \
  tests/python/test_continuum_reports.py \
  tests/python/test_continuum_flagship.py \
  tests/python/test_continuum_protocol_review.py \
  tests/python/test_continuum_abi_review.py
# 36 passed

uv run --locked pytest -q \
  tests/python/test_continuum_conversion.py \
  tests/python/test_continuum_tcp_transport.py \
  tests/python/test_continuum_security.py \
  tests/python/test_continuum_gpu.py -rs
# 45 passed, 1 skipped: PyTorch unavailable

uv run --locked pytest -q \
  tests/python/test_continuum*.py -rs
# 193 passed, 1 skipped: PyTorch unavailable

uv run --locked sloforge continuum migrate \
  --mode pre-copy --seed 8202 --session review-session \
  --output /tmp/sloforge-continuum-final-review.hXlX5d/migration
uv run --locked sloforge continuum migration verify \
  --artifact /tmp/sloforge-continuum-final-review.hXlX5d/migration/flagship.json \
  --output /tmp/sloforge-continuum-final-review.hXlX5d/migration/verification.json
# 37 contiguous tokens (0..36), epoch 2, 21 final-delta chunks / 1,499 wire bytes,
# and a completed changed-weight recomputation transaction with five resumed tokens

cargo fmt --all --check
CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo clippy \
  -p sloforge-continuum-ir -p sloforge-state-transaction \
  -p sloforge-state-modelcheck --all-targets --all-features --locked -- -D warnings
CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo test \
  -p sloforge-continuum-ir -p sloforge-state-transaction \
  -p sloforge-state-modelcheck --all-features --locked
# 6 ABI conformance, 5 model-check, and 9 transaction tests passed

uv run --locked sloforge continuum report \
  --evaluation artifacts/continuum/evaluation/evaluation-summary.json \
  --output /tmp/continuum-review/report-index.json
```

Final source severity: **zero open high-severity implementation findings and eight
medium-severity limitations/completion gaps.** Final release severity remains
**one blocking provenance gate** until the committed-source evidence regeneration,
archive clean-room run, and final-report audit pass.

## Post-review closure: authenticated evaluation aggregates

The editable-summary-aggregates medium finding is closed. Each seed now emits a
bounded, content-hashed `measurements.json`; evaluation loading and report
generation authenticate every referenced raw artifact, cross-check each per-seed
summary field against raw runtime/conversion/planner evidence, and independently
recompute all confidence intervals, hypothesis outcomes, and negative-result
counts. Tests reject aggregate values altered and re-sealed in the summary while
the authenticated seed evidence remains unchanged. This reduces the documented
open medium-severity limitations/completion gaps from eight to **seven**.
