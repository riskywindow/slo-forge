# Focused kernel synthesis

Genesis Kernel Lab targets operations only after immutable evidence identifies an operator and genome region with enough expected upside. The CPU-exercised target is HybridDecoder's `quantized-state-update`: a per-token persistent-state operation computing symmetric int8, ties-to-even rounding of `previous * 0.625 + activation * 31.0`, saturated to `[-127, 127]`. The flagship now retains the seeded six-request reference workload trace and profiles this operation inside the real reference prefill/decode path over seven trials. Its fraction is relative to the profiled `_advance` computation, is affected by profiler overhead, and is target-selection evidence—not SLOForge Autopsy output or causal attribution.

## Contract and generated surface

The typed operator schema declares symbolic batch bounds, int8 and float64 dtypes, allowed strides, non-contiguous views, exact in-place previous-state aliasing, rejection of NaN and infinity, exact numerical behavior, determinism, the CPU architecture scope, and the reference fallback. Batching represents independent HybridDecoder requests; it does not broaden single-request semantics.

The local generator enumerates bounded combinations of unroll factor, clamp form, and constant caching from an explicit seed. Each source file is content-hashed. Before execution, a trusted AST allowlist rejects unexpected imports, dynamic loading, file access, process access, arbitrary attributes, or unexpected functions. Generated source then runs only through the existing fail-closed sandbox with no network, explicit read-only inputs, a separate writable artifact directory, sanitized environment, resource limits, output limits, process cleanup, and a wall timeout.

## Independent verification

The correctness harness does not import generated code into the controller process. It uses sandbox execution and independently compares returned results with the reference operation. Cases cover saturation boundaries, ties-to-even rounding, randomized shapes, contiguous and non-contiguous strides, exact in-place aliasing, batch edges, and declared rejection of NaN and positive and negative infinity. Seeds, case counts, source digests, sandbox backend, and mismatches remain in evidence.

## Performance evidence

Benchmarking uses equal warmups, randomized reference/candidate order, repeated raw nanosecond samples, deterministic bootstrap confidence intervals, effect size, practical significance, and an explicit noise floor. It first records scalar, non-contiguous batch, and repeated operator-loop regimes. Those isolated regimes cannot promote a candidate.

The full-stack gate then materializes a content-addressed copy of HybridDecoder with the generated vector kernel and a generated scalar adapter, statically re-inspects that package, and generates a separate conservative bounded streaming runtime for the reference and candidate packages. Both runtimes execute the same six-request interleaved trace in the fail-closed sandbox for at least seven randomized paired trials. Model-generation, trace, trial-order, bootstrap, and sandbox seeds are disjoint and recorded. The timed boundary begins after runtime start, includes bounded admission, batching, prefill, decode, streaming terminal events, and a checked zero-owned-state boundary, and excludes package import, process startup, and runtime shutdown. Statistics use a seeded paired bootstrap over trial-aligned percent improvements; deterministic validation reconstructs the randomized order.

Token streams are checked on every measured trial. The independent validation run separately replays both generated runtimes and reconstructs each request's complete typed persistent state with the runtime's exact phase/position seed schedule. It reopens package, inspection, runtime bundle, candidate source, trace, runner, raw sample, and replay hashes and reconstructs all statistics. A candidate may claim only a local CPU generated-runtime speedup when exact token/state semantics and every correctness, isolated-regime, and end-to-end confidence gate pass. A failed or inconclusive end-to-end interval is retained as a negative result and produces no speedup claim.

The CPU fingerprint and Python software manifest scope every result. These are real local host measurements over a retained reference-package workload, not Autopsy or GPU evidence. The acceptance gate uses an explicitly declared noise threshold; it does not claim that this short run measures host thermal or clock noise. The optional fused-logits Triton harness requires both its call-site flag and `SLOFORGE_GENESIS_ALLOW_GPU=1`, a real CUDA device, Triton, randomized correctness, bounded contiguous float16/bfloat16/float32 inputs, the selected device's CUDA stream, balanced randomized interleaving, paired raw CUDA-event trials, a paired bootstrap interval, workload/hardware/software provenance, and independent recomputation. Externally supplied correctness evidence is rejected. A scoped isolated-kernel speedup is recorded only when the lower paired interval clears the practical threshold; it never becomes an end-to-end serving claim or automatic runtime enablement.

## Validation status

| Evidence source | Status | Permitted claim |
| --- | --- | --- |
| Retained reference workload trace and deterministic operator fixtures | Exercised in CPU tests/demo | Trace-profile target selection and correctness within declared fixtures and seeds; not causal attribution |
| Local macOS CPU generated-runtime execution and timing | Exercised by the CPU kernel-lab tests/demo | Host-CPU exact token/state evidence and end-to-end trace timing for the recorded software/hardware fingerprint |
| Triton capability adapter | Implemented, fail-closed, unexercised for compilation/execution | Availability and opt-in status only |
| CUDA/GPU kernel execution | Not exercised | No GPU correctness, latency, throughput, or speedup claim |
| Multi-GPU end-to-end serving impact | Not exercised | No communication or deployment performance claim |

Reproduce the CPU checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q tests/python/test_genesis_kernel_lab.py
```

Raw benchmark output from a demo is evidence only when its artifact directory, software manifest,
hardware fingerprint, seed, and exact command are retained. Checked-in source and passing unit tests
do not by themselves establish a speedup.

## Artifacts

`run_kernel_lab_demo` writes candidate source, correctness records, isolated benchmark summaries, both generated runtime bundles, source and patched package inspections, the interleaved trace, complete raw JSONL runtime samples, independent sandbox replay, decisions, and the aggregate report beneath its caller-supplied output directory. The repository does not check in generated measurement outputs. Every negative or inconclusive decision can therefore be reconstructed from its raw artifacts. The end-to-end experiment is real local CPU serving execution, not GPU or production-hardware evidence.
