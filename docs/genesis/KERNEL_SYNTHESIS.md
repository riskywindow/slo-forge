# Focused kernel synthesis

Genesis Kernel Lab targets operations only after immutable bottleneck evidence identifies an operator and genome region with enough expected upside. The CPU-exercised target is HybridDecoder's `quantized-state-update`: a per-token persistent-state operation computing symmetric int8, ties-to-even rounding of `previous * 0.625 + activation * 31.0`, saturated to `[-127, 127]`.

## Contract and generated surface

The typed operator schema declares symbolic batch bounds, int8 and float64 dtypes, allowed strides, non-contiguous views, exact in-place previous-state aliasing, rejection of NaN and infinity, exact numerical behavior, determinism, the CPU architecture scope, and the reference fallback. Batching represents independent HybridDecoder requests; it does not broaden single-request semantics.

The local generator enumerates bounded combinations of unroll factor, clamp form, and constant caching from an explicit seed. Each source file is content-hashed. Before execution, a trusted AST allowlist rejects unexpected imports, dynamic loading, file access, process access, arbitrary attributes, or unexpected functions. Generated source then runs only through the existing fail-closed sandbox with no network, explicit read-only inputs, a separate writable artifact directory, sanitized environment, resource limits, output limits, process cleanup, and a wall timeout.

## Independent verification

The correctness harness does not import generated code into the controller process. It uses sandbox execution and independently compares returned results with the reference operation. Cases cover saturation boundaries, ties-to-even rounding, randomized shapes, contiguous and non-contiguous strides, exact in-place aliasing, batch edges, and declared rejection of NaN and positive and negative infinity. Seeds, case counts, source digests, sandbox backend, and mismatches remain in evidence.

## Performance evidence

Benchmarking uses equal warmups, randomized reference/candidate order, repeated raw nanosecond samples, deterministic bootstrap confidence intervals, effect size, practical significance, and an explicit noise floor. It records scalar, non-contiguous batch, and end-to-end recurrent token-loop regimes. A candidate is accepted only when correctness passes and both the intended microbenchmark and token-loop intervals exceed the practical/noise threshold. A favorable point estimate is insufficient. Regression, unavailable, and inconclusive results are retained with `no speedup claim`.

The CPU fingerprint and Python software manifest scope every result. These are real local measurements, not GPU evidence. The optional Triton adapter only reports installed-version and opt-in status; it remains fail-closed and unexercised unless a separate real GPU harness and `SLOFORGE_GENESIS_ALLOW_GPU=1` are available. No CPU measurement is presented as a Triton, CUDA, or GPU-kernel result.

## Validation status

| Evidence source | Status | Permitted claim |
| --- | --- | --- |
| Deterministic operator fixtures and simulator-derived serving inputs | Exercised in CPU tests | Correctness/search evidence only within declared fixtures and seeds; not measured performance |
| Local macOS CPU execution and timing | Exercised by the CPU kernel-lab tests/demo | Host-CPU correctness and timing for the recorded software/hardware fingerprint |
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

`run_kernel_lab_demo` writes candidate source, correctness records, benchmark summaries, complete raw JSONL samples, decisions, and the aggregate report beneath its caller-supplied output directory. The repository does not check in generated measurement outputs. Every accepted or negative decision can therefore be reconstructed from its raw artifacts.
