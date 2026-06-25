# Exact quantized recurrent-state update

This is a local upstream-ready review bundle for a focused Python CPU implementation of a persistent symmetric-int8 state update. It has not been submitted for external review, and no pull request or issue has been opened.

## Contract

For every logical element:

```text
updated = clamp(round_ties_to_even(previous * 0.625 + activation * 31.0), -127, 127)
```

The supported view has 1–128 elements, previous values in `[-127, 127]`, finite real activations, offsets greater than or equal to zero, and strides 1, 2, or 3. Output may be returned separately or committed into the exact previous-state view. Aliased updates validate and calculate the complete result before commit, preventing partial mutation on rejected input.

The implementation is standalone and uses only the Python standard library. It contains no SLOForge-specific import or model-specific constant other than the declared operator coefficients and domain.

## Files

- `quantized_state_update.py`: candidate plus scalar reference implementation.
- `test_quantized_state_update.py`: boundary, strided, aliasing, failure-atomicity, invalid-domain, and seeded randomized differential tests.
- `benchmark.py`: equal-warmup, seeded randomized-order CPU benchmark retaining every raw sample.
- `benchmark_cpu_sample.json`: one preserved local run; its summary explicitly denies an acceptance claim.
- `LIMITATIONS.md`: exact scope and non-claims.
- `EVIDENCE.md`: executed commands and repository evidence references.

## Correctness reproduction

From this directory:

```bash
python -m unittest -v test_quantized_state_update.py
```

## Benchmark reproduction

The benchmark requires an explicit seed and refuses to overwrite its output:

```bash
python benchmark.py \
  --seed 73129 \
  --warmup 3 \
  --repetitions 11 \
  --iterations 2000 \
  --count 32 \
  --output /tmp/hybrid-quantized-state-update-benchmark.json
```

The report contains raw `perf_counter_ns` observations and a descriptive median point estimate. It deliberately sets acceptance to `not_evaluated_single_machine_point_estimate_only`; this bundle makes no speedup claim.

## Integration

The callable can be reviewed as a focused implementation or converted into a patch for a runtime with the same operator contract. Integration must preserve ties-to-even rounding, symmetric saturation, exact strided aliasing, failure atomicity, and the documented fallback. End-to-end serving impact must be remeasured in the target runtime before adoption.
