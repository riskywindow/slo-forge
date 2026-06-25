# Correctness evidence

This file is completed only from locally executed commands. Source digests and the observed command outcome are recorded after validation; they are not an external certification.

Repository evidence with the same operator contract:

- [`reference.py`](../../../models/reference_tasks/hybrid_decoder/reference.py) defines the reference state update.
- [`reference_package.json`](../../../models/reference_tasks/hybrid_decoder/reference_package.json) declares the state and custom-operator contract.
- [`cases.py`](../../../python/sloforge/genesis/kernel_lab/cases.py) generates edge, stride, alias, non-finite, and seeded randomized cases.
- [`test_genesis_kernel_lab.py`](../../../tests/python/test_genesis_kernel_lab.py) exercises restricted generation, sandboxed correctness, raw benchmarking, and fail-closed GPU status.

## Executed local validation

On 2026-08-02, the standalone command

```bash
python -m unittest -v test_quantized_state_update.py
```

completed successfully. It exercised the full symmetric previous-state domain in bounded partitions, rounding/saturation boundaries, finite binary64 overflow, non-contiguous views, exact alias commit, failure atomicity, invalid domains, and a seed-73129 randomized differential corpus.

Ruff formatting/lint and `py_compile` also completed successfully for all three Python files.

## Content identity

```text
6c9b1ab3ab5a928eaaa998fff175149897ce1e6ead1506be49de38284bb6458a  quantized_state_update.py
a7203188458342fc57650d87d1f2634a2bc9d719328ffa15cf462a2d5c9ccaf0  test_quantized_state_update.py
e1afa28f0eaad117aa028917e1232e3a0c6726fc185cf90f31920e4cecf1be39  benchmark.py
babfb65b19befa5174530cf08ecc290ad871018919344de6d40bbe7e9f0ec6a4  benchmark_cpu_sample.json
```

## Measured negative result

`benchmark_cpu_sample.json` contains 22 raw paired observations from the documented command on CPython 3.12.7/macOS arm64. The candidate median was slower than the scalar reference median. The artifact records the exact point estimate and sets `acceptance` to `not_evaluated_single_machine_point_estimate_only`. No speedup or full-stack benefit is claimed.
