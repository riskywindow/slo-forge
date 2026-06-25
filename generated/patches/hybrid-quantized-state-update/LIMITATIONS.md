# Limitations

- This is a Python CPU implementation, not Triton, CUDA, C++, SIMD, or a GPU kernel.
- The maximum logical count is 128 and allowed strides are 1, 2, and 3.
- Aliased output supports mutable Python sequences; NumPy, Torch, device tensors, and arbitrary buffer protocols need dedicated adapters and tests.
- Python `round` defines ties-to-even behavior. A port must verify its compiler and hardware rounding mode independently.
- The update is atomic only with respect to validation and commits made through this function. It does not provide thread synchronization or transactional coordination with other state owners.
- NaN and infinity are rejected. Finite scaled overflow saturates according to sign.
- The local benchmark does not characterize tail latency, energy, cost, other machines, other Python versions, or full serving impact.
- No external maintainer has reviewed or accepted this bundle.

