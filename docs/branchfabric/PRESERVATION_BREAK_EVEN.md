# Preservation Break-Even

Status: **not fitted**.

The only integrated v11 preservation transaction failed before source release.
The protocol required that transaction to pass before executing kill/recompute,
so there is no measured recompute arm. A numerical break-even surface would
therefore be fabricated and is intentionally absent.

The retained symbolic model is:

```text
recompute work tokens = B(P + A)
preserved state bytes = q(P + BA)

T_recompute = B(P + A) / measured_prefill_throughput + measured_fixed_recompute_overhead
T_preserve  = measured_state_movement_term + measured_fixed_preservation_overhead
```

Here `P` is shared-prefix tokens, `B` is branch count, `A` is branch-private age,
and `q` is unique logical state bytes per token. None of the required integrated
preservation coefficients or kill/recompute coefficients is available from the
failed transaction. The analytical 133,120-token expectation is not promoted as
a measurement, and the successful micro-validation export/restore timings are
not substituted for integrated coefficients.
