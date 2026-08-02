# Operator and numerical verification

The Genesis operator verifier is independent of proposal generation. It accepts a reference callable, candidate callable, typed `OperatorContract`, and explicit seed. It never accepts a candidate based on its own claimed equivalence.

## Declared domain

An operator contract identifies the operator and bounds every shape dimension. It declares dtype set, non-contiguous-input support, aliasing permission, exact versus approximate comparison, absolute and relative tolerances, NaN/infinity preservation, deterministic behavior, and a maximum case count.

The verifier deterministically covers minimum, maximum, and seeded intermediate dimensions for every declared dtype and stride class; an insufficient case budget is rejected instead of leaving part of the declared domain unexercised. It constructs contiguous and non-contiguous inputs where meaningful and injects NaN, infinity, signed zero, integer/bool values, and randomized finite values. It checks output shape/dtype, exceptional-value masks, exact values or numerical tolerances, forbidden output aliasing, unexpected input mutation, and repeat determinism using the same stride class. Seed, rank, generated-element, and case-count domains are bounded.

A failure produces a typed `OperatorCounterexample` with case index, seed, shape, dtype, stride variant, violated property, expected/observed content digests, and a bounded prefix of the triggering input values. The value prefix makes the failure inspectable but is not a general delta-debugging minimizer.

## Quality verification

Quality evaluation is separate from synthesis scoring. Given aligned reference/candidate logits and token arrays, it measures exact-token match, top-1 agreement, top-k set overlap, KL divergence, Jensen-Shannon divergence, and maximum absolute logit error against explicit thresholds. The evaluator requires a non-empty aligned dataset, a non-empty vocabulary, finite logits representable as float64, and a bounded logit-cell count. Metrics are evaluated in float64 even when candidate outputs use a lower-precision floating dtype.

The API does not encode dataset secrecy. Callers are responsible for keeping search/calibration data separate from final evaluation data and for binding dataset identity into capsule evidence.

## Verification levels

Genesis records claims independently at Level 0 build, Level 1 differential, Level 2 property, Level 3 bounded exhaustive, Level 4 solver-backed, and Level 5 hardware/operational. The NumPy operator harness reports Level 2 property evidence within its generated cases. It does not upgrade itself to solver-backed or hardware evidence.

## Exercised and unexercised scope

Tests exercise deterministic passing cases, a real rare-shape semantic failure, quality/distribution regression, non-contiguous inputs, exactness, exceptional values and alias/mutation checks. The high-performing claim in a separate benchmark cannot override an operator failure.

The current verifier does not run AddressSanitizer, CUDA racecheck, SMT index proofs, high-precision arbitrary-precision references, denormal-mode probes, or exhaustive dtype bit-pattern enumeration. It supports NumPy CPU arrays for the implemented harness; generated GPU kernels require the kernel lab and hardware-specific tools.
