# Continuum Runtime Adapter Status

| Runtime | Version | Status | Discovery exercised | Migration exercised | Limitation |
|---|---|---|---|---|---|
| continuum-reference-token-major | 1.0.0 | implemented_and_exercised | true | true | deterministic CPU runtime with simulated devices, not a hardware runtime |
| continuum-reference-head-major | 1.0.0 | implemented_and_exercised | true | true | deterministic CPU runtime with simulated devices, not a hardware runtime |
| pytorch | unavailable | partially_implemented_package_not_installed | false | false | version-gated public API probe only; no complete active-state migration was exercised in this CPU campaign |
| genesis | 1.0.0 | partially_implemented_ready | true | false | version-gated public API probe only; no complete active-state migration was exercised in this CPU campaign |
| vllm | unavailable | partially_implemented_package_not_installed | false | false | version-gated public API probe only; no complete active-state migration was exercised in this CPU campaign |
| sglang | unavailable | partially_implemented_package_not_installed | false | false | version-gated public API probe only; no complete active-state migration was exercised in this CPU campaign |

Only the two deterministic reference adapters performed active-state migration in this campaign. Public API discovery does not constitute migration validation.
