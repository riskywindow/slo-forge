# Agent 07b — Replacement invocation budget/remediation audit

Status: **PASS**

This was an offline, read-only audit of execution code and evidence. No GPU, Modal, network, or reservation action was performed. Only `/root` remains authorized to reserve and invoke GPU work.

## Decision

Exactly one replacement v11 micro-validation invocation is authorized at **1 × A100-80GB for at most 525 seconds**, provided `/root` atomically creates a fresh reservation immediately before launch and all listed fail-closed checks still pass.

The current ledger records 13,081.823626085978 consumed GPU-seconds and no active reservations under the configured 21,600-second ceiling. The replacement requires 525 GPU-seconds; its 15% planning margin is 78.75 GPU-seconds, for a conservative envelope of 603.75 GPU-seconds.

| Quantity | GPU-seconds | A100-hours |
|---|---:|---:|
| Current consumed | 13,081.823626085978 | 3.633839896134994 |
| Current remaining | 8,518.176373914022 | 2.366160103865006 |
| Replacement exact bound | 525.0 | 0.14583333333333334 |
| Replacement plus 15% margin | 603.75 | 0.16770833333333332 |
| Projected cumulative, exact reservation | 13,606.823626085978 | 3.7796732294683273 |
| Projected cumulative, including margin | 13,685.573626085978 | 3.801548229468327 |
| Remaining after exact reservation | 7,993.176373914022 | 2.220326770531673 |
| Remaining after planning margin | 7,914.426373914022 | 2.198451770531673 |

At the ledger rate of $2.4984 per A100-hour, the exact replacement bound is $0.36435 and the planning envelope is $0.4190025. Both fit the explicit $40 paid-launch authorization.

## First-attempt accounting

The failed attempt stopped at `REMOTE_AUTHORIZATION_BEFORE_MODEL_LOAD`: no model was loaded, no remote state work began, and no measurement was produced. Actual GPU seconds were unavailable. Charging the full 525-second preflight bound is therefore conservative and does not invent an actual runtime.

The failure artifact SHA-256 matches the ledger commitment. The old reservation is no longer active; it appears exactly once as a conservative failure charge. The postflight cleanup record shows the app stopped with zero tasks and empty active-app, container, endpoint, reservation, child-process, and profiler lists. Only the two authorized persistent volumes remain.

## Remediation review

The recorded cause was that the one-time coordinator capability enabled local graph hydration but was absent from the remote container environment. The revised runtime attaches that same capability as an ephemeral runtime secret to the single bounded function. Remote validation still binds:

- the immutable config SHA-256;
- the coordinator capability SHA-256;
- exactly one 525-second A100 reservation;
- the paid budget authorization.

This is directly aligned with the recorded failure and does not bake the capability into the image. The static remediation and offline tests pass. The replacement invocation remains the first real remote validation of the fix; this audit does not claim otherwise.

## Verification

- 17 targeted budget/runtime tests passed.
- Strict ledger schema and derived-consumption validation passed.
- Failure-evidence commitment, authorization ceiling/scope, and cleanup assertions passed.
- Runtime compilation passed.
- Ruff lint and format checks passed.

Key commitments:

- Current ledger: `fa7800a901f443696d0405d32843b0114f4f2adb780e2fbf0704a09366b6d172`
- Authorization: `86158d49e10c39afd3a07f0dc665202cadaa5cccd053748d98634c8d7f387300`
- Failure evidence: `3e6654d4e27e2309e60b4dfa8cbc8222fb78ace11dadbb41a91c1c48923e4554`
- Postflight cleanup: `f0e66566c3cd31ef5d5d96c87e7283e9bb86680243ebb7ea665c4573b2722f38`
- Remediated runtime: `428fa30235305246ec6168e2fb8220ccde335947e2d3aa08534b0151c82f57cb`

## Launch-time fail-closed conditions

The PASS becomes `BUDGET_BLOCKER` if the ledger changes so fewer than 603.75 GPU-seconds remain; any active reservation or active Modal resource exists; the replacement exceeds one A100 or 525 seconds; the budget/token environment gate is absent; or any reservation, config, authorization, or source commitment differs. `/root` must re-read the ledger and cleanup evidence immediately before creating the fresh single reservation.
