# Continuum Compatibility Evaluation

Matching shapes were held constant while attention state-producing weights changed. Direct reuse had to be rejected; token-history recomputation was allowed only with explicit dependency evidence.

| Seed | Shapes match | Direct reuse | Rejection reasons | Recompute class | Components |
|---:|---|---|---|---|---|
| 101 | true | incompatible | STATE_PRODUCING_DEPENDENCY_CHANGED | recomputation_assisted | attention.kv |
| 202 | true | incompatible | STATE_PRODUCING_DEPENDENCY_CHANGED | recomputation_assisted | attention.kv |
| 303 | true | incompatible | STATE_PRODUCING_DEPENDENCY_CHANGED | recomputation_assisted | attention.kv |
| 404 | true | incompatible | STATE_PRODUCING_DEPENDENCY_CHANGED | recomputation_assisted | attention.kv |
| 505 | true | incompatible | STATE_PRODUCING_DEPENDENCY_CHANGED | recomputation_assisted | attention.kv |

Unsafe direct-reuse acceptances: **0**.

The recomputation result is a compatibility plan with verification obligations; this campaign does not claim an executed changed-weight migration.
