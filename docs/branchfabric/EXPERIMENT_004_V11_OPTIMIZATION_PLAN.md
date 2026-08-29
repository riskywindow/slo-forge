# Experiment 004 v11 Software Optimization Plan

Status: **PLAN ONLY — NOT IMPLEMENTED OR EXECUTED**.

Every estimate below is computed from measured v10 pass labels, byte touches, and deduplicated operation intervals. Bytes and time are removable upper bounds, not promised speedups; zero means v10 exposed no separately attributable pass for that family.

Frozen logical-state denominator: `1056964608` bytes.

| Rank | Candidate | Measured passes targeted | Physical bytes potentially removed | Critical-path seconds potentially removed | Semantic risk | Complexity |
|---:|---|---:|---:|---:|---|---|
| 1 | Fused capture transforms | 3 | 9512681472 | 1.326930 | medium | medium |
| 2 | Direct native-to-transport conversion | 4 | 12683575296 | 1.333571 | high | high |
| 3 | Eliminate complete canonical intermediates | 5 | 15854469120 | 8.050707 | medium | high |
| 4 | Pinned double buffering | 1 | 2113929216 | 0.168294 | low | medium |
| 5 | Asynchronous D2H/H2D | 1 | 2113929216 | 0.168294 | medium | medium |
| 6 | Overlap transformation and copy | 4 | 11626610688 | 1.495224 | medium | high |
| 7 | Combine integrity scans | 5 | 13740539904 | 4.818632 | high | medium |
| 8 | Eliminate unnecessary full-page zero-fill | 0 | 0 | 0.000000 | medium | low |
| 9 | Fused restore transform/native write | 0 | 0 | 0.000000 | high | high |
| 10 | Device-side integrity proof | 5 | 13740539904 | 4.818632 | high | high |
| 11 | Avoid full destination D2H recapture | 1 | 2113929216 | 1.313802 | medium | medium |
| 12 | Validation sampling if semantics permit | 1 | 2113929216 | 1.313802 | very high | low |

Correctness gates remain fresh destination allocation, full integrity until a proven equivalent replaces it, eight branches, at least eight continuation tokens, and 8/8 independent first-token matches. No optimization is authorized by this plan.
