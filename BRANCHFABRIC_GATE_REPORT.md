# BranchFabric hardware gate report

Phase: **preliminary**.
Outcome: **FAIL_NO_BUILD**.
Hardware implementation allowed: **false**.

No threshold was loosened. CPU-reference, synthetic, simulated-hardware, and artifact-replay evidence cannot satisfy the target-hardware requirement.

## Candidate results

| Candidate | Real evidence | End-to-end relevance | Headroom | Platform | Workload value | Disposition |
|---|---|---|---|---|---|---|
| reshard_data_path | FAIL | FAIL | FAIL | FAIL | FAIL | NOT_JUSTIFIED |
| one_to_many_multicast | FAIL | FAIL | FAIL | FAIL | FAIL | NOT_JUSTIFIED |
| branch_translation_metadata | FAIL | FAIL | FAIL | FAIL | FAIL | NOT_JUSTIFIED |
| shared_root_cow | FAIL | FAIL | FAIL | FAIL | FAIL | NOT_JUSTIFIED |

## Required action

Terminate Hardware Path.
