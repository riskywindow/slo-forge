# Continuum Fault-Tolerance Evaluation

Each seed injects a destination crash at `DESTINATION_VALIDATING`, before ownership commit. The coordinator is closed and reopened before the second migration.

| Seed | Fault label | Final phase | Source epoch | Duplicate accepted | Token gaps | Second migration |
|---:|---|---|---:|---:|---:|---|
| 101 | continuum.fault.destination_crash_during_validation | ROLLED_BACK | 1 | 0 | 0 | COMPLETED |
| 202 | continuum.fault.destination_crash_during_validation | ROLLED_BACK | 1 | 0 | 0 | COMPLETED |
| 303 | continuum.fault.destination_crash_during_validation | ROLLED_BACK | 1 | 0 | 0 | COMPLETED |
| 404 | continuum.fault.destination_crash_during_validation | ROLLED_BACK | 1 | 0 | 0 | COMPLETED |
| 505 | continuum.fault.destination_crash_during_validation | ROLLED_BACK | 1 | 0 | 0 | COMPLETED |

External exactly-once delivery is not claimed. These results establish exactly-once acceptance at the SLOForge gateway for the bounded resumable protocol fixture.
