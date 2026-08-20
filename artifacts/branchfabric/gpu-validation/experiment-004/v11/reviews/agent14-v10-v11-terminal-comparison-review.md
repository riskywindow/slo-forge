# Agent 14 — terminal v10/v11 comparison review

Verdict: **PASS_TERMINAL_COMPARISON_WITH_INTEGRATED_NOT_REACHED**.

This is an independent read-only post-integrated review. I did not invoke GPU, Modal, or network work and did not modify source. The valid comparison boundary is frozen v10 versus real-A100 v11 attempt-D micro measurements for bytes, amplification, pass families, memory tiers, and temporary host memory. It is not valid to compare the PCIe micro endpoint timings directly with the frozen SXM4 integrated serving timings.

## Measured micro comparison

The logical state is 1,056,964,608 bytes and attempt D has 8/8 exact first resumed-token matches.

| Metric | Frozen v10 | Measured v11 micro | Difference |
|---|---:|---:|---:|
| Full physical bytes | 61,303,947,264 | 22,197,063,040 | 39,106,884,224 removed |
| Full amplification | 58× | 21.000762818× | 36.999237182× removed |
| External bytes | 3,170,893,824 | 2,114,200,960 | 1,056,692,864 removed |
| External amplification | 3× | 2.000257098× | 0.999742902× removed |
| Avoidable bytes | 32,765,902,848 | 12,684,381,568 | 20,081,521,280 removed |
| Avoidable amplification | 31× | 12.000762818× | 18.999237182× removed |
| Full-state pass families | 30 | 10 | 20 removed |
| Peak temporary host bytes | 2,965,372,928 | 143,552,512 | 2,821,820,416 saved |

The authoritative full-physical reduction is 63.79178824422134%. The immutable pass-diff and movement artifacts contain a superseded derived value of 63.79181449900818%; their measured bytes are unchanged, and the content-addressed terminal arithmetic-correction artifact supplies the exact calculation. The pass-diff covers all 30 v10 passes: 3 retained, 9 fused, 10 removed, and 8 replaced.

Measured bytes removed by tier are 23,252,954,112 GPU-intermediate bytes, 14,797,237,248 host-intermediate bytes, zero native-endpoint bytes, and 1,056,692,864 link bytes. The capture/publish, restore/native-write excluding destination validation, and destination-validation chains remove 12,683,547,648, 10,568,867,456, and 15,854,469,120 bytes respectively.

## Integrated comparison

Integrated attempt `exp004-v11-integrated-s41-b` is scientifically invalid. Its pre-reclaim backlog reached 65 against the bounded ceiling of 64, so the reclaim trigger and the v11 state pipeline were never reached.

Every integrated v11 state and timing cell is therefore **NOT_REACHED/null**: full/external/avoidable movement, pass count, tier bytes, HBM reclamation, first useful GPU1 capacity, two-GPU excess service, queue drain, serving SLO restoration, rollout restore, GPU0 restore interference, and integrated branch correctness. Null is not zero, an improvement, or a regression.

The frozen v10 values—5.222174431 s to HBM reclamation, 5.413681427 s to first useful GPU1 capacity, 11.373208336 s to SLO restoration, 7.727983415 s to first resumed token, and 7.848228746 s to continuation completion—remain reference values only. No v11 absolute or percentage timing improvement is calculable.

No micro timing was relabelled as integrated, no v10 time was used as a v11 denominator, and no pre-reclaim serving sample was treated as state-pipeline evidence. The evidence and immutable source hashes are recorded in the JSON companion.
