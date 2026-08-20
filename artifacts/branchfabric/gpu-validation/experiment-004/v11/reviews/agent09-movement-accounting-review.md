# Agent 09 movement-accounting review

**Result: PASS. Integrated admission: PASS. Classification: MEANINGFUL.**

I independently reduced all 10,454 raw `StatePassRecord` entries from the real A100 attempt-D worker result. I did not use compatibility aliases, profiler observations, or transfer counters as additional byte sources.

## Measured movement

The exact logical state is 1,056,964,608 bytes. The canonical records sum to:

- Full physical work: **22,197,063,040 bytes = 21.000762818351625×**
- External movement: **2,114,200,960 bytes = 2.000257098485553×**
- Avoidable physical work: **12,684,381,568 bytes = 12.000762818351625×**
- Required physical work: **9,512,681,472 bytes = 9×**
- D2H: **1,056,964,608 bytes = 1×**
- H2D: **1,056,964,608 bytes = 1×**
- Link control: **271,744 bytes**

The external value is slightly above 2× because the canonical ledger includes 267,264 bytes of index-upload control traffic and 4,480 one-byte raw-validation results. These are measured transfers, not inferred state copies.

Export contributes 9.000026157924108× across four full-state pass families. Restore contributes 12.000736660427517× across six. The combined conceptual full-state pass count is **10**, versus frozen v10's reconstructed aggregate count of **30**, a reduction of 20 passes or 66.67%. The 10,454 execution records are the real chunk/layer/run instances and must not be confused with full-state family coverage.

## Tier conservation

Canonical tier traffic is:

- GPU native endpoints: 3,170,893,824 bytes (3×)
- GPU temporary: 7,399,019,520 bytes (7.000252859933036×)
- Host pinned: 9,512,681,472 bytes (9×)
- Host pageable metadata: 267,264 bytes
- D2H link: 1,056,964,608 bytes (1×)
- H2D link: 1,056,964,608 bytes (1×)
- Control link: 271,744 bytes

These tiers sum exactly to 22,197,063,040 bytes. Required + avoidable + diagnostic also closes exactly; diagnostic bytes are zero on the successful path.

## Ledger integrity

All 10,454 pass IDs are unique. No record is missing a required field. Every record conserves read + write + external bytes, tier bytes, and required/avoidable/diagnostic bytes. CUDA records all carry stream, start-event, end-event, and elapsed-time evidence; CPU records claim none. All dependency IDs exist, the dependency graph is acyclic, timing groups have exactly one owner, and no full-state family is incomplete, gapped, or overlapping.

The serialized `bytes_read`, `bytes_written`, and `transfer_bytes` aliases equal their canonical fields but were not added again. The worker's aggregate agrees exactly with the independent reduction.

## Frozen-v10 comparison and gate

Against frozen v10:

- Full physical work falls from 61,303,947,264 bytes (58×) to 22,197,063,040 bytes (21.000762818351625×): **39,106,884,224 bytes removed, 63.7918% lower**.
- External movement falls from 3,170,893,824 bytes (3×) to 2,114,200,960 bytes (2.000257098485553×): **1,056,692,864 bytes removed, 33.3248% lower**.
- Avoidable work falls from 32,765,902,848 bytes (31×) to 12,684,381,568 bytes (12.000762818351625×): **20,081,521,280 bytes removed, 61.2879% lower**.

All micro correctness fields pass, every branch generated at least eight continuation tokens, and first resumed tokens match independent recomputation 8/8. Because measured amplification is above 12× but at or below 25×, the required classification is **MEANINGFUL** and the strict decision is **GO_TO_ONE_INTEGRATED_V11_TRANSACTION**.

The scientific run settled at 262.89 A100-seconds ($0.18244566), and cleanup shows zero active apps, tasks, containers, endpoints, reservations, owned child processes, or profilers. This review launches no GPU or Modal work. It does not claim integrated serving performance or hardware interest.
