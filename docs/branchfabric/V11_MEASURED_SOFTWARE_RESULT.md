# v11 Measured Software Result

The optimized v11 state pipeline passed its real-A100 micro gate on the exact 16K/fanout-8 workload.

The logical state was 1,056,964,608 bytes across 1,152 real vLLM blocks. All allocator epochs, ownership release proofs, fresh destination allocation commitments, hashes, raw-bit comparisons, admission gates, and eight branch continuations passed. First resumed tokens matched independent recomputation 8/8.

| Metric | Frozen v10 | Measured v11 micro |
|---|---:|---:|
| Full physical bytes | 61,303,947,264 | 22,197,063,040 |
| Full amplification | 58× | 21.000762818× |
| External bytes | 3,170,893,824 | 2,114,200,960 |
| External amplification | 3× | 2.000257098× |
| Avoidable bytes | 32,765,902,848 | 12,684,381,568 |
| Avoidable amplification | 31× | 12.000762818× |
| Full-state passes | 30 | 10 |
| Peak temporary host bytes | 2,965,372,928 | 143,552,512 |

The measured physical-work reduction is 39,106,884,224 bytes, or 63.7918%. Host temporary memory fell by 2,821,820,416 bytes, or 20.6571×. The pinned staging component was exactly 58,720,256 bytes; it is not the total temporary-host peak.

The authoritative terminal arithmetic views are `artifacts/branchfabric/gpu-validation/experiment-004/v11/state-passes/v10-v11-pass-diff-terminal-corrected.json` and `artifacts/branchfabric/gpu-validation/experiment-004/v11/movement/v10-v11-measured-comparison-terminal-corrected.json`. They preserve the immutable prelaunch evidence graph while correcting the derived physical-work reduction to exactly 63.79178824422134%.

Measured micro export was 2.613090870 s wall / 0.133267776 s CUDA. Restore was 4.438193614 s wall / 0.697858271 s CUDA. These timings came from an A100 PCIe device and are not directly comparable to frozen v10's two-A100 SXM4 integrated timing.

The single integrated attempt authenticated two A100-SXM4-80GB devices and passed both capacity sanity guards, but failed at the bounded pre-reclaim backlog ceiling before executing v11 export or import. Therefore no integrated v11 reclamation, serving recovery, restore, or interference metric exists.
