# Resource verification

Genesis performs conservative admission-time resource arithmetic. A `ResourceContract` declares device and host capacity, safety margin, fragmentation allowance, and process/thread/file-descriptor limits. A `ResourceDemand` separately declares model, persistent state, queue, communication, workspace, challenger, host, and state-conversion overlap demands.

The analyzer sums all simultaneously live device components, includes champion/challenger coexistence and conversion overlap, then inflates device demand by the fragmentation allowance. Usable capacities are reduced by the safety margin. It independently checks device memory, host memory, processes, threads and file descriptors. Negative demand and invalid capacity/fraction contracts fail closed.

Evidence records passed/failed status, conservative and usable device/host bytes, violations, and these explicit assumptions:

- champion and challenger coexist;
- state-conversion buffers overlap;
- fragmentation applies to the complete device peak.

The state transformation compiler adds a focused coexistence check for source and target layouts and computes bounded migration chunk count. The generated baseline runtime separately enforces fixed admission/output queues and one worker process.

## Current limitations

Tests exercise challenger/conversion/fragmentation overflow and state-transition coexistence rejection. This is conservative static arithmetic, not a profiler. It does not currently derive temporary tensor liveness, allocator-specific fragmentation, pinned-memory accounting, compiled-graph caches, outstanding transfers, startup disk, thread creation by external libraries, or rollback overlap automatically from executable code. Callers must populate demand from measured or compiler-derived evidence and disclose unresolved risk.

Passing this gate does not establish that a real allocator will succeed. Hardware-backed admission should retain an additional safety margin and compare observed peaks to the estimate. Underestimation invalidates the relevant capsule evidence.

