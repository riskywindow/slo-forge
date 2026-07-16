# WarmPath integration

WarmPath prepares reconstructible destination resources: runtime image, model/tokenizer, compiled kernels, communication groups, pinned buffers, allocations, and readiness metadata. Active session chunks are not transferred until compatibility analysis has established the destination contract.

`plan_with_fabric_and_warmpath` consumes `WarmPathPlan.predicted_p95_ready_time_ms` alongside state transfer and conversion estimates. This exposes whether warming can overlap pre-copy, whether a warm replica is economical, and whether destination startup dominates interruption.

WarmPath does not import or own logical session state, choose an exactness class, or commit ownership. Continuum validates the imported state and its continuation before the coordinator changes epoch.
