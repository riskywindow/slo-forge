# Fabric integration

SLOForge Fabric contributes source/destination topology, path and rail properties, GPU/NIC/NUMA affinity, failure domains, memory tiers, transport support, and bandwidth/latency samples. Continuum contributes state segment sizes and deadlines, memory types, streamability, required-before-use order, exactness, and conversion-placement candidates.

`fabric_transfer_rates` accepts only successful relevant Fabric benchmark cases and retains each raw artifact/hash, sample count, variation, and measurement mode. The joint planner can choose chunking and transfer rate from these samples while WarmPath readiness remains a separate cost term.

`with_continuum_extension` attaches a versioned `sloforge.ai/continuum-migration-v1` document to existing physical-plan extensions. This preserves PhysicalExecutionPlan compatibility and avoids duplicating the Fabric IR. The extension references, rather than embeds, state and conversion artifacts.

On the current host, Fabric integration is exercised with deterministic/synthetic CPU fixtures. No physical GPU-to-NIC, RDMA, or multi-node migration measurement is claimed.
