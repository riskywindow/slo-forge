# vLLM Continuum binding

The binding validates `KVTransferConfig` and `KVConnectorBase_V1` before producing
a connector configuration. Those APIs move vLLM-native KV or hidden-state buffers;
they do not establish a portable logical-state ABI, ownership epoch, or token
commit protocol. Continuum performs those checks before selecting this hook.

The host used for normal CI has no vLLM installation or compatible GPU, so only
version/API conformance fixtures execute there. No cross-runtime result is claimed.

## BranchFabric live-state experiment boundary

The BranchFabric GPU experiment has a separate, non-portable adapter boundary
documented in `docs/branchfabric/GPU_RUNTIME_ADAPTER_DECISION.md` and
`docs/branchfabric/REAL_GPU_COW_ADAPTER.md`. It pins vLLM exactly to 0.23.0 and,
only with V1 synchronous in-process execution, observes `KVCacheManager`,
`BlockPool`, `KVCacheBlock`, `KVCacheConfig`, request block tables, and the GPU
model runner's KV tensors.

That path may demonstrate same-engine physical prefix-page sharing and
append-only private suffix allocation. It does not add those live handles to
this portable binding, does not make block IDs stable outside the process, and
does not change the prohibition on claiming complete portable state export.
