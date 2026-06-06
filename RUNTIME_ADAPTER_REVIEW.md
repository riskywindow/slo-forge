# SLOForge Fabric runtime-adapter review

Review date: 2026-08-01
Review scope: vLLM, SGLang, NVIDIA Dynamo, Kubernetes GPU/RDMA resources,
Modal, and Baseten Truss
Validation mode: official documentation and tagged source, lock-file inspection,
offline rendering, schema tests, and locally installed SDK/CLI introspection. No
GPU workload, Kubernetes admission request, external deployment, or billable
resource was used.

## Result

The adapter boundary is fail closed for physical plans it cannot represent. The
review fixed four correctness defects:

1. Runtime data parallelism was previously copied from the global physical plan
   into every process group. Direct vLLM and SGLang groups now use DP 1; Dynamo
   vLLM components use the number of complete replicas in that worker role.
2. vLLM expert parallelism is accepted only when the physical EP degree equals
   the runtime's `TP_SIZE * rendered DP_SIZE`. SGLang EP is emitted directly and
   never implicitly enables DP attention.
3. Direct vLLM P/D disaggregation emits only native `--kv-transfer-config` with
   `NixlConnector`. `--disaggregation-mode` is emitted only by the
   `python -m dynamo.vllm` wrapper, where Dynamo documents it as a wrapper-owned
   option.
4. The DGD subset no longer accepts `runtimeVersionOverride`. That field is on
   Dynamo's development branch but is absent from the tagged v1.3.0 API. Worker
   probes are deliberately left for the v1.2-v1.3 operator to inject into the
   `main` container, and the generated component records that contract in its
   annotations.

## Validated interfaces

| System | Reviewed version/surface | Contract used by SLOForge | Official evidence |
| --- | --- | --- | --- |
| vLLM | locked and current PyPI release 0.26.0 | Native TP/PP/DP flags, EP enablement with `EP_SIZE = TP_SIZE * DP_SIZE`, and NIXL KV connector configuration | [Expert-parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/), [disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/), [PyPI release](https://pypi.org/project/vllm/) |
| SGLang | locked 0.5.2; current 0.5.15.post1 surface also reviewed | Canonical `--tp-size`, `--pp-size`, `--dp-size`, `--ep-size`; explicit P/D mode, transfer backend, and IB device; DP attention kept separate from EP | [v0.5.2 server arguments](https://github.com/sgl-project/sglang/blob/v0.5.2/python/sglang/srt/server_args.py), [current server arguments](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py), [expert parallelism](https://docs.sglang.io/docs/advanced_features/expert_parallelism), [P/D disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) |
| NVIDIA Dynamo | v1.2.0 through tagged v1.3.0 | Served `nvidia.com/v1beta1` DGD component list, `podTemplate` with a `main` container, Grove/LWS multi-node selection, and wrapper-owned disaggregation mode | [v1.2 release](https://docs.nvidia.com/dynamo/dev/reference/releases/v1-2-0), [v1.3 release](https://docs.nvidia.com/dynamo/dev/reference/releases/v1-3-0), [DGD guide](https://docs.nvidia.com/dynamo/dev/kubernetes/model-deployment/deploy-with-dgd), [vLLM wrapper reference](https://docs.nvidia.com/dynamo/backends/v-llm/reference-guide) |
| Dynamo probes | tagged v1.3.0 operator | Operator injects worker liveness `/live`, readiness `/health`, and startup `/live` probes on the `system` port (9090); user-specified probes would override these defaults | [tagged shared component type](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/deploy/operator/api/v1beta1/dynamocomponentdeployment_types.go), [tagged probe contract](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/deploy/operator/docs/footer.md) |
| Kubernetes | current stable v1.36 documentation; emitted stable fields valid from 1.31 | Node affinity, topology spread, rolling Deployments, and vendor extended resources. Generic multi-node export is rejected because this renderer does not emit a PodGroup, LeaderWorkerSet, or another atomic gang contract | [v1.36 release](https://kubernetes.io/blog/2026/04/22/kubernetes-v1-36-release/), [device plugins and extended resources](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/), [node affinity](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/) |
| Modal | pinned and locally installed 1.5.3 | `Image.env(dict[str, str])` only; physical placement remains advisory | [current `Image.env` SDK reference](https://modal.com/docs/sdk/py/latest/Image#modal.Image.env) |
| Truss | pinned and locally installed 0.18.24 | String-valued `environment_variables` and flexible `model_metadata` only; physical placement remains advisory | [Truss configuration](https://docs.baseten.co/reference/truss-configuration), [upstream schema](https://github.com/basetenlabs/truss/blob/main/truss/config.schema.json) |

The machine-readable version and field record is
`deploy/fabric/validated-versions.json`. The emitted Dynamo subset is constrained
by `deploy/dynamo/dgd-v1beta1-subset.schema.json`, which intentionally excludes
development-branch-only CRD fields.

## Validation executed

- The project lock resolves vLLM 0.26.0, SGLang 0.5.2, Modal 1.5.3, and Truss
  0.18.24.
- The project environment exposed Modal 1.5.3 with
  `Image.env(self, vars: dict[str, str])` and Truss CLI 0.18.24.
- vLLM, SGLang, Dynamo, and the Kubernetes Python client were not installed in
  the base environment. Their commands and CRDs were therefore validated
  statically, not executed.
- Adapter tests validate DGD JSON Schema, direct-versus-wrapper CLI separation,
  role-local DP, EP representability, transfer-backend consistency, official
  provenance, and rejection of pre-v1beta1 Dynamo and unenforced multi-node
  Kubernetes plans.

## Residual boundaries

- The DGD was not admitted by a live Dynamo v1.3 operator. The schema is an
  intentionally strict subset, not a substitute for server-side CRD validation.
- Physical GPU UUID, NUMA, NIC, and rail bindings are carried as SLOForge
  assertions/metadata. Kubernetes device-resource allocation does not guarantee
  those exact identities; targets without an enforcement mechanism remain
  advisory or are rejected.
- Direct vLLM and SGLang adapters do not generate a multi-host rendezvous, so a
  replica spanning hosts is rejected. The multi-node engine path is Dynamo DGD.
- SGLang role-level independent replica lowering through Dynamo is rejected:
  SGLang DP/DP-attention is not interchangeable with multiple independent
  physical replicas.
- The vLLM disaggregation adapter supports the reviewed NIXL connector only.
  Other upstream connectors are not inferred or silently substituted.
- Modal and Truss outputs contain supported metadata only and do not claim to
  enforce rank placement. No credentials or external mutations are used.
