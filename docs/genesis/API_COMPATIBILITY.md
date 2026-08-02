# Genesis external API compatibility record

Reviewed on 2026-08-02. Genesis does not import optional GPU or serving stacks on its CPU path.
Unstable integrations remain behind explicit adapters and version checks; absence is reported and is
never replaced by a different engine or device.

## Exercised environment

The locked project environment used for acceptance contained Modal 1.5.3 and Truss 0.18.24. It did
not contain PyTorch, Triton, `z3-solver`, vLLM, or SGLang. The host supplied the Z3 4.15.1 command,
but Genesis CI uses its self-contained Rust explicit-state checker and does not make Z3 part of the
trusted acceptance path. No `nvcc`, NVIDIA GPU, NCCL runtime, DeepEP, or NIXL installation was
available. The host Python outside the locked environment happened to contain PyTorch 2.10.0; it was
not used as acceptance evidence.

## Reviewed interfaces and disposition

| Component | Reviewed primary source | Genesis disposition |
|---|---|---|
| PyTorch export and compile | [export programming model](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/programming_model.html), [export API](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/api_reference.html) | Optional strict, sandboxed subprocess adapter. Default inspection is non-executing AST analysis. Dynamic-shape and SSA metadata are accepted only when explicitly recovered; otherwise tensor algebra remains an unresolved obligation. Unexercised in the locked environment. |
| Triton | [official documentation](https://triton-lang.org/main/index.html) | Optional generated-kernel backend only. The exercised kernel lab used restricted CPU source and makes no Triton or GPU claim. |
| CUDA | [CUDA 13.3 documentation](https://docs.nvidia.com/cuda/), [programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) | Adapter/schema path only; unavailable and unexercised. No device fallback. |
| CuTe/CUTLASS | [CuTe quick start](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html) | Design reference only; no generated CuTe artifact is accepted or benchmarked. |
| Z3 | [official Z3 guide](https://microsoft.github.io/z3guide/) | Host tool inventory only. Normal checking is the checked Rust bounded explorer, so CI has no external solver dependency. |
| vLLM | [vLLM V1 guide](https://docs.vllm.ai/en/stable/usage/v1_guide/) | Offline command generation uses the Fabric compatibility lock (`0.26.x`). No runtime execution occurred. |
| SGLang | [official documentation](https://docs.sglang.io/) | Offline command generation uses the Fabric compatibility lock (`0.5.x`). No runtime execution occurred. |
| NVIDIA Dynamo | [architecture](https://docs.nvidia.com/dynamo/dev/knowledge-base/overview), [support matrix](https://docs.nvidia.com/dynamo/latest/resources/support-matrix) | Offline CRD/command adapter uses the reviewed 1.3 compatibility range. No cluster execution occurred. |
| NCCL | [NCCL 2.30 documentation](https://docs.nvidia.com/deeplearning/nccl/index.html) | Fabric profile/plan adapter only; no collective ran on this host. |
| DeepEP | [official source](https://github.com/deepseek-ai/DeepEP) | Optional expert-parallel evidence source only; unavailable. |
| NIXL | [official source and design](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md) | Optional transfer adapter only; unavailable. |
| Modal | [GPU resource API](https://modal.com/docs/guide/gpu) | Exact optional dependency 1.5.3; offline metadata generation only. GPU creation requires the existing explicit budget/deployment gate. |
| Truss | [configuration reference](https://docs.baseten.co/reference/truss-configuration) | Exact optional dependency 0.18.24; offline metadata generation only. No Baseten deployment occurred. |

The detailed serving/exporter compatibility pins and reviewed field lists remain in
`deploy/fabric/validated-versions.json`. That record is reused rather than introducing a competing
Genesis RPC or deployment-version registry.
