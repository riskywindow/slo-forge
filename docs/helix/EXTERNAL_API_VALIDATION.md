# Helix external API validation

Reviewed on 2026-08-03. Helix reuses Continuum and Fabric adapters instead of adding another RPC
or inspecting engine-private state. The optional dependency lock is installation evidence, not an
execution result.

| System | Reviewed public contract | Locked / accepted boundary | Helix decision |
|---|---|---|---|
| PyTorch | `Module.state_dict`, `load_state_dict`, `torch.load(weights_only=True)`, distributed checkpoint state dictionaries | lock resolves 2.11.0; Continuum accepts >=2.5,<2.14; Helix optional extra narrows to >=2.11,<2.14 | tiny autograd trainer is implemented; unavailable in default CPU environment; no arbitrary process-state portability claim |
| PEFT | `LoraConfig`, `get_peft_model`, `PeftModel.save_pretrained(safe_serialization=True)` | PEFT 0.20.0, Transformers 5.14.1 in the optional lock; PEFT >=0.19.1,<1 | local-model-only LoRA SFT adapter; requires an explicitly controlled environment and a complete Helix batch |
| vLLM | offline `LLM`/`SamplingParams` and v1 KV connector/configuration APIs | vLLM 0.23.0; accepted >=0.9,<0.24 | Continuum connector movement may be used after compatibility analysis; connector movement is not a portable ExecutionStateCapsule |
| SGLang | server arguments and prefill/decode disaggregation configuration | SGLang 0.5.2 in lock; probe verifies symbols at runtime | launch/transfer configuration only; model-derived state is recomputed or rejected unless an explicit portable export exists |
| Genesis | generated runtime loader, bounded streaming, cancellation | in-repository schema 1.0.0 | CPU smoke is available; generated runtimes without a live-state export fail closed for exact migration |
| Docker, Kubernetes, Modal | existing SLOForge exporters and pinned deployment contracts | unchanged by Helix | Helix produces offline plans; no cloud or paid resource is created without the existing explicit deployment controls |

Primary material reviewed:

- <https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html>
- <https://docs.pytorch.org/docs/stable/generated/torch.load.html>
- <https://docs.pytorch.org/docs/stable/fsdp.html>
- <https://huggingface.co/docs/peft/package_reference/lora>
- <https://huggingface.co/docs/peft/package_reference/peft_model>
- <https://huggingface.co/docs/transformers/peft>
- <https://docs.vllm.ai/en/latest/getting_started/quickstart/>
- <https://docs.vllm.ai/en/latest/usage/reproducibility.html>
- <https://docs.vllm.ai/en/latest/api/vllm/config/kv_transfer.html>
- <https://docs.sglang.ai/backend/pd_disaggregation.html>
- <https://github.com/sgl-project/sglang/blob/v0.5.12/python/sglang/srt/server_args.py>

The checked host has no installed optional PyTorch/PEFT/vLLM/SGLang runtime. Their probes and
fixture contracts are exercised, while real training and rollout status remains implemented but
unexercised. No result is inferred from package availability alone.

TRL, VERL, FSDP distributed execution, NVIDIA Dynamo deployment, and new Modal sandbox calls were
not selected for the required path: the reference and PyTorch/PEFT adapters cover the deliberately
narrow trainer boundary, while existing Fabric deployment code already owns Dynamo and Modal
generation. Helix does not claim an integration merely because an upstream framework exists.
