# GPU benchmark status

Status: **unavailable**

No working nvidia-smi inventory was detected on this host.

No GPU performance numbers are reported by this artifact.

## Reproduction commands

```console
sloforge trace generate --output workloads/gpu-benchmark.jsonl --count 180 --seed 41
sloforge hardware probe --device cuda --hourly-price-usd 0 --output artifacts/hardware/gpu.json
sloforge profile --model Qwen/Qwen3-0.6B --engines transformers,vllm,sglang --hardware artifacts/hardware/gpu.json --trace workloads/gpu-benchmark.jsonl --budget-usd ${SLOFORGE_GPU_BUDGET_USD:-0} --output artifacts/profiles/gpu
```
