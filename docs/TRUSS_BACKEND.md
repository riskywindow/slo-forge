# Baseten Truss exporter

The Truss backend is an offline packaging emitter that generates a conventional Truss model directory from a subset of the plan. It is an independent integration and is not represented as officially supported by Baseten.

## Generated files

```text
config.yaml
model/__init__.py
model/model.py
sloforge-export.json
```

`config.yaml` declares model name/description, model class and module directory, pinned Torch/Transformers requirements, CPU/memory and optional accelerator, runtime concurrency, streaming timeout, tracing data and health thresholds. Plan/model metadata is supplied through non-secret environment variables. CPU plans explicitly set `use_gpu: false` in the generated subset.

The `Model` class implements `__init__`, `load` and `predict`. It loads a pinned-revision Transformers text-generation pipeline, validates non-empty prompts, bounds generated tokens and returns a choices-shaped object.

## Schema basis and validation

The first line points editors to the upstream [Truss JSON schema](https://raw.githubusercontent.com/basetenlabs/truss/main/truss/config.schema.json). The field mapping was checked against Baseten's current [Truss configuration reference](https://docs.baseten.co/reference/truss-configuration), which documents `model_class_name`, `model_module_dir`, requirements, resource accelerator strings such as `L4:4`, `runtime.predict_concurrency`, streaming timeout, trace data and health checks.

Normal validation is offline: YAML is parsed, a vendored official-schema subset under [`schemas/truss-config-subset.schema.json`](../schemas/truss-config-subset.schema.json) validates the used fields, and `model.py` is byte-compiled. When the pinned optional package is installed, `TrussHandle(..., validate=True)` performs package-level validation in a bounded subprocess without credentials. The manifest hashes generated files and records `deployed: false`.

## Mapping

| Plan concept | Truss field |
|---|---|
| CPU and RAM | `resources.cpu`, `resources.memory` |
| accelerator and count | `resources.accelerator` |
| request concurrency | `runtime.predict_concurrency` |
| streaming deadline | `runtime.streaming_read_timeout` |
| trace collection | `runtime.enable_tracing_data` |
| startup/health thresholds | `runtime.health_checks` |
| model ID/revision | environment variables read by `model.py` |

Replica min/max, region placement, canary weight, rollback criteria and predictive control are deployment-level concerns not expressible in this generated Truss model directory. They require Baseten deployment configuration or a SLOForge control integration.

## Limitations

- The exporter always generates a Transformers `Model`; it does not currently emit native vLLM/SGLang/TRT-LLM Truss server configuration. Non-Transformers plans must be rejected until that lowering exists, because silently switching engines violates the plan.
- Dtype, quantization, tensor/pipeline parallelism, batching, prefix cache, speculative decode and compilation settings are not mapped into `config.yaml` or `model.py`.
- Validation covers the used official-schema subset, not every upstream schema condition.
- The generated app has not been built or pushed to Baseten on the current machine.
- Model downloads, secrets, GPU compatibility, scale behavior and paid inference are unexercised.

No credential is required for generation. SLOForge does not invoke `truss push` in normal CI. A paid deployment requires explicit review, credentials and the repository budget guard.
