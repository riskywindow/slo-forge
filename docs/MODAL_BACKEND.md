# Modal exporter

The Modal backend is an offline packaging emitter. It maps a subset of a validated plan to a pinned Modal Python application, always performs syntax checks, and optionally imports the app when the pinned SDK is installed. It does not contact Modal, authenticate or create resources.

## Generated files

```text
app.py
requirements.txt
README.md
sloforge-export.json
```

The app uses `modal.Image.debian_slim(...).uv_pip_install`, `modal.App`, `@app.cls`, `@modal.concurrent`, `@modal.enter` and `@modal.fastapi_endpoint`. Plan fields map to GPU string/count, CPU, memory, minimum/maximum containers, concurrent target/limit, scaledown window, region and optional memory snapshot. The generated `Inference` class selects a plan-matching Transformers pipeline, vLLM `LLM`, SGLang `Engine`, TensorRT-LLM `LLM`, or explicit mock path for loading and generation. Runtime-specific package versions are pinned in the generated image; `requirements.txt` pins `modal==1.5.3` for the generator revision.

## Offline validation

Validation parses the Python AST, requires the current class/concurrency/lifecycle/web decorator calls, rejects source containing remote invocation or deployment calls, and byte-compiles the module. When the pinned optional SDK is installed, validation also imports the generated file in a bounded subprocess without credentials. The export manifest hashes every generated file and records `deployed: false`.

The generated module declares resources lazily. SLOForge never runs `modal deploy` as part of generation, tests or the CPU demo. Paid execution requires a separate explicit operator action and must be authorized under `SLOFORGE_GPU_BUDGET_USD`.

## Current API basis

The mapping was checked against Modal's current documentation for [cold-start controls](https://modal.com/docs/guide/cold-start), [`modal.enter`](https://modal.com/docs/sdk/py/latest/modal.enter) and [memory snapshots](https://modal.com/docs/guide/memory-snapshots), then exercised by importing the generated app with Modal 1.5.3. Modal documents `min_containers` and `scaledown_window` as warm-capacity controls, and documents `enable_memory_snapshot=True` plus `@modal.enter(snap=True)` for snapshot capture.

Memory snapshots need care. Without the separate experimental GPU-snapshot option, GPUs are unavailable inside the snapshot phase; GPU snapshots are alpha and generally problematic for multi-GPU programs. The generator exposes ordinary memory snapshots only when the plan opts in. It does not claim transparent GPU-state snapshot support, and multi-GPU snapshot plans should keep this disabled unless independently validated.

## Limitations

- Engine selection is explicit and never silently replaced, but the non-mock engine paths were not image-built or executed on this CPU-only host.
- Dtype, quantization, tensor/pipeline parallelism, batching, prefix cache, speculative decode, compilation, routing, canary and rollback semantics are not fully lowered into the generated application.
- Static compilation proves API-shaped syntax, not SDK runtime compatibility, image build success, model license/download access, GPU availability or regional capacity.
- Autoscaling maps only min/max/target concurrency and scaledown; plan cooldown, canary and rollback semantics need an external controller.
- No cloud run was executed on this machine, and no Modal performance claim is made.

## Operator validation

After reviewing generated code and budget:

```bash
cd artifacts/demo/exports/modal
python -m py_compile app.py
python -m venv .modal-venv
.modal-venv/bin/pip install -r requirements.txt
.modal-venv/bin/python -c 'import app'
```

Import may build a lazy object graph but should not invoke remote work. Deployment is deliberately omitted here. Before any paid command, estimate maximum cost, reserve 15% of `SLOFORGE_GPU_BUDGET_USD` for reruns and record spend.
