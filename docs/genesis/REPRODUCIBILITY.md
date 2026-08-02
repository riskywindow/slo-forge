# Genesis reproducibility

Every deterministic Genesis path accepts an explicit seed. The seed is part of genome, candidate, model-check, policy-mutation, benchmark, generated-runtime, evolution, CEGIS and ServingSynthBench identities. Derived seeds use stable content plus SHA-256 rather than process-randomized Python hashing.

## Content identity

Genesis uses canonical UTF-8 JSON with sorted keys, no insignificant whitespace and finite numbers. Python and Rust canonicalize the same golden fixtures to identical bytes and SHA-256 values. Reference-package identity covers the canonical manifest and every declared source, tokenizer, sample and evaluation artifact. Generated runtimes recheck that identity before loading the model.

Artifact and capsule publication is content-addressed or publish-once. Writers use same-directory temporary files, `fsync` where implemented, and atomic replacement; generation refuses to overwrite non-identical existing artifacts. Run commands should therefore use a fresh output directory instead of deleting or mutating prior evidence.

Schema versions, compiler/verifier versions, dependency locks, workload/hardware contracts and fingerprints belong in capsule validation context. Known alpha IR migration is explicit and lossless. Unknown versions fail rather than silently changing interpretation.

## Deterministic versus measured outputs

The following should reproduce byte-for-byte given identical inputs, seed and supported dependency versions:

- static inspection and package hashes;
- baseline genome and generated-runtime source/configuration;
- policy parsing, formatting, bytecode, simplification and mutation;
- tensor rewrite ordering and structural keys;
- local CEGIS proposals, minimized schedule and learned constraint;
- bounded model-check results for an identical request;
- task generation and run ordering in ServingSynthBench.

Wall-clock benchmark samples are observations and are not expected to be byte-identical. Reproducibility for performance means retaining raw samples, warmup, randomized order, workload/hardware/software fingerprints, seed, summary method and uncertainty. A new run creates new evidence; it does not overwrite the first run.

## Local checks

Focused CPU checks can be run without a GPU or external credentials:

```bash
PYTHONPATH=python pytest -q tests/python/test_genesis_ir.py
PYTHONPATH=python pytest -q tests/python/test_genesis_frontend.py
PYTHONPATH=python pytest -q tests/python/test_genesis_compiler.py
PYTHONPATH=python pytest -q tests/python/test_genesis_runtime.py
PYTHONPATH=python pytest -q tests/python/test_genesis_policy_dsl.py
PYTHONPATH=python pytest -q tests/python/test_genesis_tensor_rewrites.py
PYTHONPATH=python pytest -q tests/python/test_genesis_state_transforms.py
PYTHONPATH=python pytest -q tests/python/test_genesis_verification.py
cargo test -p sloforge-genesis-ir -p sloforge-genesis-modelcheck
```

The project-level `make genesis-check` and demo targets are the integration authority when present in the current checkout. Focused tests do not replace them.

## Environment-dependent paths

Optional `torch.export` inspection records the installed PyTorch version and graph evidence. Strict generated-code execution requires an accepted macOS `sandbox-exec` or Linux `bubblewrap` capability; missing isolation fails closed. GPU, multi-node, external synthesis, cloud deployment and live promotion require their explicit environment opt-ins and budgets. Absence of an opt-in is a reproducible non-execution result, not permission to fall back silently.

