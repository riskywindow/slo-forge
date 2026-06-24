# Zero-day model frontend

The Genesis frontend accepts a typed reference package and recovers a conservative model graph without importing the model by default. Its purpose is to make previously unseen Python model code inspectable, not to claim support for arbitrary Python.

## Reference package

A package is a directory containing `reference_package.json` or `reference_package.yaml`. The manifest uses schema version `1.0.0` and names only relative, normalized package paths. Loading rejects parent traversal and missing artifacts. The package identity covers the canonical manifest plus the SHA-256 digest of every source, tokenizer, sample, search-corpus, and final-evaluation artifact.

The manifest explicitly declares:

- reference, tokenizer, and sample-generator modules;
- synchronous model-load, state-allocation, prefill, decode, sampling, tokenization, detokenization, and sample-generation entry points;
- persistent state fields, ownership, mutation atomicity, quantization, cancellation release, and migration support;
- semantic rules for token commitment, batching, request isolation, streaming order, cancellation, retry, and control flow;
- separate search and final quality corpora and their quality metrics;
- bounded tensor/scalar input domains, symbolic dimensions, dtypes, contiguity and stride constraints;
- custom-operator semantics and verification obligations;
- optional workflow steps and software preconditions;
- an optional explicit `torch_export` fixture.

The strict Pydantic definitions live in `python/sloforge/genesis/frontend/models.py`. Core fields are typed and reject unknown keys.

## Inspection boundary

`inspect_reference_package(path)` parses Python ASTs and does not import source code. It records function/operator boundaries, tensor-like calls, custom operators, state reads and writes, aliases, control-flow boundaries, legal batching axes, symbolic shapes, dtypes, and declared state dependencies. Missing entry points and undeclared control flow are unsupported diagnostics. Calls and aliases without a semantic contract become proof obligations instead of inferred facts.

This distinction is deliberate:

- declared facts come from the validated manifest;
- recovered facts come from syntax and source locations;
- unresolved behavior is retained as a diagnostic or proof obligation;
- no unsupported behavior is silently assigned guessed semantics.

`use_torch_export=True` is an explicit execution boundary. When requested, the declared fixture is imported and passed to the installed `torch.export.export(..., strict=True)` API. Genesis records the PyTorch version, FX graph nodes, tensor shape/dtype/stride metadata, declared dynamic shapes, and export range constraints. This path is optional and must run inside the generated-code sandbox for untrusted packages. The CPU static frontend has no PyTorch dependency.

## Output and fixture

Inspection returns an immutable `InspectionResult` and can write canonical JSON for later genome construction. The result contains source hashes, recovered graph, contracts, diagnostics, and optional export evidence.

`models/reference_tasks/hybrid_decoder` is an affordable unseen-style package. It combines sliding-window attention-like computation, recurrent state, sparse expert dispatch, a declared saturating int8 state transform, a speculative head, a custom seeded top-2 sampler, dynamic sequence length, and an optional tensor export fragment. It is a reference implementation only; there is no handwritten serving backend.

Run the frontend tests with:

```bash
PYTHONPATH=python pytest -q tests/python/test_genesis_frontend.py
```

The tests also prove that default inspection does not execute a source file whose module body raises immediately. Optional PyTorch coverage skips only when PyTorch is absent.

## Limits

AST recovery is conservative. It does not prove numerical equivalence, resolve arbitrary Python reflection, infer mutation through opaque native extensions, or treat dynamic control flow as safe. Such cases remain obligations or unsupported diagnostics. `torch.export` provides additional graph evidence but does not replace the semantic and streaming contracts.

The static path and HybridDecoder fixture are exercised in CPU tests. The optional `torch.export` path is exercised only when compatible PyTorch is installed and the fixture declares it. Inspection of arbitrary third-party packages, GPU export graphs, and opaque extension operators is unexercised unless a run artifact explicitly records it.
