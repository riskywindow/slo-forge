# Helix demo script

The flagship is a deterministic local CPU demonstration. Start from a clean environment and run:

```bash
make check
uv run sloforge helix demo --output artifacts/helix-demo --seed 41
uv run sloforge helix evaluate --output artifacts/helix-evaluation
```

Inspect the emitted manifest before interpreting individual JSON files. Walk the audience through:

1. the champion policy, captured environment/effect/model boundary, and hash-bound branch point;
2. isolated sibling branches and their explicit state-reuse reports;
3. strict-policy trajectories, deterministic reward evidence, and branch-relative credit;
4. the provenance-complete batch, bounded trainer result, and candidate epoch;
5. a separately configured gate rejection, successful shadow/canary path, active-session pinning,
   and rollback;
6. scheduler baseline plans and their serving-hard accounting; and
7. limitations and hypothesis statuses in the evaluation output.

Do not describe supplied resource forecasts as measurements, the tiny policy as an LLM, semantic
replay as exact, or local SQLite atomicity as distributed consensus. Paths and flags may evolve; the
CLI help is authoritative.

The coding task, candidate edits, failure-seed search bound, trainer hyperparameters, and quality
threshold are pre-authored to exercise both rejection and acceptance paths. The evaluation cases are
withheld from the policy/trainer dataflow, but they are part of the same source-controlled benchmark
and are not secret from the demo author. This demonstrates transaction and evidence mechanics, not
generalization, autonomous patch discovery, or independent evaluator governance. A replay comparison
proves restoration only when its artifact also records an actual backend restoration and re-execution;
comparing two constructed traces alone is a comparator test.
