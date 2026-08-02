# Deterministic multi-stage search

Genesis search is a credential-free compiler search loop. Proposal engines produce typed `CandidateDesign` values; an independent `StageEvaluator` supplies scoped evidence. Proposal scores never authorize acceptance.

## Candidate design and proposal engines

A design records a content-identified genome hash, explicit seed, parent candidates, typed transformations, affected genome regions, parameter values, and a finite feature vector. Mutation options declare expected upside and invalidity risk as proposal heuristics, not benchmark evidence.

The local portfolio contains four deterministic engines:

- beam search orders single and cross-layer transformation combinations by declared upside and risk;
- evolutionary search recombines two parent designs and introduces a seeded mutation;
- local search adds the best legal neighboring mutation to a parent;
- novelty search maximizes Jaccard distance from prior transformation signatures.

The portfolio selects round-robin across engines, removes duplicate transformation signatures, and retains unseeded novelty candidates. Every engine enforces the mutable-region whitelist, so Autopsy or an operator can freeze unaffected genome regions. Candidate IDs, genome hashes, and per-stage seeds use canonical SHA-256 identities.

## Multi-fidelity execution

`SearchEngine` supports this ordered ladder:

1. static pruning;
2. analytical lower bound;
3. compilation;
4. digital twin;
5. deterministic reference tests;
6. property verification;
7. bounded model checking;
8. full simulation;
9. hardware microbenchmark;
10. end-to-end benchmark;
11. shadow validation;
12. canary validation.

The lifecycle-bearing stages map to the canonical Genesis `Candidate` states and cannot be skipped. Analytical, digital-twin, and end-to-end evidence remain separately visible events without inventing lifecycle states. A terminal verifier failure maps to an explicit canonical failure state. A passing hardware-microbenchmark stage must identify itself as hardware-backed; synthetic evidence cannot silently advance that gate. Hardware evaluation is rejected unless `allow_hardware` is explicitly set.

Evaluator adapters receive a deterministic stage seed and must report actual resource usage, evidence IDs, a scoped pass/fail result, and optionally an objective vector. The engine preauthorizes a declared maximum before invoking the adapter and then rejects any adapter that reports usage above that reservation.

## Nine-dimensional hard budgets

Budget enforcement is atomic across:

- wall time;
- CPU time;
- GPU time;
- cloud cost;
- external synthesis cost;
- candidate count;
- compilation count;
- benchmark count;
- verifier time.

No stage is called when its maximum reservation does not fit. Actual usage is consumed only after it is checked against both the reservation and total run budget. Tests independently exhaust all nine dimensions. Search does not call paid services or create hardware resources.

## Pareto and experiment selection

The bounded Pareto archive compares correctness confidence, quality, TTFT, token latency, goodput, throughput, cost, energy, startup, memory, reliability, implementation complexity, and transition cost with their correct optimization directions. Dominated candidates are removed. When a nondominated frontier exceeds its bound, deterministic crowding distance preserves tradeoff diversity.

The experiment ranker uses distance-weighted observed utility and distance-to-evidence uncertainty. With no observations it uses only proposal heuristics. Acquisition ordering is deterministic with candidate identity as the final tie breaker. Objective values and utilities come from evaluator evidence; the search engine does not generate benchmark measurements.

## Audit trail

Every proposal, reservation consumption, stage result, lifecycle transition, Pareto update, and budget exhaustion is written as canonical, sorted JSON in a bounded atomic event log. Event sequences are contiguous and reload-validated. Canonical candidate documents are persisted after usage and lifecycle changes. Existing event logs are not silently overwritten.

Run the focused checks with:

```bash
PYTHONPATH=python pytest -q tests/python/test_genesis_search.py
```

The direct evaluator interface is intentionally synchronous. A production generated-code or external-tool adapter must run inside the Genesis sandbox, which supplies the hard process wall-clock and cleanup boundary. The search core itself never launches an uncontrolled subprocess.
