# Topology-aware collective rank ordering

This experiment uses the checked-in physical topology, physical plan, and
collective trace evidence to compare the plan's sequential ring with an exact
search for small groups or a bounded greedy/2-opt search for larger groups.
Every candidate route requires measured latency and bandwidth curves. There is
no theoretical-bandwidth or transport fallback.

Run from the repository root:

```bash
uv run --locked python benchmarks/fabric/rank_ordering/run_experiment.py
```

The checked fixture is a deterministic synthetic calibration and is explicitly
labeled as such. It exercises four asymmetric GPU links on CPU and does not
claim a GPU measurement. The integration gate requires a fault-free trace,
sufficient collective critical-path share, and a paired bootstrap lower bound
above the practical threshold in every message-size regime. Synthetic evidence
can only produce `measure_on_hardware`; it can never enable the optimization by
default.

Outputs:

- `results/rank-ordering-experiment.json`: content-addressed result and decision
- `results/rank-ordering-raw-samples.jsonl`: warmup and measured paired trials
- `reports/rank-ordering-experiment.md`: report derived from the result object

To evaluate a hardware-backed run, materialize the trace evidence locally and
create the same strict input bundle with `calibration_mode=measured_hardware`,
measured curves from the Fabric profiler, and evidence from a matched fault-free
Autopsy capture. The guard verifies trace bytes and gate fields and rejects
synthetic topology or evidence mislabeled as measured hardware. Even with
hardware-backed inputs, this script runs modeled paired trials; it can recommend
an on-device A/B experiment but cannot enable the ordering by default.
