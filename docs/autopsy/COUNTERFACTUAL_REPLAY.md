# Counterfactual replay

Counterfactual replay asks whether removing a proposed cause from the calibrated
physical twin reduces the observed regression.

## Scenario model

A scenario binds a stable ID and rationale to one diagnosis hypothesis and one or
more unique modifications:

- remove an identified timed fault;
- multiply a resource curve's latency and bandwidth;
- multiply one rank's duration;
- replace one physical resource with another.

The diagnosis kind and hypothesis ID must agree. Duplicate modifications or
scenarios are rejected.

## Execution

The exact degraded request is canonicalized and hashed. Its existing
counterfactual list is copied and extended; evidence is never edited in place.
The Rust simulator runs once for the degraded reference and once per scenario.
Input/output are bounded and execution has a timeout. Malformed or failed output
becomes `simulation_failed` with no numeric estimate.

The flagship persists the exact degraded simulation request, healthy/degraded
runs, scenarios, healthy reference, and SHA-256 bindings in
`artifacts/fabric-demo/autopsy/replay-metadata.json`. Passing that Autopsy
directory to `sloforge autopsy replay --evidence ...` resolves and verifies the
bundle without ad hoc input reconstruction. The older single-file form remains
supported but requires explicit baseline, simulation input, and healthy
reference arguments.

Each successful scenario records simulated makespan and interval, operation and
event counts, output hash, improvement bounds, healthy-reference residual, and
confidence. `attach_counterfactuals` adds the strongest evaluated estimate for
each matching hypothesis, retains the original support and contradiction
statements, and reranks the diagnosis from the causal outcomes.

## Causal interpretation

A supported counterfactual is evidence that the modeled cause is sufficient to
improve the calibrated twin. It is not proof that an unobserved real system has
the same cause. Calibration provenance and held-out residuals constrain the
claim. Combined scenarios can reveal interacting faults even if an individual
signal is diluted.

The flagship replay evaluated seven scenarios and selected
`remove-both-faults`; its exact results are in
`artifacts/fabric-demo/autopsy/counterfactuals.json` and its choice is summarized
in `artifacts/fabric-demo/manifest.json`.
