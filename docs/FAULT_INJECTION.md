# Fault injection and bottleneck diagnosis

SLOForge has two fault paths: live deterministic faults exposed by Rust mock backends, and typed YAML scenarios used for repeatable diagnosis evaluation. Neither path is enabled against arbitrary production backends.

## Scenario format

```yaml
schema_version: sloforge.chaos/v1
name: example
seed: 41
events:
  - fault_id: slow-backend
    fault_type: backend_slowdown
    start_ms: 1500
    duration_ms: 900
    backend_id: balanced
    magnitude: 3.0
    probability: 1.0
```

IDs must be unique; time, duration, magnitude and probability are validated. A seeded RNG determines application when probability is below one.

Supported types are backend crash, backend slowdown, startup slowdown, request errors, queue saturation, capacity loss, network jitter and simulated OOM. Rust simulation additionally supports timed recovery, replica add/remove and exact network/capacity mutations.

## Live mock injection

The demo starts three mock backend processes with admin fault endpoints. A scheduled task applies slowdown, crash, clear/recovery and cold-start commands while actual HTTP traffic flows through the gateway. Responses, target URL and application timestamp are written to the gateway replay artifact. The endpoints are localhost test facilities and must not be exposed in a production image.

## Diagnosis model

The classifier consumes a `CounterSnapshot`:

- gateway and backend queue latency;
- startup, prefill, decode and network latency;
- arrivals and admitted capacity;
- backend error rate and healthy fraction;
- memory rejections and warm-replica fraction.

It distinguishes arrival overload, gateway queueing, insufficient warm capacity, cold-start dominance, prefill dominance, decode dominance, unhealthy backend and configuration infeasibility. Concrete signals use guarded precedence: memory rejection, health failure, dominant startup, dominant gateway/network queue, insufficient warm capacity and prefill/decode split. The remaining cases use normalized scores.

The output contains expected and predicted labels, confidence, evidence string, counterfactual improvement, correctness and modeled diagnosis latency. The latency is a deterministic function of configured fault duration, not measured classifier wall time. Confidence is a rule score, not a calibrated probability. This is deterministic rules-and-counters analysis; no LLM generates the explanation.

## Current evaluation

<!-- Metrics source: ../artifacts/demo/chaos/result.json -->

All 8 required events were applied in the seed-41 CPU scenario and all 8 labels matched, for 100% label agreement, 0 reported false-positive rate and 17.54 ms mean modeled diagnosis latency. The scenario had no non-applied events, so the zero false-positive value has no denominator and is not evidence of specificity. Rule scores ranged from 0.50 to 0.99. This is a closed-set self-consistency test: the injector and classifier share known counter mappings. It does not establish accuracy, latency or calibrated confidence on unlabeled production incidents.

The live gateway run separately applied slowdown, crash, recovery and cold-start commands while 120 requests completed. The simulator replay produced 4 failed requests and 49 deadline misses under its timed actions. Keeping the live, simulator and classifier artifacts separate prevents a synthetic classifier score from being mistaken for transport failure evidence.

## Running

```bash
sloforge chaos --scenario scenarios/cpu-demo.yaml \
  --output /tmp/sloforge-chaos-result.json
```

Inspect every incorrect or low-confidence row rather than reporting only aggregate accuracy. When adding a fault, add its expected label, counter mutation, deterministic test and a scenario case. Fault commands must remain bounded and reversible, and demo processes must be cleaned up even when replay fails.
