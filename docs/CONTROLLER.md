# Adaptive controller

The Python controller evaluates guarded capacity and routing changes over fixed workload windows. Its goal is the lowest declared hourly cost predicted to satisfy TTFT with a safety margin, not maximum utilization.

## Observed state

Each window records arrival rate, sample count, interactive and long-context fractions, p95 prompt/output length, observed p95 TTFT, backend error rate and current replicas/concurrency. Workload drift is the relative deviation from a recent-rate baseline.

The demo evaluator synthesizes observed TTFT from the same explicit queueing relation used by the policy experiment. A production integration should replace this observation adapter with gateway/OTEL measurements and invoke the calibrated Rust simulator for action evaluation.

## Forecast

The predictive baseline uses an exponentially weighted rate estimate:

```text
smoothed_t = alpha * observed_t + (1 - alpha) * smoothed_(t-1)
forecast_t = smoothed_t + 0.65 * max(observed_t - observed_(t-1), 0)
```

The uncertainty envelope is the larger of 10% of forecast and half the latest trend. Candidate safety is tested against the upper envelope, not only its center. The horizon is two control windows.

## Candidate actions

The action type supports hold, replica scaling, concurrency changes, routing changes, admission limiting and prebuilt variant selection. The current evaluator enumerates replicas, selected concurrency values and round-robin/SLO-slack routing. Capacity scales with replicas and the square root of relative concurrency. Queue delay rises nonlinearly near saturation; high concurrency carries an explicit latency penalty.

An action is safe only if predicted TTFT is below `target * (1 - safety_margin)` and predicted utilization is below 0.96. Safe alternatives are ordered by hourly cost, then TTFT and concurrency. If none is safe, the minimum-TTFT action is considered, but the state guards may still force a hold.

## Guarded state machine

```text
stable --safe material change--> cooldown or canary
  ^                                |
  |       cooldown expires         |
  +--------------------------------+

canary --promotion failure--> rollback_cooldown --> stable
```

Guards include:

- minimum sample count;
- cooldown windows after a mutation;
- bounded changes per rolling hour;
- configured min/max replicas and concurrency;
- safety margin against the upper forecast;
- promotion limit for canary TTFT.

Before a material change the controller saves a complete rollback action. Routing or variant changes enter canary. If measured p95 TTFT exceeds `target * (1 + safety_margin)`, the old replicas, concurrency and routing are restored and rollback cooldown begins. Every decision records the observation, forecast, all evaluated alternatives, chosen action, checks, state transition, outcome and rollback information.

## Reactive baseline

The reactive policy increments or decrements one replica when observed utilization crosses scale-up or scale-down hysteresis. It shares minimum-sample and cooldown protections but does not forecast or change routing/concurrency. This makes the baseline comparable while preserving basic production safety.

## CPU evaluation

<!-- Metrics source: ../artifacts/demo/controller/evaluation.json -->

Across 16 one-second windows, the predictive controller made 2 scale actions, incurred 0 SLO-violation windows, exposed 1 cold-start window, changed replica direction once and accumulated 0.015889 USD of modeled cost. The reactive baseline made 2 scale actions, incurred 1 violation window and 2 cold exposures at 0.014056 USD. The predictive policy therefore traded 0.001833 USD more modeled cost for one fewer violation in this single seeded CPU/mock trace.

No canary or rollback was triggered in this run. Zero rollback count is not evidence of successful rollback under load; contract tests cover the state path, and a dedicated canary-triggering evaluation remains necessary before production use.

## Integration contract

The compiler embeds control interval, cooldowns, sample minimum, safety margin and maximum change into the plan. A runtime adapter must apply actions atomically, surface actual state, and append the completed `DecisionRecord` to evidence. Failure to apply an action must leave the prior state authoritative. Cloud mutations are never executed by the offline demo.
