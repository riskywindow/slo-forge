from __future__ import annotations

import math
from collections import deque
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sloforge.trace import TraceRequest
from sloforge.util import percentile, utc_now


class ControllerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_replicas: int = Field(default=1, ge=1)
    max_replicas: int = Field(default=5, ge=1)
    initial_replicas: int = Field(default=1, ge=1)
    min_concurrency: int = Field(default=1, ge=1)
    max_concurrency: int = Field(default=8, ge=1)
    initial_concurrency: int = Field(default=2, ge=1)
    capacity_per_replica_rps: float = Field(default=6.5, gt=0)
    target_p95_ttft_ms: float = Field(default=250.0, gt=0)
    safety_margin: float = Field(default=0.15, ge=0, lt=1)
    cooldown_windows: int = Field(default=2, ge=0)
    min_samples: int = Field(default=8, ge=1)
    max_changes_per_hour: int = Field(default=12, ge=1)
    forecast_alpha: float = Field(default=0.55, gt=0, le=1)
    scale_up_hysteresis: float = Field(default=0.78, gt=0, le=1)
    scale_down_hysteresis: float = Field(default=0.38, ge=0, lt=1)
    hourly_replica_cost_usd: float = Field(default=1.0, ge=0)
    calibrated_base_ttft_ms: float = Field(default=38.0, gt=0)
    calibrated_prefill_ms_per_token: float = Field(default=0.0, ge=0)
    profiled_prompt_p95_tokens: float = Field(default=512.0, gt=0)
    calibrated_cold_start_p95_ms: float = Field(default=85.0, ge=0)


class ObservedState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_index: int
    window_start_ms: float
    window_end_ms: float
    sample_count: int
    arrival_rate_rps: float
    interactive_fraction: float
    long_context_fraction: float
    prompt_p95: float
    output_p95: float
    observed_p95_ttft_ms: float
    backend_error_rate: float
    replicas: int
    concurrency: int


class Forecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arrival_rate_rps: float
    lower_rps: float
    upper_rps: float
    drift_score: float
    horizon_windows: int


class ControllerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal[
        "hold", "scale", "change_concurrency", "change_routing", "admission_limit", "select_variant"
    ]
    target_replicas: int
    target_concurrency: int
    routing_policy: Literal["round_robin", "slo_slack"]
    admission_limit_rps: float | None = None
    variant: str = "balanced"


class EvaluatedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ControllerAction
    predicted_p95_ttft_ms: float
    predicted_cost_per_hour_usd: float
    predicted_utilization: float
    predicted_slo_attainment: float
    safe: bool
    rejection_reason: str | None = None


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    generated_at: str
    observed: ObservedState
    forecast: Forecast
    alternatives: list[EvaluatedAction]
    chosen: ControllerAction
    safety_checks: list[str]
    controller_state_before: str
    controller_state_after: str
    canary: bool
    actual_outcome_p95_ttft_ms: float | None = None
    rolled_back: bool = False
    rollback_reason: str | None = None


class ControllerSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller: Literal["predictive", "reactive"]
    slo_violations: int
    estimated_cost_usd: float
    recovery_windows: int
    replica_oscillations: int
    cold_start_exposure: int
    action_count: int
    rollback_count: int


class ControllerEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.controller-evaluation/v1"] = (
        "sloforge.controller-evaluation/v1"
    )
    predictive: ControllerSummary
    reactive: ControllerSummary
    predictive_decisions: list[DecisionRecord]
    reactive_decisions: list[DecisionRecord]
    windows: list[ObservedState]


class PredictiveController:
    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.replicas = config.initial_replicas
        self.concurrency = config.initial_concurrency
        self.routing_policy: Literal["round_robin", "slo_slack"] = "round_robin"
        self.variant = "balanced"
        self.state = "stable"
        self.cooldown_remaining = 0
        self.forecast_rps: float | None = None
        self.recent_rates: deque[float] = deque(maxlen=8)
        self.change_windows: deque[int] = deque()
        self.previous_action: ControllerAction | None = None
        self.decision_count = 0

    def _forecast(self, observed: ObservedState) -> Forecast:
        previous = self.forecast_rps if self.forecast_rps is not None else observed.arrival_rate_rps
        alpha = self.config.forecast_alpha
        trend = observed.arrival_rate_rps - (
            self.recent_rates[-1] if self.recent_rates else observed.arrival_rate_rps
        )
        smoothed = alpha * observed.arrival_rate_rps + (1.0 - alpha) * previous
        projected = max(0.0, smoothed + max(0.0, trend) * 0.65)
        self.forecast_rps = smoothed
        baseline = (
            sum(self.recent_rates) / len(self.recent_rates)
            if self.recent_rates
            else observed.arrival_rate_rps
        )
        drift = abs(observed.arrival_rate_rps - baseline) / max(baseline, 1e-6)
        self.recent_rates.append(observed.arrival_rate_rps)
        spread = max(0.10 * projected, abs(trend) * 0.5)
        return Forecast(
            arrival_rate_rps=projected,
            lower_rps=max(0.0, projected - spread),
            upper_rps=projected + spread,
            drift_score=drift,
            horizon_windows=2,
        )

    def _evaluate_action(
        self, action: ControllerAction, forecast: Forecast, observed: ObservedState
    ) -> EvaluatedAction:
        capacity = (
            action.target_replicas
            * self.config.capacity_per_replica_rps
            * math.sqrt(action.target_concurrency / self.config.initial_concurrency)
        )
        utilization = min(1.5, forecast.upper_rps / max(capacity, 1e-9))
        queue_penalty = 1.0 + utilization**2 / max(1.0 - min(utilization, 0.99), 0.03)
        routing_gain = 0.88 if action.routing_policy == "slo_slack" else 1.0
        concurrency_penalty = 1.0 + max(0, action.target_concurrency - 4) * 0.045
        workload_adjusted_service = self.config.calibrated_base_ttft_ms + (
            max(0.0, observed.prompt_p95 - self.config.profiled_prompt_p95_tokens)
            * self.config.calibrated_prefill_ms_per_token
        )
        predicted_ttft = (
            workload_adjusted_service * queue_penalty * routing_gain * concurrency_penalty
        )
        attainment = max(0.0, min(1.0, self.config.target_p95_ttft_ms / max(predicted_ttft, 1e-9)))
        safe_limit = self.config.target_p95_ttft_ms * (1.0 - self.config.safety_margin)
        safe = predicted_ttft <= safe_limit and utilization < 0.96
        reason = (
            None
            if safe
            else f"upper forecast predicts {predicted_ttft:.1f}ms TTFT at {utilization:.2f} utilization"
        )
        return EvaluatedAction(
            action=action,
            predicted_p95_ttft_ms=predicted_ttft,
            predicted_cost_per_hour_usd=action.target_replicas
            * self.config.hourly_replica_cost_usd,
            predicted_utilization=utilization,
            predicted_slo_attainment=attainment,
            safe=safe,
            rejection_reason=reason,
        )

    def decide(self, observed: ObservedState) -> DecisionRecord:
        self.decision_count += 1
        forecast = self._forecast(observed)
        before = self.state
        checks: list[str] = []
        actions: list[ControllerAction] = []
        for replicas in range(self.config.min_replicas, self.config.max_replicas + 1):
            for concurrency in sorted(
                {
                    self.config.min_concurrency,
                    self.concurrency,
                    min(4, self.config.max_concurrency),
                    self.config.max_concurrency,
                }
            ):
                routing: Literal["round_robin", "slo_slack"] = (
                    "slo_slack" if forecast.drift_score > 0.35 else self.routing_policy
                )
                action_type: Literal[
                    "hold",
                    "scale",
                    "change_concurrency",
                    "change_routing",
                    "admission_limit",
                    "select_variant",
                ]
                if replicas != self.replicas:
                    action_type = "scale"
                elif concurrency != self.concurrency:
                    action_type = "change_concurrency"
                elif routing != self.routing_policy:
                    action_type = "change_routing"
                else:
                    action_type = "hold"
                actions.append(
                    ControllerAction(
                        action_type=action_type,
                        target_replicas=replicas,
                        target_concurrency=concurrency,
                        routing_policy=routing,
                    )
                )
        evaluations = [self._evaluate_action(action, forecast, observed) for action in actions]
        safe = sorted(
            (item for item in evaluations if item.safe),
            key=lambda item: (
                item.predicted_cost_per_hour_usd,
                item.predicted_p95_ttft_ms,
                item.action.target_concurrency,
            ),
        )
        chosen_evaluation = (
            safe[0] if safe else min(evaluations, key=lambda item: item.predicted_p95_ttft_ms)
        )
        chosen = chosen_evaluation.action
        if observed.sample_count < self.config.min_samples:
            checks.append("minimum sample guard forced hold")
            chosen = ControllerAction(
                action_type="hold",
                target_replicas=self.replicas,
                target_concurrency=self.concurrency,
                routing_policy=self.routing_policy,
            )
        elif self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            checks.append("cooldown guard forced hold")
            chosen = ControllerAction(
                action_type="hold",
                target_replicas=self.replicas,
                target_concurrency=self.concurrency,
                routing_policy=self.routing_policy,
            )
        else:
            while self.change_windows and observed.window_index - self.change_windows[0] >= 360:
                self.change_windows.popleft()
            if len(self.change_windows) >= self.config.max_changes_per_hour:
                checks.append("change budget forced hold")
                chosen = ControllerAction(
                    action_type="hold",
                    target_replicas=self.replicas,
                    target_concurrency=self.concurrency,
                    routing_policy=self.routing_policy,
                )
            else:
                checks.extend(
                    ["minimum samples satisfied", "cooldown satisfied", "change budget satisfied"]
                )
        canary = chosen.action_type in {"change_routing", "select_variant"}
        if chosen.action_type != "hold":
            self.previous_action = ControllerAction(
                action_type="hold",
                target_replicas=self.replicas,
                target_concurrency=self.concurrency,
                routing_policy=self.routing_policy,
                variant=self.variant,
            )
            self.replicas = chosen.target_replicas
            self.concurrency = chosen.target_concurrency
            self.routing_policy = chosen.routing_policy
            self.cooldown_remaining = self.config.cooldown_windows
            self.change_windows.append(observed.window_index)
            self.state = "canary" if canary else "cooldown"
        elif self.cooldown_remaining == 0:
            self.state = "stable"
        return DecisionRecord(
            decision_id=f"predictive-{observed.window_index:05d}",
            generated_at=utc_now(),
            observed=observed,
            forecast=forecast,
            alternatives=evaluations,
            chosen=chosen,
            safety_checks=checks,
            controller_state_before=before,
            controller_state_after=self.state,
            canary=canary,
        )

    def record_outcome(self, decision: DecisionRecord, actual_p95_ttft_ms: float) -> DecisionRecord:
        decision.actual_outcome_p95_ttft_ms = actual_p95_ttft_ms
        promotion_limit = self.config.target_p95_ttft_ms * (1.0 + self.config.safety_margin)
        if (
            decision.canary
            and actual_p95_ttft_ms > promotion_limit
            and self.previous_action is not None
        ):
            rollback = self.previous_action
            self.replicas = rollback.target_replicas
            self.concurrency = rollback.target_concurrency
            self.routing_policy = rollback.routing_policy
            self.state = "rollback_cooldown"
            self.cooldown_remaining = self.config.cooldown_windows
            decision.rolled_back = True
            decision.rollback_reason = f"canary TTFT {actual_p95_ttft_ms:.1f}ms exceeded promotion limit {promotion_limit:.1f}ms"
            decision.controller_state_after = self.state
        return decision


class ReactiveController(PredictiveController):
    def decide(self, observed: ObservedState) -> DecisionRecord:
        self.decision_count += 1
        capacity = self.replicas * self.config.capacity_per_replica_rps
        utilization = observed.arrival_rate_rps / max(capacity, 1e-9)
        target = self.replicas
        if utilization > self.config.scale_up_hysteresis:
            target = min(self.config.max_replicas, self.replicas + 1)
        elif utilization < self.config.scale_down_hysteresis:
            target = max(self.config.min_replicas, self.replicas - 1)
        action_type: Literal[
            "hold",
            "scale",
            "change_concurrency",
            "change_routing",
            "admission_limit",
            "select_variant",
        ] = "scale" if target != self.replicas else "hold"
        chosen = ControllerAction(
            action_type=action_type,
            target_replicas=target,
            target_concurrency=self.concurrency,
            routing_policy="round_robin",
        )
        before = self.state
        if self.cooldown_remaining > 0 or observed.sample_count < self.config.min_samples:
            chosen.action_type = "hold"
            chosen.target_replicas = self.replicas
            self.cooldown_remaining = max(0, self.cooldown_remaining - 1)
        elif chosen.action_type == "scale":
            self.replicas = target
            self.cooldown_remaining = self.config.cooldown_windows
            self.state = "cooldown"
        forecast = Forecast(
            arrival_rate_rps=observed.arrival_rate_rps,
            lower_rps=observed.arrival_rate_rps,
            upper_rps=observed.arrival_rate_rps,
            drift_score=0.0,
            horizon_windows=1,
        )
        evaluated = self._evaluate_action(chosen, forecast, observed)
        return DecisionRecord(
            decision_id=f"reactive-{observed.window_index:05d}",
            generated_at=utc_now(),
            observed=observed,
            forecast=forecast,
            alternatives=[evaluated],
            chosen=chosen,
            safety_checks=["reactive utilization threshold", "cooldown and sample guard"],
            controller_state_before=before,
            controller_state_after=self.state,
            canary=False,
        )


def _window_trace(trace: list[TraceRequest], window_ms: float = 1000.0) -> list[list[TraceRequest]]:
    max_index = int(trace[-1].arrival_ms // window_ms)
    windows: list[list[TraceRequest]] = [[] for _ in range(max_index + 1)]
    for request in trace:
        windows[int(request.arrival_ms // window_ms)].append(request)
    return windows


def _state_for_window(
    requests: list[TraceRequest],
    *,
    index: int,
    replicas: int,
    concurrency: int,
    config: ControllerConfig,
) -> ObservedState:
    count = len(requests)
    rate = float(count)
    prompts = [float(item.prompt_tokens) for item in requests] or [1.0]
    outputs = [float(item.output_tokens) for item in requests] or [1.0]
    utilization = rate / max(replicas * config.capacity_per_replica_rps, 1e-9)
    service = config.calibrated_base_ttft_ms + (
        max(0.0, percentile(prompts, 0.95) - config.profiled_prompt_p95_tokens)
        * config.calibrated_prefill_ms_per_token
    )
    ttft = service * (1.0 + utilization**2 / max(1.0 - min(utilization, 0.99), 0.03))
    return ObservedState(
        window_index=index,
        window_start_ms=index * 1000.0,
        window_end_ms=(index + 1) * 1000.0,
        sample_count=count,
        arrival_rate_rps=rate,
        interactive_fraction=sum(item.request_class == "interactive" for item in requests)
        / max(count, 1),
        long_context_fraction=sum(item.request_class == "long-context" for item in requests)
        / max(count, 1),
        prompt_p95=percentile(prompts, 0.95),
        output_p95=percentile(outputs, 0.95),
        observed_p95_ttft_ms=ttft,
        backend_error_rate=0.0,
        replicas=replicas,
        concurrency=concurrency,
    )


def _run_controller(
    controller: PredictiveController,
    trace: list[TraceRequest],
    name: Literal["predictive", "reactive"],
) -> tuple[ControllerSummary, list[DecisionRecord], list[ObservedState]]:
    decisions: list[DecisionRecord] = []
    states: list[ObservedState] = []
    violations = 0
    cost = 0.0
    cold_exposure = 0
    oscillations = 0
    rollbacks = 0
    recovery_windows = 0
    previous_delta = 0
    in_violation = False
    for index, requests in enumerate(_window_trace(trace)):
        state = _state_for_window(
            requests,
            index=index,
            replicas=controller.replicas,
            concurrency=controller.concurrency,
            config=controller.config,
        )
        states.append(state)
        decision = controller.decide(state)
        delta = decision.chosen.target_replicas - state.replicas
        if delta and previous_delta and delta * previous_delta < 0:
            oscillations += 1
        if delta:
            previous_delta = delta
            if delta > 0:
                cold_exposure += 1
        actual_state = _state_for_window(
            requests,
            index=index,
            replicas=controller.replicas,
            concurrency=controller.concurrency,
            config=controller.config,
        )
        # Cold replicas add one-window startup exposure.
        actual_ttft = actual_state.observed_p95_ttft_ms + (
            controller.config.calibrated_cold_start_p95_ms if delta > 0 else 0.0
        )
        decision = controller.record_outcome(decision, actual_ttft)
        if decision.rolled_back:
            rollbacks += 1
        decisions.append(decision)
        violation = actual_ttft > controller.config.target_p95_ttft_ms
        violations += int(violation)
        if violation:
            in_violation = True
        elif in_violation:
            recovery_windows += 1
            in_violation = False
        cost += controller.replicas * controller.config.hourly_replica_cost_usd / 3600.0
    summary = ControllerSummary(
        controller=name,
        slo_violations=violations,
        estimated_cost_usd=cost,
        recovery_windows=recovery_windows,
        replica_oscillations=oscillations,
        cold_start_exposure=cold_exposure,
        action_count=sum(item.chosen.action_type != "hold" for item in decisions),
        rollback_count=rollbacks,
    )
    return summary, decisions, states


def evaluate_controllers(
    trace: list[TraceRequest], config: ControllerConfig
) -> ControllerEvaluation:
    predictive, predictive_decisions, windows = _run_controller(
        PredictiveController(config), trace, "predictive"
    )
    reactive, reactive_decisions, _ = _run_controller(ReactiveController(config), trace, "reactive")
    return ControllerEvaluation(
        predictive=predictive,
        reactive=reactive,
        predictive_decisions=predictive_decisions,
        reactive_decisions=reactive_decisions,
        windows=windows,
    )
