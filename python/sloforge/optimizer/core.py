from __future__ import annotations

import hashlib
import itertools
import random
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.models.service_curve import CalibratedModels, CandidateModels
from sloforge.profiler.core import BackendCandidate, ProfileBundle
from sloforge.trace import TraceRequest
from sloforge.util import canonical_json, percentile, utc_now

MetricName = Literal[
    "p95_ttft_ms",
    "p99_itl_ms",
    "p95_e2e_ms",
    "goodput_tokens_s",
    "throughput_tokens_s",
    "availability",
    "cost_per_million_tokens",
    "cold_start_p95_ms",
]


class Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: MetricName
    operator: Literal["<=", ">="]
    value: float

    def satisfied(self, metrics: Metrics) -> bool:
        observed = float(getattr(metrics, self.metric))
        return observed <= self.value if self.operator == "<=" else observed >= self.value


class OptimizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraints: list[Constraint]
    objective: MetricName = "cost_per_million_tokens"
    direction: Literal["minimize", "maximize"] = "minimize"
    max_replicas: int = Field(default=4, ge=1, le=128)
    trial_budget: int = Field(default=24, ge=1)
    uncertainty_safety: float = Field(default=1.0, ge=0, le=5)
    seed: int = 0


class CandidateConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_id: str
    backend_candidate_id: str
    replicas: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    max_batched_tokens: int = Field(ge=64)
    chunked_prefill: bool
    routing_policy: Literal["round_robin", "least_outstanding", "earliest_finish", "slo_slack"]
    warm_replicas: int = Field(ge=0)

    @model_validator(mode="after")
    def warm_capacity_fits(self) -> CandidateConfiguration:
        if self.warm_replicas > self.replicas:
            raise ValueError("warm_replicas cannot exceed replicas")
        return self


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p95_ttft_ms: float = Field(ge=0)
    p99_itl_ms: float = Field(ge=0)
    p95_e2e_ms: float = Field(ge=0)
    goodput_tokens_s: float = Field(ge=0)
    throughput_tokens_s: float = Field(ge=0)
    availability: float = Field(ge=0, le=1)
    cost_per_million_tokens: float = Field(ge=0)
    cold_start_p95_ms: float = Field(ge=0)


class CandidateEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configuration: CandidateConfiguration
    predicted: Metrics
    uncertainty: Metrics
    measured: Metrics | None
    feasible: bool
    constraint_margins: dict[str, float]
    rejection_reasons: list[str]
    fidelity: Literal["predicted", "measured"]
    acquisition_score: float


class OptimizerStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    strategy: str
    config_id: str
    fidelity: Literal["predicted", "measured"]
    acquisition_score: float
    incumbent_config_id: str | None
    reason: str


class BaselineOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["exhaustive", "random", "successive_halving", "uncertainty_aware"]
    trials: int
    best_config_id: str | None
    best_objective: float | None


class OptimizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.optimization/v1"] = "sloforge.optimization/v1"
    optimization_id: str
    generated_at: str
    request: OptimizationRequest
    selected: CandidateEvaluation
    pareto_frontier: list[CandidateEvaluation]
    evaluated_candidates: list[CandidateEvaluation]
    optimizer_history: list[OptimizerStep]
    baseline_outcomes: list[BaselineOutcome]
    rejected_candidates: list[CandidateEvaluation]


def parse_slo_expression(expression: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    for raw in expression.split(","):
        part = raw.strip()
        if not part:
            continue
        operator = "<=" if "<=" in part else ">=" if ">=" in part else None
        if operator is None:
            raise ValueError(
                f"invalid SLO constraint {part!r}; expected metric<=value or metric>=value"
            )
        metric, value = (item.strip() for item in part.split(operator, maxsplit=1))
        try:
            constraints.append(
                Constraint.model_validate(
                    {"metric": metric, "operator": operator, "value": float(value)}
                )
            )
        except ValueError as exc:
            raise ValueError(f"invalid SLO constraint {part!r}: {exc}") from exc
    if not constraints:
        raise ValueError("at least one SLO constraint is required")
    return constraints


def _configuration_id(payload: dict[str, object]) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:12]
    return f"cfg-{digest}"


def generate_configurations(
    profile: ProfileBundle, *, max_replicas: int
) -> list[CandidateConfiguration]:
    configurations: list[CandidateConfiguration] = []
    policies: tuple[
        Literal["round_robin", "least_outstanding", "earliest_finish", "slo_slack"], ...
    ] = (
        "round_robin",
        "earliest_finish",
        "slo_slack",
    )
    for candidate_profile in profile.candidates:
        if not candidate_profile.feasible:
            continue
        backend = candidate_profile.candidate
        concurrency_values = sorted(
            {
                1,
                min(2, backend.max_concurrency),
                min(4, backend.max_concurrency),
                backend.max_concurrency,
            }
        )
        for replicas, concurrency, batched_tokens, chunked, routing in itertools.product(
            range(1, max_replicas + 1),
            concurrency_values,
            (512, 2048, 8192),
            (False, True),
            policies,
        ):
            if not chunked and batched_tokens < 2048:
                continue
            payload: dict[str, object] = {
                "backend_candidate_id": backend.candidate_id,
                "replicas": replicas,
                "concurrency": concurrency,
                "max_batched_tokens": batched_tokens,
                "chunked_prefill": chunked,
                "routing_policy": routing,
                "warm_replicas": replicas if backend.startup_ms > 500 else max(1, replicas - 1),
            }
            configurations.append(
                CandidateConfiguration(
                    config_id=_configuration_id(payload),
                    backend_candidate_id=backend.candidate_id,
                    replicas=replicas,
                    concurrency=concurrency,
                    max_batched_tokens=batched_tokens,
                    chunked_prefill=chunked,
                    routing_policy=routing,
                    warm_replicas=(replicas if backend.startup_ms > 500 else max(1, replicas - 1)),
                )
            )
    return configurations


def _candidate_map(profile: ProfileBundle) -> dict[str, BackendCandidate]:
    return {
        item.candidate.candidate_id: item.candidate for item in profile.candidates if item.feasible
    }


def _model_map(models: CalibratedModels) -> dict[str, CandidateModels]:
    return {item.candidate_id: item for item in models.candidates}


def _metric_uncertainty(center: Metrics, prefill_radius: float, decode_radius: float) -> Metrics:
    relative = min(
        0.35, (prefill_radius + decode_radius) / max(center.p95_ttft_ms + center.p99_itl_ms, 1.0)
    )
    return Metrics(
        p95_ttft_ms=max(prefill_radius, center.p95_ttft_ms * relative),
        p99_itl_ms=max(decode_radius, center.p99_itl_ms * relative),
        p95_e2e_ms=center.p95_e2e_ms * relative,
        goodput_tokens_s=center.goodput_tokens_s * relative,
        throughput_tokens_s=center.throughput_tokens_s * relative,
        availability=min(0.25, max(0.001, relative * 0.1)),
        cost_per_million_tokens=center.cost_per_million_tokens * relative,
        cold_start_p95_ms=max(prefill_radius, center.cold_start_p95_ms * relative),
    )


def _predict(
    configuration: CandidateConfiguration,
    *,
    backend: BackendCandidate,
    model: CandidateModels,
    trace: list[TraceRequest],
) -> tuple[Metrics, Metrics]:
    prompts = [float(item.prompt_tokens) for item in trace]
    outputs = [float(item.output_tokens) for item in trace]
    prompt_p95 = percentile(prompts, 0.95)
    output_p95 = percentile(outputs, 0.95)
    mean_output = sum(outputs) / len(outputs)
    prefill, prefill_low, prefill_high = model.prefill.predict(prompt_p95)
    active = min(configuration.concurrency, backend.max_concurrency)
    decode, decode_low, decode_high = model.decode.predict(float(active))
    duration_s = max((trace[-1].arrival_ms - trace[0].arrival_ms) / 1000.0, 0.001)
    arrival_rps = len(trace) / duration_s
    service_e2e_ms = prefill + max(0.0, output_p95 - 1.0) * decode
    request_capacity_rps = configuration.replicas * active * 1000.0 / max(service_e2e_ms, 0.001)
    utilization = min(0.999, arrival_rps / max(request_capacity_rps, 1e-9))
    routing_factor = {
        "round_robin": 1.12,
        "least_outstanding": 1.05,
        "earliest_finish": 0.98,
        "slo_slack": 0.94,
    }[configuration.routing_policy]
    chunk_factor = 0.90 if configuration.chunked_prefill and prompt_p95 > 512 else 1.0
    batching_factor = 0.92 if configuration.max_batched_tokens >= 2048 else 1.0
    queue_factor = 1.0 + utilization**2 / max(2.0 * (1.0 - utilization), 0.01)
    ttft = prefill * routing_factor * chunk_factor * queue_factor
    itl = decode * routing_factor * batching_factor * (1.0 + 0.10 * utilization)
    e2e = ttft + max(0.0, output_p95 - 1.0) * itl
    throughput = min(arrival_rps, request_capacity_rps) * mean_output
    deadline_good = sum(
        request.output_tokens
        for request in trace
        if request.deadline_ms is None
        or ttft + max(0, request.output_tokens - 1) * itl <= request.deadline_ms
    )
    goodput = min(throughput, deadline_good / duration_s)
    availability = max(
        0.0, min(1.0, (1.0 - backend.failure_rate) ** max(1.0 / configuration.replicas, 0.1))
    )
    hourly_cost = backend.hourly_price_usd * configuration.replicas
    cost_per_million = hourly_cost / max(throughput * 3600.0, 1e-9) * 1_000_000.0
    cold_fraction = (
        max(0, configuration.replicas - configuration.warm_replicas) / configuration.replicas
    )
    cold_start = model.startup_p95_ms * (1.0 + cold_fraction)
    metrics = Metrics(
        p95_ttft_ms=ttft,
        p99_itl_ms=itl,
        p95_e2e_ms=e2e,
        goodput_tokens_s=goodput,
        throughput_tokens_s=throughput,
        availability=availability,
        cost_per_million_tokens=cost_per_million,
        cold_start_p95_ms=cold_start,
    )
    uncertainty = _metric_uncertainty(metrics, prefill_high - prefill_low, decode_high - decode_low)
    return metrics, uncertainty


def _measured_metrics(
    configuration: CandidateConfiguration, *, profile: ProfileBundle, predicted: Metrics
) -> Metrics | None:
    # Profile load tests directly observe one replica without compiler-added routing, chunking, or
    # batching changes. Other configurations must remain predictions: scaling these measurements
    # algebraically and calling the result "measured" would corrupt the evidence chain.
    if not (
        configuration.replicas == 1
        and configuration.concurrency == 1
        and configuration.max_batched_tokens == 2048
        and not configuration.chunked_prefill
        and configuration.routing_policy == "round_robin"
    ):
        return None
    candidate = next(
        (
            item
            for item in profile.candidates
            if item.candidate.candidate_id == configuration.backend_candidate_id
        ),
        None,
    )
    if candidate is None or not candidate.feasible or not candidate.summaries:
        return None
    summary = candidate.summaries
    measured_throughput = summary["measured_goodput_tokens_s"]
    measured_cost = (
        candidate.candidate.hourly_price_usd / max(measured_throughput * 3600.0, 1e-9) * 1_000_000.0
    )
    return Metrics(
        p95_ttft_ms=summary["ttft_p95_ms"],
        p99_itl_ms=summary["itl_p99_ms"],
        p95_e2e_ms=summary["e2e_p95_ms"],
        goodput_tokens_s=measured_throughput,
        throughput_tokens_s=measured_throughput,
        availability=summary["availability"],
        cost_per_million_tokens=measured_cost,
        cold_start_p95_ms=summary["startup_p95_ms"],
    )


def _margin(constraint: Constraint, metrics: Metrics, uncertainty: Metrics, safety: float) -> float:
    center = float(getattr(metrics, constraint.metric))
    radius = float(getattr(uncertainty, constraint.metric)) * safety
    conservative = center + radius if constraint.operator == "<=" else center - radius
    return (
        constraint.value - conservative
        if constraint.operator == "<="
        else conservative - constraint.value
    )


def _evaluate(
    configuration: CandidateConfiguration,
    *,
    request: OptimizationRequest,
    backend: BackendCandidate,
    model: CandidateModels,
    profile: ProfileBundle,
    trace: list[TraceRequest],
    allow_profile_measurement: bool,
) -> CandidateEvaluation:
    predicted, uncertainty = _predict(configuration, backend=backend, model=model, trace=trace)
    actual = (
        _measured_metrics(configuration, profile=profile, predicted=predicted)
        if allow_profile_measurement
        else None
    )
    # A Stage-E profile sample observes service time for an isolated request shape; it does not
    # observe the queue induced by the representative arrival process.  Retain that measurement
    # for calibration/provenance, but make deployment decisions with the workload-level model.
    # Otherwise an overloaded one-replica candidate can appear feasible merely because its
    # individual requests were probed serially.
    constraint_metrics = predicted
    margins: dict[str, float] = {
        constraint.metric: _margin(
            constraint, constraint_metrics, uncertainty, request.uncertainty_safety
        )
        for constraint in request.constraints
    }
    reasons = [
        f"{metric} misses its uncertainty-adjusted constraint by {abs(margin):.3f}"
        for metric, margin in margins.items()
        if margin < 0
    ]
    objective = float(getattr(constraint_metrics, request.objective))
    improvement = -objective if request.direction == "minimize" else objective
    uncertainty_value = float(getattr(uncertainty, request.objective))
    normalized_violation = sum(
        max(0.0, -margin) / max(abs(constraint.value), 1.0)
        for constraint in request.constraints
        if (margin := margins[constraint.metric]) < 0
    )
    # Constrained acquisition is feasibility-first. A small objective value must never outweigh
    # an arbitrarily large SLO miss (the previous cost-scaled penalty did exactly that).
    acquisition = (
        improvement
        + uncertainty_value * 0.25
        - normalized_violation * (abs(improvement) + 1.0) * 1_000.0
    )
    return CandidateEvaluation(
        configuration=configuration,
        predicted=predicted,
        uncertainty=uncertainty,
        measured=actual,
        feasible=not reasons,
        constraint_margins=margins,
        rejection_reasons=reasons,
        fidelity="measured" if actual is not None else "predicted",
        acquisition_score=acquisition,
    )


def evaluate_configuration(
    *,
    configuration: CandidateConfiguration,
    profile: ProfileBundle,
    models: CalibratedModels,
    trace: list[TraceRequest],
    request: OptimizationRequest,
) -> CandidateEvaluation:
    """Evaluate one fixed serving configuration on a workload without relabeling predictions.

    This is used for reproducible serving-baseline comparisons. Direct profile measurements are
    intentionally disabled because they were collected on the original profiling trace.
    """
    backends = _candidate_map(profile)
    model_by_id = _model_map(models)
    return _evaluate(
        configuration,
        request=request,
        backend=backends[configuration.backend_candidate_id],
        model=model_by_id[configuration.backend_candidate_id],
        profile=profile,
        trace=trace,
        allow_profile_measurement=False,
    )


def _dominates(left: CandidateEvaluation, right: CandidateEvaluation) -> bool:
    # Pareto comparison is over the representative-workload prediction. Direct profile samples
    # are calibration evidence and cannot represent compiler-added topology or queueing.
    left_metrics = left.predicted
    right_metrics = right.predicted
    minimize = (
        "p95_ttft_ms",
        "p99_itl_ms",
        "p95_e2e_ms",
        "cost_per_million_tokens",
        "cold_start_p95_ms",
    )
    maximize = ("goodput_tokens_s", "throughput_tokens_s", "availability")
    no_worse = all(
        getattr(left_metrics, metric) <= getattr(right_metrics, metric) for metric in minimize
    )
    no_worse = no_worse and all(
        getattr(left_metrics, metric) >= getattr(right_metrics, metric) for metric in maximize
    )
    strictly = any(
        getattr(left_metrics, metric) < getattr(right_metrics, metric) for metric in minimize
    )
    strictly = strictly or any(
        getattr(left_metrics, metric) > getattr(right_metrics, metric) for metric in maximize
    )
    return no_worse and strictly


def pareto_frontier(evaluations: Iterable[CandidateEvaluation]) -> list[CandidateEvaluation]:
    values = list(evaluations)
    return [
        candidate
        for candidate in values
        if not any(_dominates(other, candidate) for other in values if other is not candidate)
    ]


def _objective_value(evaluation: CandidateEvaluation, request: OptimizationRequest) -> float:
    return float(getattr(evaluation.predicted, request.objective))


def _best(
    evaluations: list[CandidateEvaluation], request: OptimizationRequest
) -> CandidateEvaluation | None:
    feasible = [item for item in evaluations if item.feasible]
    if not feasible:
        return None
    return (
        min(feasible, key=lambda item: _objective_value(item, request))
        if request.direction == "minimize"
        else max(feasible, key=lambda item: _objective_value(item, request))
    )


def _baseline_outcomes(
    evaluations: list[CandidateEvaluation], request: OptimizationRequest, rng: random.Random
) -> list[BaselineOutcome]:
    exhaustive_best = _best(evaluations, request)
    random_order = evaluations.copy()
    rng.shuffle(random_order)
    random_subset = random_order[: request.trial_budget]
    random_best = _best(random_subset, request)
    predicted_rank = sorted(evaluations, key=lambda item: item.acquisition_score, reverse=True)
    pool = predicted_rank
    trials = 0
    while len(pool) > 1 and trials < request.trial_budget:
        take = min(len(pool), max(2, request.trial_budget - trials))
        round_values = pool[:take]
        trials += take
        pool = sorted(
            round_values,
            key=lambda item: _objective_value(item, request),
            reverse=request.direction == "maximize",
        )[: max(1, take // 2)]
    halving_best = _best(pool, request)
    primary_subset = predicted_rank[: request.trial_budget]
    primary_best = _best(primary_subset, request)
    return [
        BaselineOutcome(
            strategy="exhaustive",
            trials=len(evaluations),
            best_config_id=exhaustive_best.configuration.config_id if exhaustive_best else None,
            best_objective=_objective_value(exhaustive_best, request) if exhaustive_best else None,
        ),
        BaselineOutcome(
            strategy="random",
            trials=len(random_subset),
            best_config_id=random_best.configuration.config_id if random_best else None,
            best_objective=_objective_value(random_best, request) if random_best else None,
        ),
        BaselineOutcome(
            strategy="successive_halving",
            trials=trials,
            best_config_id=halving_best.configuration.config_id if halving_best else None,
            best_objective=_objective_value(halving_best, request) if halving_best else None,
        ),
        BaselineOutcome(
            strategy="uncertainty_aware",
            trials=len(primary_subset),
            best_config_id=primary_best.configuration.config_id if primary_best else None,
            best_objective=_objective_value(primary_best, request) if primary_best else None,
        ),
    ]


def optimize(
    *,
    profile: ProfileBundle,
    models: CalibratedModels,
    trace: list[TraceRequest],
    request: OptimizationRequest,
) -> OptimizationResult:
    configurations = generate_configurations(profile, max_replicas=request.max_replicas)
    if not configurations:
        raise RuntimeError("static feasibility pruning rejected every configuration")
    backends = _candidate_map(profile)
    model_by_id = _model_map(models)
    rng = random.Random(request.seed)
    predicted = [
        _evaluate(
            configuration,
            request=request,
            backend=backends[configuration.backend_candidate_id],
            model=model_by_id[configuration.backend_candidate_id],
            profile=profile,
            trace=trace,
            allow_profile_measurement=True,
        )
        for configuration in configurations
    ]
    acquisition_order = sorted(predicted, key=lambda item: item.acquisition_score, reverse=True)
    evaluations = [
        _evaluate(
            item.configuration,
            request=request,
            backend=backends[item.configuration.backend_candidate_id],
            model=model_by_id[item.configuration.backend_candidate_id],
            profile=profile,
            trace=trace,
            allow_profile_measurement=True,
        )
        for item in predicted
    ]
    promoted = {item.configuration.config_id for item in acquisition_order[: request.trial_budget]}
    selection_pool = [
        item for item in evaluations if item.feasible and item.configuration.config_id in promoted
    ]
    frontier = pareto_frontier(selection_pool)
    selected = _best(frontier, request)
    if selected is None:
        closest = max(evaluations, key=lambda item: sum(item.constraint_margins.values()))
        details = "; ".join(closest.rejection_reasons)
        raise RuntimeError(
            f"no candidate satisfies the requested SLOs; closest was {closest.configuration.config_id}: {details}"
        )
    history: list[OptimizerStep] = []
    incumbent: CandidateEvaluation | None = None
    evaluated_by_id = {item.configuration.config_id: item for item in evaluations}
    for step, proposal in enumerate(acquisition_order[: request.trial_budget], start=1):
        evaluated = evaluated_by_id[proposal.configuration.config_id]
        candidate_best = _best(
            [item for item in (incumbent, evaluated) if item is not None], request
        )
        incumbent = candidate_best or incumbent
        history.append(
            OptimizerStep(
                step=step,
                strategy="uncertainty_aware_pareto",
                config_id=evaluated.configuration.config_id,
                fidelity=evaluated.fidelity,
                acquisition_score=evaluated.acquisition_score,
                incumbent_config_id=incumbent.configuration.config_id if incumbent else None,
                reason="high uncertainty-adjusted Pareto improvement under remaining trial budget",
            )
        )
    result_payload = {
        "profile_id": profile.profile_id,
        "seed": request.seed,
        "selected": selected.configuration.config_id,
        "constraints": [item.model_dump() for item in request.constraints],
    }
    optimization_id = (
        "opt-" + hashlib.sha256(canonical_json(result_payload).encode()).hexdigest()[:16]
    )
    return OptimizationResult(
        optimization_id=optimization_id,
        generated_at=utc_now(),
        request=request,
        selected=selected,
        pareto_frontier=frontier,
        evaluated_candidates=evaluations,
        optimizer_history=history,
        baseline_outcomes=_baseline_outcomes(evaluations, request, rng),
        rejected_candidates=[item for item in evaluations if not item.feasible],
    )
