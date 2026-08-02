from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.util import utc_now

FaultType = Literal[
    "backend_crash",
    "backend_slowdown",
    "startup_slowdown",
    "request_errors",
    "queue_saturation",
    "capacity_loss",
    "network_jitter",
    "simulated_oom",
]

DiagnosisLabel = Literal[
    "arrival_overload",
    "gateway_queueing",
    "insufficient_warm_capacity",
    "cold_start_dominance",
    "prefill_dominance",
    "decode_dominance",
    "unhealthy_backend",
    "configuration_infeasibility",
]


class FaultEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str
    fault_type: FaultType
    start_ms: float = Field(ge=0)
    duration_ms: float = Field(gt=0)
    backend_id: str | None = None
    magnitude: float = Field(default=2.0, gt=0)
    probability: float = Field(default=1.0, ge=0, le=1)


class FaultScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.chaos/v1"] = "sloforge.chaos/v1"
    name: str
    seed: int = 0
    events: list[FaultEvent]

    @model_validator(mode="after")
    def unique_fault_ids(self) -> FaultScenario:
        ids = [event.fault_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("fault IDs must be unique")
        return self


class CounterSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gateway_queue_ms: float = Field(ge=0)
    backend_queue_ms: float = Field(ge=0)
    startup_ms: float = Field(ge=0)
    prefill_ms: float = Field(ge=0)
    decode_ms: float = Field(ge=0)
    network_ms: float = Field(ge=0)
    arrival_rate_rps: float = Field(ge=0)
    admitted_capacity_rps: float = Field(ge=0)
    backend_error_rate: float = Field(ge=0, le=1)
    healthy_backend_fraction: float = Field(ge=0, le=1)
    memory_rejections: int = Field(ge=0)
    warm_replica_fraction: float = Field(ge=0, le=1)


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str
    expected_label: DiagnosisLabel
    predicted_label: DiagnosisLabel
    confidence: float = Field(ge=0, le=1)
    evidence: list[str]
    counterfactual_improvement_ms: float = Field(ge=0)
    correct: bool
    diagnosis_latency_ms: float = Field(ge=0)


class FaultExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: FaultEvent
    applied: bool
    counters_before: CounterSnapshot
    counters_during: CounterSnapshot
    diagnosis: Diagnosis


class FaultScenarioResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.chaos-result/v1"] = "sloforge.chaos-result/v1"
    scenario_name: str
    executed_at: str
    seed: int
    executions: list[FaultExecution]
    diagnosis_accuracy: float
    false_positive_rate: float
    false_positive_count: int
    negative_window_count: int
    confusion_matrix: dict[str, dict[str, int]]
    mean_diagnosis_latency_ms: float


EXPECTED: dict[FaultType, DiagnosisLabel] = {
    "backend_crash": "unhealthy_backend",
    "backend_slowdown": "decode_dominance",
    "startup_slowdown": "cold_start_dominance",
    "request_errors": "unhealthy_backend",
    "queue_saturation": "gateway_queueing",
    "capacity_loss": "insufficient_warm_capacity",
    "network_jitter": "gateway_queueing",
    "simulated_oom": "configuration_infeasibility",
}


def load_scenario(path: Path) -> FaultScenario:
    max_bytes = 4 * 1024 * 1024
    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"{path}: fault scenario exceeds {max_bytes} byte safety limit")
    document = yaml.safe_load(payload.decode("utf-8"))
    return FaultScenario.model_validate(document)


def _apply(base: CounterSnapshot, event: FaultEvent) -> CounterSnapshot:
    values = base.model_dump()
    magnitude = event.magnitude
    if event.fault_type == "backend_crash":
        values["healthy_backend_fraction"] = max(
            0.0, base.healthy_backend_fraction - min(0.9, magnitude / 4)
        )
        values["backend_error_rate"] = min(1.0, 0.25 * magnitude)
        values["backend_queue_ms"] *= magnitude
    elif event.fault_type == "backend_slowdown":
        values["decode_ms"] *= magnitude
        values["backend_queue_ms"] *= max(1.0, magnitude * 0.7)
    elif event.fault_type == "startup_slowdown":
        values["startup_ms"] *= magnitude
        values["warm_replica_fraction"] = max(0.0, 1.0 - magnitude / 5)
    elif event.fault_type == "request_errors":
        values["backend_error_rate"] = min(1.0, base.backend_error_rate + 0.18 * magnitude)
    elif event.fault_type == "queue_saturation":
        values["gateway_queue_ms"] *= magnitude * 20
        values["arrival_rate_rps"] = base.admitted_capacity_rps * max(1.2, magnitude)
    elif event.fault_type == "capacity_loss":
        values["admitted_capacity_rps"] /= magnitude
        values["warm_replica_fraction"] /= magnitude
        values["backend_queue_ms"] *= magnitude * 4
    elif event.fault_type == "network_jitter":
        values["network_ms"] *= magnitude * 25
        values["gateway_queue_ms"] += values["network_ms"] * 0.3
    elif event.fault_type == "simulated_oom":
        values["memory_rejections"] = max(1, int(magnitude * 3))
        values["backend_error_rate"] = min(1.0, 0.1 * magnitude)
    return CounterSnapshot.model_validate(values)


def classify_bottleneck(
    counters: CounterSnapshot,
) -> tuple[DiagnosisLabel, float, list[str], float]:
    scores: dict[DiagnosisLabel, tuple[float, str, float]] = {
        "configuration_infeasibility": (
            min(1.0, counters.memory_rejections / 3),
            f"{counters.memory_rejections} memory admission rejections",
            counters.memory_rejections * 35.0,
        ),
        "unhealthy_backend": (
            max(counters.backend_error_rate, 1.0 - counters.healthy_backend_fraction),
            f"error rate {counters.backend_error_rate:.2f}, healthy fraction {counters.healthy_backend_fraction:.2f}",
            counters.backend_queue_ms,
        ),
        "arrival_overload": (
            min(1.0, counters.arrival_rate_rps / max(counters.admitted_capacity_rps, 1e-9) - 0.8),
            f"arrival/capacity ratio {counters.arrival_rate_rps / max(counters.admitted_capacity_rps, 1e-9):.2f}",
            counters.gateway_queue_ms + counters.backend_queue_ms,
        ),
        "gateway_queueing": (
            min(1.0, (counters.gateway_queue_ms + counters.network_ms * 0.25) / 90.0),
            f"gateway queue {counters.gateway_queue_ms:.1f}ms and network {counters.network_ms:.1f}ms",
            counters.gateway_queue_ms + counters.network_ms,
        ),
        "insufficient_warm_capacity": (
            min(1.0, (1.0 - counters.warm_replica_fraction) + counters.backend_queue_ms / 180.0),
            f"warm fraction {counters.warm_replica_fraction:.2f} with backend queue {counters.backend_queue_ms:.1f}ms",
            counters.backend_queue_ms,
        ),
        "cold_start_dominance": (
            min(1.0, counters.startup_ms / max(counters.prefill_ms + counters.decode_ms, 1.0)),
            f"startup {counters.startup_ms:.1f}ms dominates service",
            counters.startup_ms,
        ),
        "prefill_dominance": (
            min(1.0, counters.prefill_ms / max(counters.prefill_ms + counters.decode_ms, 1.0)),
            f"prefill contribution {counters.prefill_ms:.1f}ms",
            counters.prefill_ms,
        ),
        "decode_dominance": (
            min(1.0, counters.decode_ms / max(counters.prefill_ms + counters.decode_ms, 1.0)),
            f"decode contribution {counters.decode_ms:.1f}ms",
            counters.decode_ms,
        ),
    }
    # Guarded rule ordering prevents generic overload from hiding known health and feasibility signals.
    if counters.memory_rejections > 0:
        winner: DiagnosisLabel = "configuration_infeasibility"
    elif counters.backend_error_rate >= 0.15 or counters.healthy_backend_fraction < 0.8:
        winner = "unhealthy_backend"
    elif counters.startup_ms > 2.0 * max(counters.prefill_ms, counters.decode_ms):
        winner = "cold_start_dominance"
    elif counters.gateway_queue_ms + counters.network_ms > max(
        100.0, counters.backend_queue_ms * 1.5
    ):
        winner = "gateway_queueing"
    elif counters.warm_replica_fraction < 0.65 and counters.backend_queue_ms > 20:
        winner = "insufficient_warm_capacity"
    elif counters.decode_ms > counters.prefill_ms * 1.4:
        winner = "decode_dominance"
    elif counters.prefill_ms > counters.decode_ms * 1.4:
        winner = "prefill_dominance"
    else:
        winner = max(scores, key=lambda key: scores[key][0])
    score, evidence, improvement = scores[winner]
    return winner, max(0.5, min(0.99, score)), [evidence], improvement


def execute_scenario(scenario: FaultScenario) -> FaultScenarioResult:
    rng = random.Random(scenario.seed)
    base = CounterSnapshot(
        gateway_queue_ms=4.0,
        backend_queue_ms=7.0,
        startup_ms=45.0,
        prefill_ms=38.0,
        decode_ms=52.0,
        network_ms=2.0,
        arrival_rate_rps=8.0,
        admitted_capacity_rps=14.0,
        backend_error_rate=0.005,
        healthy_backend_fraction=1.0,
        memory_rejections=0,
        warm_replica_fraction=1.0,
    )
    executions: list[FaultExecution] = []
    for event in scenario.events:
        applied = rng.random() <= event.probability
        during = _apply(base, event) if applied else base.model_copy(deep=True)
        diagnosis_started = time.perf_counter_ns()
        predicted, confidence, evidence, improvement = classify_bottleneck(during)
        latency = (time.perf_counter_ns() - diagnosis_started) / 1e6
        expected = EXPECTED[event.fault_type]
        diagnosis = Diagnosis(
            fault_id=event.fault_id,
            expected_label=expected,
            predicted_label=predicted,
            confidence=confidence,
            evidence=evidence,
            counterfactual_improvement_ms=improvement,
            correct=predicted == expected if applied else True,
            diagnosis_latency_ms=latency,
        )
        executions.append(
            FaultExecution(
                event=event,
                applied=applied,
                counters_before=base,
                counters_during=during,
                diagnosis=diagnosis,
            )
        )
    applied_diagnoses = [item.diagnosis for item in executions if item.applied]
    accuracy = sum(item.correct for item in applied_diagnoses) / max(1, len(applied_diagnoses))
    # Negative control windows are evaluated independently from fault probability. The classifier
    # may still describe a dominant normal-path component, but it becomes an alert only above the
    # configured confidence threshold.
    negative_windows = [
        base.model_copy(
            update={
                "gateway_queue_ms": base.gateway_queue_ms * rng.uniform(0.85, 1.15),
                "backend_queue_ms": base.backend_queue_ms * rng.uniform(0.85, 1.15),
                "prefill_ms": base.prefill_ms * rng.uniform(0.95, 1.05),
                "decode_ms": base.decode_ms * rng.uniform(0.95, 1.05),
            }
        )
        for _ in range(max(8, len(scenario.events)))
    ]
    false_positive_count = sum(
        classify_bottleneck(window)[1] >= 0.75 for window in negative_windows
    )
    false_positive_rate = false_positive_count / len(negative_windows)
    confusion: dict[str, dict[str, int]] = {}
    for item in executions:
        if not item.applied:
            continue
        expected = item.diagnosis.expected_label
        predicted = item.diagnosis.predicted_label
        confusion.setdefault(expected, {})[predicted] = (
            confusion.setdefault(expected, {}).get(predicted, 0) + 1
        )
    mean_latency = sum(item.diagnosis_latency_ms for item in applied_diagnoses) / max(
        1, len(applied_diagnoses)
    )
    return FaultScenarioResult(
        scenario_name=scenario.name,
        executed_at=utc_now(),
        seed=scenario.seed,
        executions=executions,
        diagnosis_accuracy=accuracy,
        false_positive_rate=false_positive_rate,
        false_positive_count=false_positive_count,
        negative_window_count=len(negative_windows),
        confusion_matrix=confusion,
        mean_diagnosis_latency_ms=mean_latency,
    )
