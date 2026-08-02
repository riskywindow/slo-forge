"""Counterfactual evaluation through the calibrated Fabric simulator.

The Python side owns hypothesis orchestration while the Rust simulator remains
the source of truth for physical-resource scheduling.  Every repair is an
explicit transformation of the exact degraded simulation request.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import Field, TypeAdapter, model_validator

from sloforge.util import canonical_json, sha256_bytes

from .models import (
    BottleneckKind,
    CausalHypothesis,
    CounterfactualEstimate,
    DiagnosisRecord,
    NonEmpty,
    StrictModel,
)

MAX_PROTOCOL_BYTES = 64 * 1024 * 1024


class RemoveFault(StrictModel):
    type: Literal["remove_fault"] = "remove_fault"
    fault_id: NonEmpty


class ScaleResourceCurve(StrictModel):
    type: Literal["scale_resource_curve"] = "scale_resource_curve"
    resource_id: NonEmpty
    latency_multiplier: float = Field(gt=0.0)
    bandwidth_multiplier: float = Field(gt=0.0)


class ScaleRank(StrictModel):
    type: Literal["scale_rank"] = "scale_rank"
    rank_id: NonEmpty
    duration_multiplier: float = Field(gt=0.0)


class ReplaceResource(StrictModel):
    type: Literal["replace_resource"] = "replace_resource"
    from_resource_id: NonEmpty
    to_resource_id: NonEmpty


CounterfactualModifier: TypeAlias = Annotated[
    RemoveFault | ScaleResourceCurve | ScaleRank | ReplaceResource,
    Field(discriminator="type"),
]
_MODIFIER_ADAPTER: TypeAdapter[CounterfactualModifier] = TypeAdapter(CounterfactualModifier)


class CounterfactualScenario(StrictModel):
    scenario_id: NonEmpty
    hypothesis_id: NonEmpty
    hypothesis_kind: BottleneckKind
    rationale: NonEmpty
    modifications: tuple[CounterfactualModifier, ...]

    @model_validator(mode="after")
    def nonempty_unique_modifications(self) -> CounterfactualScenario:
        if not self.modifications:
            raise ValueError("a counterfactual requires at least one modification")
        canonical = [canonical_json(item.model_dump(mode="json")) for item in self.modifications]
        if len(canonical) != len(set(canonical)):
            raise ValueError("counterfactual modifications must be unique")
        return self


class SimulationObservation(StrictModel):
    makespan_us: float = Field(ge=0.0)
    predicted_lower_us: float = Field(ge=0.0)
    predicted_upper_us: float = Field(ge=0.0)
    operation_count: int = Field(ge=0)
    processed_events: int = Field(ge=0)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def ordered_interval(self) -> SimulationObservation:
        if self.predicted_lower_us > self.makespan_us:
            raise ValueError("prediction lower bound exceeds makespan")
        if self.makespan_us > self.predicted_upper_us:
            raise ValueError("makespan exceeds prediction upper bound")
        return self


class ScenarioEvaluation(StrictModel):
    scenario: CounterfactualScenario
    status: Literal["supported", "contradicted", "inconclusive", "simulation_failed"]
    expected_improvement_ms: float | None
    lower_improvement_ms: float | None
    upper_improvement_ms: float | None
    healthy_reference_residual_ms: float | None
    confidence: float = Field(ge=0.0, le=1.0)
    observation: SimulationObservation | None
    rejected_reason: str | None

    @model_validator(mode="after")
    def complete_result(self) -> ScenarioEvaluation:
        values = (
            self.expected_improvement_ms,
            self.lower_improvement_ms,
            self.upper_improvement_ms,
            self.healthy_reference_residual_ms,
        )
        if self.status == "simulation_failed":
            if self.observation is not None or any(item is not None for item in values):
                raise ValueError("failed simulations cannot contain numeric outcomes")
            if self.rejected_reason is None:
                raise ValueError("failed simulations require a rejection reason")
        elif self.observation is None or any(item is None for item in values):
            raise ValueError("completed simulations require all numeric outcomes")
        else:
            assert self.expected_improvement_ms is not None
            assert self.lower_improvement_ms is not None
            assert self.upper_improvement_ms is not None
            if not (
                self.lower_improvement_ms
                <= self.expected_improvement_ms
                <= self.upper_improvement_ms
            ):
                raise ValueError("counterfactual improvement interval is not ordered")
            expected_status = (
                "supported"
                if self.lower_improvement_ms > 0.0
                else "contradicted"
                if self.upper_improvement_ms <= 0.0
                else "inconclusive"
            )
            if self.status != expected_status:
                raise ValueError("counterfactual status disagrees with its improvement interval")
        return self


class CounterfactualReplay(StrictModel):
    schema_version: Literal["sloforge.autopsy.counterfactual/v1"] = (
        "sloforge.autopsy.counterfactual/v1"
    )
    diagnosis_id: NonEmpty
    simulation_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    degraded: SimulationObservation
    healthy_reference_us: float = Field(ge=0.0)
    evaluations: tuple[ScenarioEvaluation, ...]
    selected_scenario_id: str | None
    rejected_scenario_ids: tuple[str, ...]

    @model_validator(mode="after")
    def consistent_selection(self) -> CounterfactualReplay:
        identifiers = [item.scenario.scenario_id for item in self.evaluations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scenario identifiers must be unique")
        if self.selected_scenario_id is not None:
            matches = [
                item
                for item in self.evaluations
                if item.scenario.scenario_id == self.selected_scenario_id
            ]
            if not matches or matches[0].status != "supported":
                raise ValueError("selected scenario must be a supported evaluation")
        expected_rejections = tuple(
            item.scenario.scenario_id
            for item in self.evaluations
            if item.scenario.scenario_id != self.selected_scenario_id
        )
        if self.rejected_scenario_ids != expected_rejections:
            raise ValueError("rejected scenario list disagrees with evaluations")
        return self


class SimulatorRunner(Protocol):
    def __call__(self, request: Mapping[str, object]) -> SimulationObservation: ...


def _parse_observation(payload: bytes) -> SimulationObservation:
    if len(payload) > MAX_PROTOCOL_BYTES:
        raise RuntimeError("Fabric simulator output exceeds 64 MiB safety limit")
    try:
        decoded = json.loads(payload)
        metrics = decoded["metrics"]
        observation = SimulationObservation(
            makespan_us=float(metrics["makespan_us"]),
            predicted_lower_us=float(metrics["predicted_lower_us"]),
            predicted_upper_us=float(metrics["predicted_upper_us"]),
            operation_count=int(metrics["operation_count"]),
            processed_events=int(metrics["processed_events"]),
            output_sha256=sha256_bytes(payload),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("Fabric simulator returned an invalid response") from error
    return observation


def run_fabric_simulator(
    request: Mapping[str, object],
    *,
    repository_root: Path,
    timeout_s: float = 120.0,
    executable: Path | None = None,
) -> SimulationObservation:
    """Execute one bounded JSON request using the Fabric simulator binary."""
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    encoded = canonical_json(request).encode()
    if len(encoded) > MAX_PROTOCOL_BYTES:
        raise ValueError("Fabric simulator input exceeds 64 MiB safety limit")
    discovered = shutil.which("sloforge-fabric-sim") if executable is None else None
    command = (
        [str(executable)]
        if executable is not None
        else [discovered]
        if discovered is not None
        else ["cargo", "run", "-q", "-p", "sloforge-fabric-sim", "--"]
    )
    command.extend(["simulate", "--compact"])
    completed = subprocess.run(
        command,
        cwd=repository_root,
        input=encoded,
        check=False,
        capture_output=True,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        stderr = completed.stderr[:16_384].decode(errors="replace")
        raise RuntimeError(f"Fabric simulator failed ({completed.returncode}): {stderr}")
    return _parse_observation(completed.stdout)


def _request_with_modifications(
    simulation_request: Mapping[str, object],
    modifications: Sequence[CounterfactualModifier],
) -> dict[str, object]:
    transformed = dict(simulation_request)
    existing = transformed.get("counterfactuals", [])
    if not isinstance(existing, list):
        raise ValueError("simulation counterfactuals field must be a list")
    transformed["counterfactuals"] = [
        *existing,
        *(item.model_dump(mode="json") for item in modifications),
    ]
    return transformed


def _hypothesis_confidence(diagnosis: DiagnosisRecord, scenario: CounterfactualScenario) -> float:
    for hypothesis in diagnosis.hypotheses:
        if hypothesis.hypothesis_id == scenario.hypothesis_id:
            if hypothesis.kind is not scenario.hypothesis_kind:
                raise ValueError("scenario hypothesis kind disagrees with the diagnosis")
            return hypothesis.confidence
    raise ValueError(f"unknown diagnosis hypothesis {scenario.hypothesis_id!r}")


def replay_counterfactuals(
    diagnosis: DiagnosisRecord,
    *,
    simulation_request: Mapping[str, object],
    scenarios: Sequence[CounterfactualScenario],
    healthy_reference_us: float,
    runner: SimulatorRunner,
) -> CounterfactualReplay:
    """Evaluate explicit causal repairs and rank them by restored healthy gap."""
    if healthy_reference_us < 0.0 or not math.isfinite(healthy_reference_us):
        raise ValueError("healthy_reference_us must be finite and non-negative")
    if not scenarios:
        raise ValueError("at least one counterfactual scenario is required")
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("counterfactual scenario identifiers must be unique")
    degraded = runner(simulation_request)
    if degraded.makespan_us <= healthy_reference_us:
        raise ValueError("degraded simulation must be slower than the healthy reference")
    evaluations: list[ScenarioEvaluation] = []
    for scenario in scenarios:
        hypothesis_confidence = _hypothesis_confidence(diagnosis, scenario)
        try:
            observation = runner(
                _request_with_modifications(simulation_request, scenario.modifications)
            )
        except RuntimeError as error:
            evaluations.append(
                ScenarioEvaluation(
                    scenario=scenario,
                    status="simulation_failed",
                    expected_improvement_ms=None,
                    lower_improvement_ms=None,
                    upper_improvement_ms=None,
                    healthy_reference_residual_ms=None,
                    confidence=0.0,
                    observation=None,
                    rejected_reason=str(error),
                )
            )
            continue
        improvement = (degraded.makespan_us - observation.makespan_us) / 1_000.0
        lower = (degraded.predicted_lower_us - observation.predicted_upper_us) / 1_000.0
        upper = (degraded.predicted_upper_us - observation.predicted_lower_us) / 1_000.0
        residual = (observation.makespan_us - healthy_reference_us) / 1_000.0
        uncertainty_width = max(0.0, upper - lower)
        scale = max(abs(improvement), 0.001)
        uncertainty_factor = 1.0 / (1.0 + uncertainty_width / scale)
        direction_factor = 1.0 if lower > 0.0 else 0.65 if improvement > 0.0 else 0.25
        confidence = min(0.99, hypothesis_confidence * uncertainty_factor * direction_factor)
        if lower > 0.0:
            status: Literal["supported", "contradicted", "inconclusive"] = "supported"
            rejected_reason = None
        elif upper <= 0.0:
            status = "contradicted"
            rejected_reason = "counterfactual did not improve predicted makespan"
        else:
            status = "inconclusive"
            rejected_reason = "counterfactual prediction interval crosses zero improvement"
        evaluations.append(
            ScenarioEvaluation(
                scenario=scenario,
                status=status,
                expected_improvement_ms=improvement,
                lower_improvement_ms=lower,
                upper_improvement_ms=upper,
                healthy_reference_residual_ms=residual,
                confidence=confidence,
                observation=observation,
                rejected_reason=rejected_reason,
            )
        )
    supported = [item for item in evaluations if item.status == "supported"]
    selected = min(
        supported,
        key=lambda item: (
            abs(item.healthy_reference_residual_ms or 0.0),
            -(item.expected_improvement_ms or 0.0),
            item.scenario.scenario_id,
        ),
        default=None,
    )
    evaluations.sort(key=lambda item: item.scenario.scenario_id)
    selected_id = selected.scenario.scenario_id if selected is not None else None
    return CounterfactualReplay(
        diagnosis_id=diagnosis.diagnosis_id,
        simulation_input_sha256=sha256_bytes(canonical_json(simulation_request).encode()),
        degraded=degraded,
        healthy_reference_us=healthy_reference_us,
        evaluations=tuple(evaluations),
        selected_scenario_id=selected_id,
        rejected_scenario_ids=tuple(
            item.scenario.scenario_id
            for item in evaluations
            if item.scenario.scenario_id != selected_id
        ),
    )


def attach_counterfactuals(
    diagnosis: DiagnosisRecord, replay: CounterfactualReplay
) -> DiagnosisRecord:
    """Return a validated diagnosis carrying the best result per hypothesis."""
    if diagnosis.diagnosis_id != replay.diagnosis_id:
        raise ValueError("counterfactual replay does not reference this diagnosis")
    best: dict[str, ScenarioEvaluation] = {}
    for evaluation in replay.evaluations:
        if evaluation.observation is None or evaluation.expected_improvement_ms is None:
            continue
        identifier = evaluation.scenario.hypothesis_id
        previous = best.get(identifier)
        status_rank = {"supported": 2, "inconclusive": 1, "contradicted": 0}
        candidate_key = (
            status_rank.get(evaluation.status, -1),
            evaluation.confidence,
            -abs(evaluation.healthy_reference_residual_ms or 0.0),
            evaluation.expected_improvement_ms,
            evaluation.scenario.scenario_id,
        )
        previous_key = (
            (
                status_rank.get(previous.status, -1),
                previous.confidence,
                -abs(previous.healthy_reference_residual_ms or 0.0),
                previous.expected_improvement_ms or -math.inf,
                previous.scenario.scenario_id,
            )
            if previous is not None
            else None
        )
        if previous_key is None or candidate_key > previous_key:
            best[identifier] = evaluation
    updated: list[CausalHypothesis] = []
    for hypothesis in diagnosis.hypotheses:
        matched_evaluation = best.get(hypothesis.hypothesis_id)
        if matched_evaluation is None:
            updated.append(hypothesis)
            continue
        assert matched_evaluation.expected_improvement_ms is not None
        assert matched_evaluation.lower_improvement_ms is not None
        assert matched_evaluation.upper_improvement_ms is not None
        assert matched_evaluation.healthy_reference_residual_ms is not None
        estimate = CounterfactualEstimate(
            scenario_id=matched_evaluation.scenario.scenario_id,
            expected_improvement_ms=matched_evaluation.expected_improvement_ms,
            lower_improvement_ms=matched_evaluation.lower_improvement_ms,
            upper_improvement_ms=matched_evaluation.upper_improvement_ms,
            healthy_gap_remaining_ms=matched_evaluation.healthy_reference_residual_ms,
            simulation_artifact=None,
        )
        updated.append(
            CausalHypothesis.model_validate(
                {
                    **hypothesis.model_dump(mode="python"),
                    "counterfactual": estimate,
                    "confidence": min(hypothesis.confidence, matched_evaluation.confidence),
                    "rejected_reason": matched_evaluation.rejected_reason,
                }
            )
        )
    ordered = tuple(
        sorted(
            updated,
            key=lambda item: (
                item.rejected_reason is not None,
                -item.confidence,
                item.kind.value,
            ),
        )
    )
    top_three = tuple(item.kind for item in ordered[:3])
    return DiagnosisRecord.model_validate(
        {
            **diagnosis.model_dump(mode="python"),
            "top_hypothesis": ordered[0].kind,
            "top_three": top_three,
            "hypotheses": ordered,
            "confidence": ordered[0].confidence,
        }
    )


def parse_modifier(value: object) -> CounterfactualModifier:
    """Validate a modifier loaded from YAML or JSON without accepting extensions."""
    return _MODIFIER_ADAPTER.validate_python(value, strict=True)


def bind_subprocess_runner(
    *, repository_root: Path, timeout_s: float = 120.0, executable: Path | None = None
) -> SimulatorRunner:
    """Create the default runner while keeping replay directly unit-testable."""

    def execute(request: Mapping[str, object]) -> SimulationObservation:
        return run_fabric_simulator(
            request,
            repository_root=repository_root,
            timeout_s=timeout_s,
            executable=executable,
        )

    return execute
