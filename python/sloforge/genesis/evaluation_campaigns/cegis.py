"""Artifact-backed multi-seed H3 counterexample-learning campaign.

This campaign compares four bounded cancellation-verification strategies against
the same executable restricted-policy candidates.  The ground truth is derived
from the repository's explicit-state checker and bounded property enumerator;
it is not a performance benchmark and carries no hardware claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.genesis.ir import (
    Counterexample,
    RequestEventCase,
    canonical_json,
    load_counterexample,
)
from sloforge.genesis.search import CandidateDesign
from sloforge.genesis.synthesis import (
    CancellationPolicyVerifier,
    ConstraintStore,
    bounded_candidate_modelcheck_document,
    bounded_candidate_policy_property_document,
    cancellation_fixture_candidates,
    run_cancellation_cegis,
)
from sloforge.genesis.synthesis.models import (
    CegisRunResult,
    ConstraintDocument,
    ProtocolWitness,
    VerificationOutcome,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Rate = Annotated[float, Field(ge=0.0, le=1.0)]

_MAXIMUM_SEEDS = 64
_MAXIMUM_FUZZ_CASES = 256
_CONTRACT = "cancelled work is not scheduled for token commitment"


class CampaignValidationError(ValueError):
    """The campaign is incomplete, altered, or inconsistent with executable evidence."""


class VerificationStrategy(StrEnum):
    TESTS_ONLY = "tests_only"
    FUZZING_ONLY = "fuzzing_only"
    MODEL_CHECK_ONLY = "model_check_only"
    FULL_CEGIS = "full_cegis"


class FaultDisposition(StrEnum):
    DETECTED = "detected"
    PREVENTED_BY_LEARNED_CONSTRAINT = "prevented_by_learned_constraint"
    ESCAPED = "escaped"
    VALID_CONFIRMED = "valid_confirmed"


class _CampaignModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class ArtifactReference(_CampaignModel):
    role: NonEmpty
    path: NonEmpty
    sha256: Sha256
    size_bytes: NonNegativeInt


class CampaignScope(_CampaignModel):
    evidence_scope: Literal["bounded_deterministic_cpu_protocol_fixture"] = (
        "bounded_deterministic_cpu_protocol_fixture"
    )
    hardware_backed: Literal[False] = False
    hardware_performance_claims: Literal[False] = False
    universal_proof: Literal[False] = False
    oracle: Literal["bounded_policy_domain_enumeration"] = "bounded_policy_domain_enumeration"
    fault_family: Literal["deadline_batching_cancellation"] = "deadline_batching_cancellation"


class ProportionInterval(_CampaignModel):
    successes: NonNegativeInt
    trials: PositiveInt
    estimate: Rate
    lower_95: Rate
    upper_95: Rate
    sample_unit: Literal["candidate_fault_instance"] = "candidate_fault_instance"
    method: Literal["wilson_score"] = "wilson_score"

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.successes > self.trials:
            raise ValueError("interval successes exceed trials")
        if not self.lower_95 <= self.estimate <= self.upper_95:
            raise ValueError("interval does not contain its estimate")
        return self


class RawFaultRecord(_CampaignModel):
    schema_version: Literal["sloforge.genesis.h3-fault/v1"] = "sloforge.genesis.h3-fault/v1"
    run_seed: NonNegativeInt
    strategy: VerificationStrategy
    candidate_ordinal: NonNegativeInt
    candidate_id: NonEmpty
    candidate_sha256: Sha256
    known_faulty: bool
    ground_truth_result: Literal["pass", "fail"]
    fault_family: Literal["deadline_batching_cancellation"] = "deadline_batching_cancellation"
    disposition: FaultDisposition
    cases_evaluated: NonNegativeInt
    verifier_invocations: NonNegativeInt
    property_states_enumerated: NonNegativeInt
    model_states_explored: NonNegativeInt
    model_transitions_explored: NonNegativeInt
    minimization_evaluations: NonNegativeInt
    initial_counterexample_events: PositiveInt | None
    minimized_counterexample_events: PositiveInt | None
    learned_constraint_id: NonEmpty | None
    evidence_path: NonEmpty
    evidence_sha256: Sha256

    @model_validator(mode="after")
    def disposition_matches_ground_truth(self) -> Self:
        if self.known_faulty != (self.ground_truth_result == "fail"):
            raise ValueError("fault flag disagrees with independently checked ground truth")
        if self.known_faulty == (self.disposition is FaultDisposition.VALID_CONFIRMED):
            raise ValueError("valid disposition disagrees with fault flag")
        if self.disposition is FaultDisposition.PREVENTED_BY_LEARNED_CONSTRAINT:
            if self.learned_constraint_id is None or self.verifier_invocations != 0:
                raise ValueError(
                    "constraint prevention must cite a constraint and skip verification"
                )
        elif self.learned_constraint_id is not None:
            raise ValueError("only prevented candidates may cite a reused constraint")
        if self.minimized_counterexample_events is not None and (
            self.initial_counterexample_events is None
            or self.minimized_counterexample_events > self.initial_counterexample_events
            or self.minimization_evaluations == 0
        ):
            raise ValueError("invalid minimization accounting")
        return self


class SeedStrategyMetrics(_CampaignModel):
    run_seed: NonNegativeInt
    strategy: VerificationStrategy
    fault_instances: PositiveInt
    detected_faults: NonNegativeInt
    prevented_repeat_faults: NonNegativeInt
    escaped_faults: NonNegativeInt
    valid_candidates_confirmed: NonNegativeInt
    repeated_fault_family_reevaluations: NonNegativeInt
    learned_constraint_reuses: NonNegativeInt
    verifier_invocations: NonNegativeInt
    generated_test_or_fuzz_cases: NonNegativeInt
    property_states_enumerated: NonNegativeInt
    model_states_explored: NonNegativeInt
    model_transitions_explored: NonNegativeInt
    minimization_evaluations: NonNegativeInt
    initial_counterexample_event_sizes: tuple[PositiveInt, ...]
    minimized_counterexample_event_sizes: tuple[PositiveInt, ...]

    @model_validator(mode="after")
    def fault_accounting(self) -> Self:
        if (
            self.detected_faults + self.prevented_repeat_faults + self.escaped_faults
            != self.fault_instances
        ):
            raise ValueError("fault accounting is incomplete")
        return self


class StrategyAggregate(_CampaignModel):
    strategy: VerificationStrategy
    seed_count: PositiveInt
    fault_instances: PositiveInt
    detected_faults: NonNegativeInt
    prevented_repeat_faults: NonNegativeInt
    contained_faults: NonNegativeInt
    escaped_faults: NonNegativeInt
    distinct_fault_families_detected: NonNegativeInt
    repeated_fault_family_reevaluations: NonNegativeInt
    learned_constraint_reuses: NonNegativeInt
    verifier_invocations: NonNegativeInt
    generated_test_or_fuzz_cases: NonNegativeInt
    property_states_enumerated: NonNegativeInt
    model_states_explored: NonNegativeInt
    model_transitions_explored: NonNegativeInt
    minimization_evaluations: NonNegativeInt
    median_initial_counterexample_events: float | None
    median_minimized_counterexample_events: float | None
    containment_interval: ProportionInterval
    per_seed: tuple[SeedStrategyMetrics, ...]

    @model_validator(mode="after")
    def aggregate_accounting(self) -> Self:
        if self.contained_faults != self.detected_faults + self.prevented_repeat_faults:
            raise ValueError("contained faults omit detections or learned prevention")
        if self.contained_faults + self.escaped_faults != self.fault_instances:
            raise ValueError("aggregate fault accounting is incomplete")
        return self


class H3CampaignReport(_CampaignModel):
    schema_version: Literal["sloforge.genesis.h3-campaign/v1"] = "sloforge.genesis.h3-campaign/v1"
    hypothesis_id: Literal["H3"] = "H3"
    statement: Literal["Counterexample-guided synthesis reduces escaped failures."] = (
        "Counterexample-guided synthesis reduces escaped failures."
    )
    base_seed: NonNegativeInt
    run_seeds: tuple[NonNegativeInt, ...]
    fuzz_cases_per_candidate: PositiveInt
    scope: CampaignScope
    raw_records: ArtifactReference
    artifacts: tuple[ArtifactReference, ...]
    aggregates: tuple[StrategyAggregate, ...]
    conclusion: Literal[
        "full_cegis_contained_all_scoped_faults_and_reused_a_learned_constraint"
    ] = "full_cegis_contained_all_scoped_faults_and_reused_a_learned_constraint"
    limitations: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def complete(self) -> Self:
        if not self.run_seeds or len(self.run_seeds) != len(set(self.run_seeds)):
            raise ValueError("run seeds must be non-empty and unique")
        if tuple(item.strategy for item in self.aggregates) != tuple(VerificationStrategy):
            raise ValueError("all strategies must be present in canonical order")
        full = self.aggregates[-1]
        if full.escaped_faults != 0 or full.learned_constraint_reuses != len(self.run_seeds):
            raise ValueError("full CEGIS conclusion is unsupported by its aggregates")
        return self


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _candidate_sha(candidate: CandidateDesign) -> str:
    return _sha256_bytes(canonical_json(candidate))


def _write_json(path: Path, value: BaseModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json(value) + b"\n")


def _write_jsonl(path: Path, records: tuple[RawFaultRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json(record) + b"\n")


def _artifact(root: Path, path: Path, role: str) -> ArtifactReference:
    return ArtifactReference(
        role=role,
        path=path.relative_to(root).as_posix(),
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _derived_seed(base_seed: int, ordinal: int) -> int:
    digest = hashlib.sha256(f"genesis-h3\0{base_seed}\0{ordinal}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _verification_seed(run_seed: int, strategy: VerificationStrategy, *parts: int) -> int:
    suffix = "\0".join(str(part) for part in parts)
    digest = hashlib.sha256(f"h3-check\0{run_seed}\0{strategy.value}\0{suffix}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _witness(actions: tuple[str, ...]) -> ProtocolWitness:
    return ProtocolWitness(
        events=tuple(
            RequestEventCase(
                at_step=index,
                request_id="request-a",
                action=action,  # type: ignore[arg-type]
                worker_id="worker-0",
            )
            for index, action in enumerate(actions)
        )
    )


def _fixed_test_witnesses() -> tuple[ProtocolWitness, ...]:
    """Conventional happy-path and late-cancellation tests, intentionally bounded."""

    return (
        _witness(("admit", "schedule", "prefill", "decode", "emit")),
        _witness(("admit", "schedule", "decode", "emit", "cancel")),
    )


def _fuzz_witnesses(seed: int, count: int) -> tuple[ProtocolWitness, ...]:
    """Generate typed, state-aware protocol schedules without embedding an oracle case."""

    rng = random.Random(seed)
    choices = ("schedule", "prefill", "decode", "emit", "cancel", "disconnect", "retry")
    witnesses: list[ProtocolWitness] = []
    for _ in range(count):
        length = rng.randint(3, 8)
        actions = ["admit"]
        actions.extend(rng.choice(choices) for _ in range(length - 1))
        witnesses.append(_witness(tuple(actions)))
    return tuple(witnesses)


def _outcome_document(witness: ProtocolWitness, outcome: VerificationOutcome) -> dict[str, object]:
    return {
        "witness": witness.model_dump(mode="json"),
        "outcome": outcome.model_dump(mode="json"),
    }


def _ground_truth(
    root: Path, run_seed: int, candidate: CandidateDesign, ordinal: int
) -> tuple[dict[str, object], Path]:
    modelcheck = bounded_candidate_modelcheck_document(candidate, seed=run_seed)
    properties = bounded_candidate_policy_property_document(candidate, seed=run_seed)
    # The strategy under evaluation must not define its own ground truth.  The
    # exhaustive policy-domain enumerator labels this bounded fixture; the
    # explicit-state result is retained as a cross-check only.
    combined_result = "pass" if properties["result"] == "pass" else "fail"
    document: dict[str, object] = {
        "schema_version": "sloforge.genesis.h3-ground-truth/v1",
        "run_seed": run_seed,
        "candidate_ordinal": ordinal,
        "candidate": candidate.model_dump(mode="json"),
        "candidate_sha256": _candidate_sha(candidate),
        "combined_result": combined_result,
        "checks_agree": modelcheck["result"] == properties["result"],
        "modelcheck": modelcheck,
        "property_enumeration": properties,
        "scope": CampaignScope().model_dump(mode="json"),
    }
    path = root / "ground-truth" / str(run_seed) / f"{ordinal:02d}-{candidate.candidate_id}.json"
    _write_json(path, document)
    return document, path


def _record(
    *,
    root: Path,
    run_seed: int,
    strategy: VerificationStrategy,
    candidate: CandidateDesign,
    ordinal: int,
    known_faulty: bool,
    disposition: FaultDisposition,
    cases: int,
    invocations: int,
    evidence_path: Path,
    property_states: int = 0,
    model_states: int = 0,
    model_transitions: int = 0,
    minimization_evaluations: int = 0,
    initial_size: int | None = None,
    minimized_size: int | None = None,
    constraint_id: str | None = None,
) -> RawFaultRecord:
    return RawFaultRecord(
        run_seed=run_seed,
        strategy=strategy,
        candidate_ordinal=ordinal,
        candidate_id=candidate.candidate_id,
        candidate_sha256=_candidate_sha(candidate),
        known_faulty=known_faulty,
        ground_truth_result="fail" if known_faulty else "pass",
        disposition=disposition,
        cases_evaluated=cases,
        verifier_invocations=invocations,
        property_states_enumerated=property_states,
        model_states_explored=model_states,
        model_transitions_explored=model_transitions,
        minimization_evaluations=minimization_evaluations,
        initial_counterexample_events=initial_size,
        minimized_counterexample_events=minimized_size,
        learned_constraint_id=constraint_id,
        evidence_path=evidence_path.relative_to(root).as_posix(),
        evidence_sha256=_sha256(evidence_path),
    )


def _example_or_fuzz_records(
    root: Path,
    run_seed: int,
    candidates: tuple[CandidateDesign, ...],
    known_faults: tuple[bool, ...],
    strategy: VerificationStrategy,
    *,
    fuzz_cases: int,
) -> list[RawFaultRecord]:
    verifier = CancellationPolicyVerifier()
    records: list[RawFaultRecord] = []
    for ordinal, (candidate, known_faulty) in enumerate(zip(candidates, known_faults, strict=True)):
        if strategy is VerificationStrategy.TESTS_ONLY:
            witnesses = _fixed_test_witnesses()
        else:
            witnesses = _fuzz_witnesses(_verification_seed(run_seed, strategy, ordinal), fuzz_cases)
        outcomes: list[dict[str, object]] = []
        detected_size: int | None = None
        for case_index, witness in enumerate(witnesses):
            outcome = verifier.verify(
                candidate,
                witness,
                seed=_verification_seed(run_seed, strategy, ordinal, case_index),
            )
            outcomes.append(_outcome_document(witness, outcome))
            if not outcome.passed:
                detected_size = len(witness.events)
                break
        detected = detected_size is not None
        disposition = (
            FaultDisposition.DETECTED
            if known_faulty and detected
            else FaultDisposition.ESCAPED
            if known_faulty
            else FaultDisposition.VALID_CONFIRMED
        )
        evidence = {
            "schema_version": "sloforge.genesis.h3-execution-evidence/v1",
            "run_seed": run_seed,
            "strategy": strategy.value,
            "candidate_id": candidate.candidate_id,
            "candidate_sha256": _candidate_sha(candidate),
            "case_budget": len(witnesses),
            "stopped_after_first_failure": True,
            "cases": outcomes,
            "detected": detected,
        }
        path = root / "runs" / str(run_seed) / strategy.value / f"{ordinal:02d}.json"
        _write_json(path, evidence)
        records.append(
            _record(
                root=root,
                run_seed=run_seed,
                strategy=strategy,
                candidate=candidate,
                ordinal=ordinal,
                known_faulty=known_faulty,
                disposition=disposition,
                cases=len(outcomes),
                invocations=len(outcomes),
                evidence_path=path,
                initial_size=detected_size,
            )
        )
    return records


def _modelcheck_records(
    root: Path,
    run_seed: int,
    candidates: tuple[CandidateDesign, ...],
    ground_truth: tuple[dict[str, object], ...],
) -> list[RawFaultRecord]:
    records: list[RawFaultRecord] = []
    for ordinal, (candidate, truth) in enumerate(zip(candidates, ground_truth, strict=True)):
        evidence = bounded_candidate_modelcheck_document(candidate, seed=run_seed)
        if evidence != truth["modelcheck"]:
            raise RuntimeError("model-check strategy differs from ground truth evaluation")
        failed = evidence["result"] == "fail"
        known_faulty = truth["combined_result"] == "fail"
        if failed and not known_faulty:
            raise RuntimeError("model-check strategy rejected the independent valid fixture")
        trace = evidence["counterexample_trace"]
        initial_size = len(trace) if isinstance(trace, list) and trace else None
        state_count = evidence["state_count"]
        transition_count = evidence["transition_count"]
        if type(state_count) is not int or type(transition_count) is not int:
            raise RuntimeError("model-check evidence omitted integer exploration counts")
        path = root / "runs" / str(run_seed) / VerificationStrategy.MODEL_CHECK_ONLY.value
        path /= f"{ordinal:02d}.json"
        _write_json(path, evidence)
        records.append(
            _record(
                root=root,
                run_seed=run_seed,
                strategy=VerificationStrategy.MODEL_CHECK_ONLY,
                candidate=candidate,
                ordinal=ordinal,
                known_faulty=known_faulty,
                disposition=(
                    FaultDisposition.DETECTED
                    if known_faulty and failed
                    else FaultDisposition.ESCAPED
                    if known_faulty
                    else FaultDisposition.VALID_CONFIRMED
                ),
                cases=0,
                invocations=1,
                evidence_path=path,
                model_states=state_count,
                model_transitions=transition_count,
                initial_size=initial_size,
            )
        )
    return records


def _verification_seed_from_counterexample(counterexample: Counterexample) -> int:
    facts = {fact.name: fact.value for fact in counterexample.environment}
    try:
        return int(facts["verification_seed"])
    except (KeyError, ValueError) as error:
        raise CampaignValidationError("counterexample omits its verification seed") from error


def _cegis_counterexamples(directory: Path) -> tuple[Counterexample, ...]:
    return tuple(load_counterexample(path) for path in sorted(directory.glob("*.json")))


def _full_cegis_records(
    root: Path,
    run_seed: int,
    candidates: tuple[CandidateDesign, ...],
    known_faults: tuple[bool, ...],
) -> list[RawFaultRecord]:
    run_directory = root / "runs" / str(run_seed) / VerificationStrategy.FULL_CEGIS.value
    result = run_cancellation_cegis(run_directory, seed=run_seed)
    counterexamples = _cegis_counterexamples(run_directory / "counterexamples")
    original = next(item for item in counterexamples if not item.minimized)
    minimized = next(item for item in counterexamples if item.minimized)
    constraints = ConstraintStore(run_directory / "constraints.json")
    if len(constraints.constraints) != 1:
        raise RuntimeError("full CEGIS fixture did not persist exactly one learned constraint")
    constraint = constraints.constraints[0]
    accepted = candidates[2]
    accepted_property = bounded_candidate_policy_property_document(accepted, seed=run_seed)
    accepted_modelcheck = bounded_candidate_modelcheck_document(accepted, seed=run_seed)
    property_states = accepted_property["states_checked"]
    model_states = accepted_modelcheck["state_count"]
    model_transitions = accepted_modelcheck["transition_count"]
    if (
        type(property_states) is not int
        or type(model_states) is not int
        or type(model_transitions) is not int
    ):
        raise RuntimeError("accepted candidate checks omitted integer exploration counts")
    summary = {
        "schema_version": "sloforge.genesis.h3-full-cegis-evidence/v1",
        "run_seed": run_seed,
        "result": result.model_dump(mode="json"),
        "events_sha256": _sha256(run_directory / "events.jsonl"),
        "constraints_sha256": _sha256(run_directory / "constraints.json"),
        "counterexamples": [
            {
                "counterexample_id": item.counterexample_id,
                "minimized": item.minimized,
                "event_count": len(item.payload.events),  # type: ignore[union-attr]
                "sha256": _sha256(
                    run_directory / "counterexamples" / f"{item.counterexample_id}.json"
                ),
            }
            for item in counterexamples
        ],
        "accepted_property": accepted_property,
        "accepted_modelcheck": accepted_modelcheck,
    }
    summary_path = run_directory / "summary.json"
    _write_json(summary_path, summary)
    minimized_size = len(minimized.payload.events)  # type: ignore[union-attr]
    original_size = len(original.payload.events)  # type: ignore[union-attr]
    return [
        _record(
            root=root,
            run_seed=run_seed,
            strategy=VerificationStrategy.FULL_CEGIS,
            candidate=candidates[0],
            ordinal=0,
            known_faulty=known_faults[0],
            disposition=FaultDisposition.DETECTED,
            cases=1,
            invocations=1 + result.minimization_evaluations,
            evidence_path=summary_path,
            minimization_evaluations=result.minimization_evaluations,
            initial_size=original_size,
            minimized_size=minimized_size,
        ),
        _record(
            root=root,
            run_seed=run_seed,
            strategy=VerificationStrategy.FULL_CEGIS,
            candidate=candidates[1],
            ordinal=1,
            known_faulty=known_faults[1],
            disposition=FaultDisposition.PREVENTED_BY_LEARNED_CONSTRAINT,
            cases=0,
            invocations=0,
            evidence_path=summary_path,
            constraint_id=constraint.learned.constraint_id,
        ),
        _record(
            root=root,
            run_seed=run_seed,
            strategy=VerificationStrategy.FULL_CEGIS,
            candidate=candidates[2],
            ordinal=2,
            known_faulty=known_faults[2],
            disposition=FaultDisposition.VALID_CONFIRMED,
            cases=1,
            invocations=3,
            evidence_path=summary_path,
            property_states=property_states,
            model_states=model_states,
            model_transitions=model_transitions,
        ),
    ]


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _wilson(successes: int, trials: int) -> ProportionInterval:
    z = 1.959963984540054
    estimate = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (estimate + z2 / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(estimate * (1.0 - estimate) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return ProportionInterval(
        successes=successes,
        trials=trials,
        estimate=estimate,
        lower_95=0.0 if successes == 0 else max(0.0, center - margin),
        upper_95=1.0 if successes == trials else min(1.0, center + margin),
    )


def _seed_metrics(records: tuple[RawFaultRecord, ...]) -> SeedStrategyMetrics:
    first = records[0]
    fault_records = tuple(item for item in records if item.known_faulty)
    initial_sizes = tuple(
        item.initial_counterexample_events
        for item in records
        if item.initial_counterexample_events is not None
    )
    minimized_sizes = tuple(
        item.minimized_counterexample_events
        for item in records
        if item.minimized_counterexample_events is not None
    )
    return SeedStrategyMetrics(
        run_seed=first.run_seed,
        strategy=first.strategy,
        fault_instances=len(fault_records),
        detected_faults=sum(
            item.disposition is FaultDisposition.DETECTED for item in fault_records
        ),
        prevented_repeat_faults=sum(
            item.disposition is FaultDisposition.PREVENTED_BY_LEARNED_CONSTRAINT
            for item in fault_records
        ),
        escaped_faults=sum(item.disposition is FaultDisposition.ESCAPED for item in fault_records),
        valid_candidates_confirmed=sum(
            item.disposition is FaultDisposition.VALID_CONFIRMED for item in records
        ),
        repeated_fault_family_reevaluations=int(
            fault_records[1].disposition is not FaultDisposition.PREVENTED_BY_LEARNED_CONSTRAINT
        ),
        learned_constraint_reuses=sum(
            item.disposition is FaultDisposition.PREVENTED_BY_LEARNED_CONSTRAINT for item in records
        ),
        verifier_invocations=sum(item.verifier_invocations for item in records),
        generated_test_or_fuzz_cases=sum(item.cases_evaluated for item in records),
        property_states_enumerated=sum(item.property_states_enumerated for item in records),
        model_states_explored=sum(item.model_states_explored for item in records),
        model_transitions_explored=sum(item.model_transitions_explored for item in records),
        minimization_evaluations=sum(item.minimization_evaluations for item in records),
        initial_counterexample_event_sizes=initial_sizes,
        minimized_counterexample_event_sizes=minimized_sizes,
    )


def _aggregates(
    records: tuple[RawFaultRecord, ...], seeds: tuple[int, ...]
) -> tuple[StrategyAggregate, ...]:
    results: list[StrategyAggregate] = []
    for strategy in VerificationStrategy:
        per_seed = tuple(
            _seed_metrics(
                tuple(
                    item for item in records if item.run_seed == seed and item.strategy is strategy
                )
            )
            for seed in seeds
        )
        faults = sum(item.fault_instances for item in per_seed)
        detected = sum(item.detected_faults for item in per_seed)
        prevented = sum(item.prevented_repeat_faults for item in per_seed)
        initial_sizes = [
            size for item in per_seed for size in item.initial_counterexample_event_sizes
        ]
        minimized_sizes = [
            size for item in per_seed for size in item.minimized_counterexample_event_sizes
        ]
        results.append(
            StrategyAggregate(
                strategy=strategy,
                seed_count=len(seeds),
                fault_instances=faults,
                detected_faults=detected,
                prevented_repeat_faults=prevented,
                contained_faults=detected + prevented,
                escaped_faults=sum(item.escaped_faults for item in per_seed),
                distinct_fault_families_detected=int(detected > 0),
                repeated_fault_family_reevaluations=sum(
                    item.repeated_fault_family_reevaluations for item in per_seed
                ),
                learned_constraint_reuses=sum(item.learned_constraint_reuses for item in per_seed),
                verifier_invocations=sum(item.verifier_invocations for item in per_seed),
                generated_test_or_fuzz_cases=sum(
                    item.generated_test_or_fuzz_cases for item in per_seed
                ),
                property_states_enumerated=sum(
                    item.property_states_enumerated for item in per_seed
                ),
                model_states_explored=sum(item.model_states_explored for item in per_seed),
                model_transitions_explored=sum(
                    item.model_transitions_explored for item in per_seed
                ),
                minimization_evaluations=sum(item.minimization_evaluations for item in per_seed),
                median_initial_counterexample_events=_median(initial_sizes),
                median_minimized_counterexample_events=_median(minimized_sizes),
                containment_interval=_wilson(detected + prevented, faults),
                per_seed=per_seed,
            )
        )
    return tuple(results)


def _all_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file() and path.name != "report.json"),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def run_cegis_campaign(
    output_directory: Path,
    *,
    base_seed: int = 73129,
    seed_count: int = 5,
    fuzz_cases_per_candidate: int = 12,
) -> H3CampaignReport:
    """Run all four H3 strategies and persist independently checkable raw evidence."""

    if base_seed < 0:
        raise ValueError("base seed must be non-negative")
    if not 1 <= seed_count <= _MAXIMUM_SEEDS:
        raise ValueError(f"seed count must be in [1,{_MAXIMUM_SEEDS}]")
    if not 1 <= fuzz_cases_per_candidate <= _MAXIMUM_FUZZ_CASES:
        raise ValueError(f"fuzz case count must be in [1,{_MAXIMUM_FUZZ_CASES}]")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite campaign directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    seeds = tuple(_derived_seed(base_seed, ordinal) for ordinal in range(seed_count))
    records: list[RawFaultRecord] = []
    for run_seed in seeds:
        candidates = cancellation_fixture_candidates(run_seed)
        truth_documents: list[dict[str, object]] = []
        truth_flags: list[bool] = []
        for ordinal, candidate in enumerate(candidates):
            truth, _path = _ground_truth(output_directory, run_seed, candidate, ordinal)
            truth_documents.append(truth)
            modelcheck = truth["modelcheck"]
            if not isinstance(modelcheck, dict):
                raise RuntimeError("ground-truth modelcheck is malformed")
            truth_flags.append(truth["combined_result"] == "fail")
        known_faults = tuple(truth_flags)
        records.extend(
            _example_or_fuzz_records(
                output_directory,
                run_seed,
                candidates,
                known_faults,
                VerificationStrategy.TESTS_ONLY,
                fuzz_cases=fuzz_cases_per_candidate,
            )
        )
        records.extend(
            _example_or_fuzz_records(
                output_directory,
                run_seed,
                candidates,
                known_faults,
                VerificationStrategy.FUZZING_ONLY,
                fuzz_cases=fuzz_cases_per_candidate,
            )
        )
        records.extend(
            _modelcheck_records(output_directory, run_seed, candidates, tuple(truth_documents))
        )
        records.extend(_full_cegis_records(output_directory, run_seed, candidates, known_faults))
    raw_records = tuple(records)
    raw_path = output_directory / "raw_fault_records.jsonl"
    _write_jsonl(raw_path, raw_records)
    artifact_paths = _all_files(output_directory)
    artifacts = tuple(_artifact(output_directory, path, "raw_evidence") for path in artifact_paths)
    report = H3CampaignReport(
        base_seed=base_seed,
        run_seeds=seeds,
        fuzz_cases_per_candidate=fuzz_cases_per_candidate,
        scope=CampaignScope(),
        raw_records=_artifact(output_directory, raw_path, "raw_fault_records"),
        artifacts=artifacts,
        aggregates=_aggregates(raw_records, seeds),
        limitations=(
            "The evaluated fault family is one bounded cancellation-policy fixture, not arbitrary generated code.",
            "Wilson intervals use correlated candidate fault instances and are descriptive, not population inference.",
            "Tests-only uses a declared happy-path and late-cancellation suite; other test suites may differ.",
            "Fuzzing is schema-aware deterministic schedule generation with a fixed per-candidate budget.",
            "Model checking is exhaustive only within the recorded single-request, depth-four abstraction.",
            "Fault labels come from exhaustive policy-domain enumeration; explicit-state model checking is evaluated as a separate strategy and retained only as a cross-check.",
            "All results are CPU protocol evidence; no latency, throughput, GPU, or universal-proof claim is made.",
        ),
    )
    _write_json(output_directory / "report.json", report)
    return report


def _safe_artifact(root: Path, reference: ArtifactReference) -> Path:
    path = (root / reference.path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise CampaignValidationError(
            f"artifact escapes campaign root: {reference.path}"
        ) from error
    if not path.is_file():
        raise CampaignValidationError(f"missing campaign artifact: {reference.path}")
    if path.stat().st_size != reference.size_bytes or _sha256(path) != reference.sha256:
        raise CampaignValidationError(f"campaign artifact changed: {reference.path}")
    return path


def _load_records(path: Path) -> tuple[RawFaultRecord, ...]:
    try:
        return tuple(
            RawFaultRecord.model_validate_json(line, strict=True)
            for line in path.read_bytes().splitlines()
            if line
        )
    except (ValueError, json.JSONDecodeError) as error:
        raise CampaignValidationError("raw fault records are invalid") from error


def _validate_execution_evidence(
    path: Path, candidate: CandidateDesign, record: RawFaultRecord
) -> None:
    try:
        document = json.loads(path.read_bytes())
        cases = document["cases"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise CampaignValidationError("test or fuzz evidence is malformed") from error
    if document.get("candidate_sha256") != _candidate_sha(candidate):
        raise CampaignValidationError("execution evidence candidate digest differs")
    verifier = CancellationPolicyVerifier()
    detected = False
    for case_index, case in enumerate(cases):
        witness = ProtocolWitness.model_validate_json(
            json.dumps(case["witness"], sort_keys=True, separators=(",", ":")), strict=True
        )
        expected = verifier.verify(
            candidate,
            witness,
            seed=_verification_seed(
                record.run_seed, record.strategy, record.candidate_ordinal, case_index
            ),
        )
        observed = VerificationOutcome.model_validate_json(
            json.dumps(case["outcome"], sort_keys=True, separators=(",", ":")), strict=True
        )
        if expected != observed:
            raise CampaignValidationError("stored verifier outcome does not replay")
        detected |= not observed.passed
    if len(cases) != record.cases_evaluated:
        raise CampaignValidationError("execution evidence case count differs")
    observed_size = None
    if detected:
        failed_case = next(case for case in cases if not case["outcome"]["passed"])
        observed_size = len(failed_case["witness"]["events"])
    if record.initial_counterexample_events != observed_size:
        raise CampaignValidationError("execution counterexample size differs")
    expected_disposition = (
        FaultDisposition.DETECTED
        if record.known_faulty and detected
        else FaultDisposition.ESCAPED
        if record.known_faulty
        else FaultDisposition.VALID_CONFIRMED
    )
    if record.disposition is not expected_disposition:
        raise CampaignValidationError("execution evidence disposition differs")


def _validate_minimal_counterexample(
    candidate: CandidateDesign, counterexample: Counterexample
) -> None:
    events = counterexample.payload.events  # type: ignore[union-attr]
    verifier = CancellationPolicyVerifier()
    seed = _verification_seed_from_counterexample(counterexample)
    outcome = verifier.verify(candidate, ProtocolWitness(events=events), seed=seed)
    if outcome.passed or outcome.failure is None or outcome.failure.violated_contract != _CONTRACT:
        raise CampaignValidationError("minimized counterexample no longer reproduces")
    for removed in range(len(events)):
        remaining = events[:removed] + events[removed + 1 :]
        if not remaining:
            continue
        witness = ProtocolWitness(
            events=tuple(
                event.model_copy(update={"at_step": index}) for index, event in enumerate(remaining)
            )
        )
        if not verifier.verify(candidate, witness, seed=seed).passed:
            raise CampaignValidationError("counterexample is not one-event minimal")


def _validate_full_cegis(
    root: Path,
    run_seed: int,
    candidates: tuple[CandidateDesign, ...],
    records: tuple[RawFaultRecord, ...],
) -> None:
    directory = root / "runs" / str(run_seed) / VerificationStrategy.FULL_CEGIS.value
    summary_path = directory / "summary.json"
    try:
        summary = json.loads(summary_path.read_bytes())
        result = CegisRunResult.model_validate_json(
            json.dumps(summary["result"], sort_keys=True, separators=(",", ":")), strict=True
        )
    except (json.JSONDecodeError, KeyError, ValueError) as error:
        raise CampaignValidationError("full CEGIS summary is malformed") from error
    if (
        result.rejected_candidate_ids != (candidates[0].candidate_id,)
        or result.suppressed_candidate_ids != (candidates[1].candidate_id,)
        or result.accepted_candidate_id != candidates[2].candidate_id
    ):
        raise CampaignValidationError("full CEGIS lifecycle differs from its candidate population")
    events_path = directory / "events.jsonl"
    constraint_path = directory / "constraints.json"
    if Path(result.events_path).resolve() != events_path.resolve():
        raise CampaignValidationError("CEGIS result points at a different event log")
    if summary.get("events_sha256") != _sha256(events_path) or summary.get(
        "constraints_sha256"
    ) != _sha256(constraint_path):
        raise CampaignValidationError("full CEGIS summary does not bind events and constraints")
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    if [event.get("sequence") for event in events] != list(range(len(events))):
        raise CampaignValidationError("CEGIS events are not a contiguous audit log")
    document = ConstraintDocument.model_validate_json(constraint_path.read_bytes(), strict=True)
    if len(document.constraints) != 1:
        raise CampaignValidationError("CEGIS did not persist one generalized constraint")
    constraint = document.constraints[0]
    if not constraint.rejects(candidates[1]) or constraint.rejects(candidates[2]):
        raise CampaignValidationError(
            "learned constraint does not suppress only the repeated fault"
        )
    counterexamples = _cegis_counterexamples(directory / "counterexamples")
    if len(counterexamples) != 2:
        raise CampaignValidationError("CEGIS counterexample corpus is incomplete")
    original = next((item for item in counterexamples if not item.minimized), None)
    minimized = next((item for item in counterexamples if item.minimized), None)
    if (
        original is None
        or minimized is None
        or minimized.parent_counterexample_id != original.counterexample_id
    ):
        raise CampaignValidationError("CEGIS counterexample minimization lineage is invalid")
    _validate_minimal_counterexample(candidates[0], minimized)
    expected_counterexamples = sorted(
        (
            item.counterexample_id,
            item.minimized,
            len(item.payload.events),  # type: ignore[union-attr]
            _sha256(directory / "counterexamples" / f"{item.counterexample_id}.json"),
        )
        for item in counterexamples
    )
    observed_counterexamples = sorted(
        (
            item["counterexample_id"],
            item["minimized"],
            item["event_count"],
            item["sha256"],
        )
        for item in summary.get("counterexamples", [])
    )
    if observed_counterexamples != expected_counterexamples:
        raise CampaignValidationError("CEGIS summary does not bind its counterexample corpus")
    expected_property = bounded_candidate_policy_property_document(candidates[2], seed=run_seed)
    expected_modelcheck = bounded_candidate_modelcheck_document(candidates[2], seed=run_seed)
    if (
        summary.get("accepted_property") != expected_property
        or summary.get("accepted_modelcheck") != expected_modelcheck
    ):
        raise CampaignValidationError("accepted candidate verification evidence does not replay")
    ordered = tuple(sorted(records, key=lambda item: item.candidate_ordinal))
    if len(ordered) != 3:
        raise CampaignValidationError("full CEGIS raw records are incomplete")
    if (
        ordered[0].disposition is not FaultDisposition.DETECTED
        or ordered[0].verifier_invocations != 1 + result.minimization_evaluations
        or ordered[0].minimization_evaluations != result.minimization_evaluations
        or ordered[0].initial_counterexample_events != len(original.payload.events)  # type: ignore[union-attr]
        or ordered[0].minimized_counterexample_events != len(minimized.payload.events)  # type: ignore[union-attr]
        or ordered[1].disposition is not FaultDisposition.PREVENTED_BY_LEARNED_CONSTRAINT
        or ordered[1].learned_constraint_id != constraint.learned.constraint_id
        or ordered[2].disposition is not FaultDisposition.VALID_CONFIRMED
        or ordered[2].verifier_invocations != 3
        or ordered[2].property_states_enumerated != expected_property["states_checked"]
        or ordered[2].model_states_explored != expected_modelcheck["state_count"]
        or ordered[2].model_transitions_explored != expected_modelcheck["transition_count"]
    ):
        raise CampaignValidationError("full CEGIS raw accounting differs from replayed evidence")


def validate_cegis_campaign(output_directory: Path) -> H3CampaignReport:
    """Reopen, hash-check, replay, and independently aggregate a campaign."""

    report_path = output_directory / "report.json"
    try:
        report = H3CampaignReport.model_validate_json(report_path.read_bytes(), strict=True)
    except (OSError, ValueError) as error:
        raise CampaignValidationError("H3 report is missing or invalid") from error
    manifest_paths = tuple(reference.path for reference in report.artifacts)
    if len(manifest_paths) != len(set(manifest_paths)):
        raise CampaignValidationError("artifact manifest contains duplicate paths")
    actual_paths = tuple(
        path.relative_to(output_directory).as_posix() for path in _all_files(output_directory)
    )
    if manifest_paths != actual_paths:
        raise CampaignValidationError("artifact manifest does not exactly cover campaign files")
    for reference in report.artifacts:
        _safe_artifact(output_directory, reference)
    artifact_by_path = {reference.path: reference for reference in report.artifacts}
    raw_path = _safe_artifact(output_directory, report.raw_records)
    records = _load_records(raw_path)
    expected_count = len(report.run_seeds) * len(VerificationStrategy) * 3
    if len(records) != expected_count:
        raise CampaignValidationError("raw record population is incomplete")
    expected_keys = {
        (seed, strategy, ordinal)
        for seed in report.run_seeds
        for strategy in VerificationStrategy
        for ordinal in range(3)
    }
    observed_keys = {
        (record.run_seed, record.strategy, record.candidate_ordinal) for record in records
    }
    if observed_keys != expected_keys or len(observed_keys) != len(records):
        raise CampaignValidationError("raw records are duplicated or omit a strategy candidate")
    for run_seed in report.run_seeds:
        candidates = cancellation_fixture_candidates(run_seed)
        truth_flags: list[bool] = []
        for ordinal, candidate in enumerate(candidates):
            truth_path = (
                output_directory
                / "ground-truth"
                / str(run_seed)
                / f"{ordinal:02d}-{candidate.candidate_id}.json"
            )
            truth = json.loads(truth_path.read_bytes())
            expected_modelcheck = bounded_candidate_modelcheck_document(candidate, seed=run_seed)
            expected_property = bounded_candidate_policy_property_document(candidate, seed=run_seed)
            if (
                truth.get("candidate_sha256") != _candidate_sha(candidate)
                or truth.get("combined_result")
                != ("pass" if expected_property["result"] == "pass" else "fail")
                or truth.get("modelcheck") != expected_modelcheck
                or truth.get("property_enumeration") != expected_property
                or truth.get("checks_agree")
                != (expected_modelcheck["result"] == expected_property["result"])
                or truth.get("scope") != CampaignScope().model_dump(mode="json")
            ):
                raise CampaignValidationError("ground truth does not independently replay")
            truth_flags.append(
                expected_modelcheck["result"] == "fail" or expected_property["result"] == "fail"
            )
        if tuple(truth_flags) != (True, True, False):
            raise CampaignValidationError(
                "fixture ground truth no longer has two unsafe candidates"
            )
        for record in (item for item in records if item.run_seed == run_seed):
            candidate = candidates[record.candidate_ordinal]
            if (
                record.candidate_id != candidate.candidate_id
                or record.candidate_sha256 != _candidate_sha(candidate)
                or record.known_faulty != truth_flags[record.candidate_ordinal]
            ):
                raise CampaignValidationError("raw record candidate identity differs")
            evidence_reference = artifact_by_path.get(record.evidence_path)
            if evidence_reference is None or evidence_reference.sha256 != record.evidence_sha256:
                raise CampaignValidationError("raw record evidence is absent from the manifest")
            evidence_path = _safe_artifact(output_directory, evidence_reference)
            if record.strategy in (
                VerificationStrategy.TESTS_ONLY,
                VerificationStrategy.FUZZING_ONLY,
            ):
                _validate_execution_evidence(evidence_path, candidate, record)
            elif record.strategy is VerificationStrategy.MODEL_CHECK_ONLY:
                observed = json.loads(evidence_path.read_bytes())
                expected = bounded_candidate_modelcheck_document(candidate, seed=run_seed)
                trace = expected["counterexample_trace"]
                expected_size = len(trace) if isinstance(trace, list) and trace else None
                if (
                    observed != expected
                    or record.model_states_explored != expected["state_count"]
                    or record.model_transitions_explored != expected["transition_count"]
                    or record.initial_counterexample_events != expected_size
                ):
                    raise CampaignValidationError("model-check evidence does not replay")
        _validate_full_cegis(
            output_directory,
            run_seed,
            candidates,
            tuple(
                item
                for item in records
                if item.run_seed == run_seed and item.strategy is VerificationStrategy.FULL_CEGIS
            ),
        )
    if report.aggregates != _aggregates(records, report.run_seeds):
        raise CampaignValidationError("reported aggregates are not derived from raw records")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", required=True, type=Path)
    run_parser.add_argument("--seed", type=int, default=73129)
    run_parser.add_argument("--seed-count", type=int, default=5)
    run_parser.add_argument("--fuzz-cases", type=int, default=12)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.command == "run":
        report = run_cegis_campaign(
            arguments.output,
            base_seed=arguments.seed,
            seed_count=arguments.seed_count,
            fuzz_cases_per_candidate=arguments.fuzz_cases,
        )
    else:
        report = validate_cegis_campaign(arguments.output)
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CampaignValidationError",
    "FaultDisposition",
    "H3CampaignReport",
    "VerificationStrategy",
    "run_cegis_campaign",
    "validate_cegis_campaign",
]
