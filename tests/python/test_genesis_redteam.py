from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.genesis.ir import (
    Counterexample,
    RequestTraceCounterexamplePayload,
    load_counterexample,
)
from sloforge.redteam import (
    BenchmarkIntegrityCode,
    RedTeamConfiguration,
    RedTeamSurface,
    RegressionCorpus,
    ResourceAdversaryConfiguration,
    ScheduleAdversarialCase,
    ScheduleAdversaryConfiguration,
    TensorAdversaryConfiguration,
    TopologyAdversaryConfiguration,
    UnsafeStreamingCandidate,
    audit_benchmark_integrity,
    corpus_from_report,
    generate_resource_cases,
    generate_schedule_cases,
    generate_tensor_cases,
    generate_topology_cases,
    minimize_sequence,
    replay_regression_corpus,
    run_demo,
    run_red_team,
    unsafe_benchmark_comparison,
)


def test_typed_adversaries_are_seeded_deterministic_and_bounded() -> None:
    tensor_configuration = TensorAdversaryConfiguration(
        seed=41,
        maximum_cases=14,
        maximum_rank=3,
        maximum_dimension=17,
    )
    tensor_cases = generate_tensor_cases(tensor_configuration)
    assert tensor_cases == generate_tensor_cases(tensor_configuration)
    assert len(tensor_cases) == 14
    assert tensor_cases[0].input.non_contiguous
    assert len(tensor_cases[0].input.strides) == len(tensor_cases[0].input.shape)
    assert tensor_cases[0].input.values_hex
    assert all(len(case.input.shape) <= 3 for case in tensor_cases)
    assert all(max(case.input.shape) <= 17 for case in tensor_cases)
    assert len({case.input.dtype for case in tensor_cases}) == 12

    schedule_configuration = ScheduleAdversaryConfiguration(
        seed=42,
        maximum_cases=10,
        maximum_events=12,
        request_count=3,
        worker_count=2,
    )
    schedules = generate_schedule_cases(schedule_configuration)
    assert schedules == generate_schedule_cases(schedule_configuration)
    assert {event.action for event in schedules[0].events} >= {"cancel", "fail", "retry"}
    assert all(
        tuple(event.at_step for event in schedule.events) == tuple(range(len(schedule.events)))
        for schedule in schedules
    )

    topology_configuration = TopologyAdversaryConfiguration(seed=43, maximum_cases=6)
    topology_cases = generate_topology_cases(topology_configuration)
    assert topology_cases == generate_topology_cases(topology_configuration)
    assert topology_cases[0].topology.failed_links == ("nvlink-0-1",)

    resource_configuration = ResourceAdversaryConfiguration(
        seed=44,
        maximum_cases=6,
        maximum_device_bytes=8192,
        maximum_host_bytes=16384,
        maximum_queue_depth=16,
        maximum_process_count=8,
    )
    resource_cases = generate_resource_cases(resource_configuration)
    assert resource_cases == generate_resource_cases(resource_configuration)
    assert resource_cases[0].resource.queue_depth == 8
    assert all(case.resource.queue_depth <= 16 for case in resource_cases)


def test_delta_minimizer_preserves_real_ordered_failure() -> None:
    source = ("admit", "noise-a", "emit", "noise-b", "cancel", "noise-c", "retry")

    def violation(items: tuple[str, ...]) -> bool:
        required = iter(("admit", "emit", "cancel", "retry"))
        current = next(required, None)
        for item in items:
            if item == current:
                current = next(required, None)
        return current is None

    first = minimize_sequence(
        source,
        violation,
        seed=99,
        maximum_evaluations=64,
        timeout_seconds=1.0,
    )
    second = minimize_sequence(
        source,
        violation,
        seed=99,
        maximum_evaluations=64,
        timeout_seconds=1.0,
    )
    assert first == second
    assert first.items == ("admit", "emit", "cancel", "retry")
    assert first.evaluations <= 64
    with pytest.raises(ValueError, match="does not reproduce"):
        minimize_sequence(
            ("safe",),
            violation,
            seed=0,
            maximum_evaluations=2,
            timeout_seconds=1.0,
        )


def test_benchmark_integrity_auditor_checks_every_declared_attack() -> None:
    unsafe = unsafe_benchmark_comparison()
    issues = audit_benchmark_integrity(unsafe)
    assert {issue.code for issue in issues} == set(BenchmarkIntegrityCode)

    clean_candidate = unsafe.baseline.model_copy(
        update={"run_id": "candidate-clean", "candidate_id": "candidate-clean"}
    )
    clean = unsafe.model_copy(update={"candidate": clean_candidate})
    assert audit_benchmark_integrity(clean) == ()


def test_unsafe_candidate_is_executed_minimized_and_converted() -> None:
    target = UnsafeStreamingCandidate()
    comparison = unsafe_benchmark_comparison()
    configuration = RedTeamConfiguration(
        seed=73129,
        maximum_findings=32,
        maximum_minimization_evaluations=128,
        minimization_timeout_seconds=2.0,
        run_timeout_seconds=10.0,
        tensor_cases=8,
        schedule_cases=8,
        topology_cases=4,
        resource_cases=4,
    )
    report = run_red_team(
        target=target,
        configuration=configuration,
        benchmark_comparison=comparison,
    )
    assert not report.timed_out
    surfaces = {finding.surface for finding in report.findings}
    assert surfaces == set(RedTeamSurface)
    protocol = next(
        finding for finding in report.findings if finding.surface is RedTeamSurface.PROTOCOL
    )
    assert isinstance(protocol.minimized_case, ScheduleAdversarialCase)
    assert isinstance(protocol.original_case, ScheduleAdversarialCase)
    assert len(protocol.minimized_case.events) == 4
    assert len(protocol.minimized_case.events) < len(protocol.original_case.events)
    assert [event.action for event in protocol.minimized_case.events] == [
        "admit",
        "emit",
        "cancel",
        "retry",
    ]
    assert isinstance(protocol.counterexample, Counterexample)
    assert isinstance(protocol.counterexample.payload, RequestTraceCounterexamplePayload)
    assert protocol.counterexample.minimized
    assert protocol.learned_constraint.counterexample_ids == (
        protocol.counterexample.counterexample_id,
    )

    corpus = corpus_from_report(report)
    assert corpus.constraints
    assert len(corpus.cases) == len(report.findings)
    replay = replay_regression_corpus(
        target=target,
        corpus=corpus,
        benchmark_comparison=comparison,
    )
    assert replay
    assert all(result.reproduced for result in replay)


def test_demo_artifacts_are_canonical_reproducible_and_loadable(tmp_path: Path) -> None:
    first = run_demo(tmp_path / "first", seed=1234)
    second = run_demo(tmp_path / "second", seed=1234)
    assert first.finding_count > 4
    assert first.reproduced_regressions == first.finding_count
    first_report = Path(first.report_path).read_bytes()
    second_report = Path(second.report_path).read_bytes()
    assert first_report == second_report
    assert Path(first.corpus_path).read_bytes() == Path(second.corpus_path).read_bytes()
    assert first_report.endswith(b"\n")

    report_document = json.loads(first_report)
    assert report_document["seed"] == 1234
    corpus = RegressionCorpus.model_validate_json(Path(first.corpus_path).read_bytes(), strict=True)
    assert corpus.corpus_digest
    tampered_case = corpus.cases[0].model_copy(update={"violated_contract": "tampered.contract"})
    with pytest.raises(ValidationError, match="digest"):
        RegressionCorpus(
            seed=corpus.seed,
            cases=(tampered_case, *corpus.cases[1:]),
            constraints=corpus.constraints,
            corpus_digest=corpus.corpus_digest,
        )
    loaded = load_counterexample(Path(first.counterexample_paths[0]))
    assert loaded.counterexample_id.startswith("cex-")
    with pytest.raises(FileExistsError, match="must be empty"):
        run_demo(tmp_path / "first", seed=1234)


def test_redteam_models_reject_untyped_extensions() -> None:
    document = RedTeamConfiguration(seed=1).model_dump(mode="json")
    document["unexpected"] = True
    with pytest.raises(ValidationError):
        RedTeamConfiguration.model_validate(document, strict=True)
