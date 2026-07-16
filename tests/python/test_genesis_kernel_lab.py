from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sloforge.genesis.kernel_lab import (
    AcceptanceStatus,
    AdapterStatus,
    AttributionScope,
    BottleneckEvidence,
    BottleneckEvidenceError,
    EvidenceSource,
    KernelBenchmarkConfig,
    LabStatus,
    benchmark_candidate,
    benchmark_generated_runtime_impact,
    decide_candidate,
    derive_runtime_impact_config,
    execute_correctness,
    generate_candidates,
    generate_correctness_cases,
    hybrid_quantized_state_schema,
    run_kernel_lab_demo,
    triton_adapter_status,
    validate_benchmark_report,
    validate_bottleneck_evidence,
    validate_correctness_evidence,
    validate_generated_source,
    validate_runtime_impact_report,
)

SEED = 73129


def test_correctness_validator_reopens_retained_runner_output(tmp_path: Path) -> None:
    candidate, source = generate_candidates(seed=SEED)[0]
    evidence = execute_correctness(
        candidate,
        source,
        generate_correctness_cases(seed=SEED, randomized_cases=1),
        output_root=tmp_path / "correctness",
        seed=SEED,
    )
    validate_correctness_evidence(evidence)
    assert evidence.runner_output_path is not None
    output = Path(evidence.runner_output_path)
    output.write_bytes(output.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after execution"):
        validate_correctness_evidence(evidence)


def _evidence(tmp_path: Path, *, observed_fraction: float = 0.18) -> BottleneckEvidence:
    raw = tmp_path / "autopsy-operator-attribution.json"
    operator_time = round(observed_fraction * 1_000_000)
    comparison_time = 1_000_000 - operator_time
    raw.write_text(
        json.dumps(
            {
                "operator_id": "quantized-state-update",
                "inclusive_cpu_time_ns": [operator_time] * 7,
                "comparison_work_time_ns": [comparison_time] * 7,
                "operator_probe_fraction": observed_fraction,
                "attribution_scope": "measured_synthetic_operator_microprobe",
                "seed": SEED,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return BottleneckEvidence(
        evidence_id="cpu-microprobe-hybrid-qstate",
        source=EvidenceSource.CPU_PROFILE_MEASURED,
        attribution_scope=AttributionScope.SYNTHETIC_OPERATOR_MICROPROBE,
        causal_attribution=False,
        operator_id="quantized-state-update",
        genome_region="state.quantized_state",
        observed_fraction=observed_fraction,
        sample_count=7,
        deterministic_seed=SEED,
        raw_evidence_path=str(raw),
        raw_evidence_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        hardware_fingerprint="cpu-local-test",
        workload_fingerprint="hybrid-agentic-mixed",
        synthetic=True,
    )


def _small_benchmark_config() -> KernelBenchmarkConfig:
    return KernelBenchmarkConfig(
        deterministic_seed=SEED,
        warmup_count=1,
        repetitions=7,
        micro_iterations=8,
        token_loop_iterations=2,
        token_steps=3,
        bootstrap_rounds=100,
        wall_time_seconds=10.0,
    )


def test_operator_schema_declares_full_exact_domain() -> None:
    schema = hybrid_quantized_state_schema()
    assert schema.operator_id == "quantized-state-update"
    assert schema.inputs[0].dtype == "int8"
    assert schema.inputs[1].dtype == "float64"
    assert schema.inputs[0].allowed_strides == (1, 2, 3)
    assert not schema.inputs[0].contiguous_required
    assert schema.aliasing.output_may_alias == ("previous_state",)
    assert schema.aliasing.exact_index_mapping_required
    assert schema.numerical.rounding == "nearest_ties_to_even"
    assert schema.numerical.nan_behavior == "reject"
    assert schema.numerical.infinity_behavior == "reject"
    assert schema.numerical.maximum_absolute_error == 0
    assert schema.architecture.architectures == ("cpu-generic",)
    assert schema.architecture.hidden_fallback_forbidden


def test_candidate_generation_is_seeded_hashed_and_restricted() -> None:
    first = generate_candidates(seed=SEED)
    second = generate_candidates(seed=SEED)
    assert first == second
    assert len(first) == 12
    assert len({candidate.candidate_id for candidate, _ in first}) == 12
    for candidate, source in first:
        assert hashlib.sha256(source.encode()).hexdigest() == candidate.source_sha256
        assert validate_generated_source(candidate, source) == ()
    candidate, source = first[0]
    diagnostics = validate_generated_source(candidate, source + "\nopen('/tmp/escape', 'w')\n")
    assert "source_digest_mismatch" in diagnostics
    assert "forbidden_name:open" in diagnostics
    monkeypatch_source = source + "\nmath.isfinite = lambda value: True\n"
    restricted = validate_generated_source(
        candidate.model_copy(
            update={"source_sha256": hashlib.sha256(monkeypatch_source.encode()).hexdigest()}
        ),
        monkeypatch_source,
    )
    assert "forbidden_attribute_access" in restricted
    assert "forbidden_syntax:Lambda" in restricted


def test_bottleneck_targeting_requires_real_immutable_evidence(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    validate_bottleneck_evidence(evidence)
    evidence_path = Path(evidence.raw_evidence_path)
    evidence_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(BottleneckEvidenceError, match="digest mismatch"):
        validate_bottleneck_evidence(evidence)
    below_threshold = _evidence(tmp_path, observed_fraction=0.01)
    with pytest.raises(BottleneckEvidenceError, match="below"):
        validate_bottleneck_evidence(below_threshold)


def test_cases_cover_stride_alias_boundary_and_nonfinite_domains() -> None:
    cases = generate_correctness_cases(seed=SEED, randomized_cases=10)
    categories = {case.category for case in cases}
    assert {"edge", "random", "noncontiguous", "nonfinite", "alias"}.issubset(categories)
    assert any(case.previous_stride > 1 for case in cases)
    assert any(case.activation_stride > 1 for case in cases)
    assert any(case.output_alias_previous for case in cases)
    assert sum(case.expected_error == "ValueError" for case in cases) == 4
    overflow = next(case for case in cases if case.case_id == "edge-finite-overflow")
    assert overflow.expected == (127, -127)
    atomic = next(case for case in cases if case.case_id == "alias-invalid-late-is-atomic")
    assert atomic.output_alias_previous
    assert atomic.expected_error == "ValueError"
    assert any(case.count == 1 for case in cases)
    assert any(case.count == 32 for case in cases)
    enumerated = {
        value
        for case in cases
        if case.case_id.startswith("edge-int8-domain")
        for value in case.previous_storage
    }
    assert enumerated == set(range(-127, 128))


def test_generated_candidate_executes_only_in_sandbox_and_passes_correctness(
    tmp_path: Path,
) -> None:
    candidate, source = generate_candidates(seed=SEED)[0]
    cases = generate_correctness_cases(seed=SEED, randomized_cases=12)
    result = execute_correctness(
        candidate,
        source,
        cases,
        output_root=tmp_path / "correctness",
        seed=SEED,
        timeout_seconds=10.0,
    )
    assert result.status is LabStatus.PASSED
    assert result.source_allowlist_valid
    assert result.sandbox_termination == "success"
    assert result.sandbox_backend != "not_run"
    assert result.cases_executed == len(cases)
    assert result.noncontiguous_cases >= 1
    assert result.nonfinite_cases == 3
    assert result.alias_cases >= 1
    assert not result.mismatches


def test_correctness_gate_rejects_partial_alias_mutation_on_late_error(
    tmp_path: Path,
) -> None:
    candidate, source = generate_candidates(seed=SEED)[0]
    unsafe = source.replace(
        "            logical[position] = rounded\n",
        "            if output_alias_previous:\n"
        "                previous_storage[previous_offset + position * previous_stride] = rounded\n"
        "            logical[position] = rounded\n",
    ).replace(
        "    if output_alias_previous:\n"
        "        for position in range(count):\n"
        "            previous_storage[previous_offset + position * previous_stride] = logical[position]\n",
        "",
    )
    unsafe_candidate = candidate.model_copy(
        update={"source_sha256": hashlib.sha256(unsafe.encode()).hexdigest()}
    )
    result = execute_correctness(
        unsafe_candidate,
        unsafe,
        generate_correctness_cases(seed=SEED, randomized_cases=1),
        output_root=tmp_path / "unsafe-alias",
        seed=SEED,
        timeout_seconds=10.0,
    )

    assert result.status is LabStatus.FAILED
    assert any(item.category == "alias_atomicity" for item in result.mismatches)


def test_static_rejection_prevents_generated_code_execution(tmp_path: Path) -> None:
    candidate, source = generate_candidates(seed=SEED)[0]
    result = execute_correctness(
        candidate,
        source + "\nopen('/tmp/escape', 'w')\n",
        generate_correctness_cases(seed=SEED, randomized_cases=1),
        output_root=tmp_path / "rejected",
        seed=SEED,
    )
    assert result.status is LabStatus.FAILED
    assert not result.source_allowlist_valid
    assert result.sandbox_termination == "not_run_static_rejection"
    assert not (tmp_path / "rejected").exists()


def test_cpu_benchmark_preserves_raw_samples_ci_order_and_end_to_end(
    tmp_path: Path,
) -> None:
    candidate, source = generate_candidates(seed=SEED)[0]
    correctness = execute_correctness(
        candidate,
        source,
        generate_correctness_cases(seed=SEED, randomized_cases=4),
        output_root=tmp_path / "correctness",
        seed=SEED,
    )
    config = _small_benchmark_config()
    report = benchmark_candidate(
        candidate,
        source,
        correctness,
        output_root=tmp_path / "benchmark",
        config=config,
    )
    assert report.sandbox_termination == "success"
    assert len(report.raw_samples) == 3 * 2 * config.repetitions
    assert {item.regime for item in report.regimes} == {
        "micro_batch_1",
        "micro_noncontiguous",
        "operator_loop_batch_32",
    }
    for regime in report.regimes:
        assert regime.warmup_count == config.warmup_count
        assert len(regime.reference_samples_ns) == config.repetitions
        assert len(regime.candidate_samples_ns) == config.repetitions
        assert len(regime.run_order) == config.repetitions * 2
        assert {regime.run_order[index] for index in range(len(regime.run_order))} == {
            "reference",
            "candidate",
        }
        assert regime.confidence_interval_low_percent <= regime.confidence_interval_high_percent
    decision = decide_candidate(correctness, report)
    assert decision.status is not AcceptanceStatus.ACCEPTED
    assert decision.full_stack_status is LabStatus.UNAVAILABLE
    assert decision.claim.startswith("no speedup claim")
    validate_benchmark_report(report)


def test_operator_loop_status_cannot_be_altered_to_change_decision(tmp_path: Path) -> None:
    candidate, source = generate_candidates(seed=SEED)[0]
    correctness = execute_correctness(
        candidate,
        source,
        generate_correctness_cases(seed=SEED, randomized_cases=1),
        output_root=tmp_path / "correctness",
        seed=SEED,
    )
    report = benchmark_candidate(
        candidate,
        source,
        correctness,
        output_root=tmp_path / "benchmark",
        config=_small_benchmark_config(),
    )
    regimes = tuple(
        item.model_copy(
            update={
                "status": (
                    LabStatus.FAILED if item.status is not LabStatus.FAILED else LabStatus.PASSED
                )
            }
        )
        if item.regime == "operator_loop_batch_32"
        else item
        for item in report.regimes
    )
    inconclusive = report.model_copy(update={"status": LabStatus.INCONCLUSIVE, "regimes": regimes})
    with pytest.raises(ValueError, match="not derived"):
        decide_candidate(correctness, inconclusive)


def test_benchmark_recomputation_rejects_tampered_claims(tmp_path: Path) -> None:
    candidate, source = generate_candidates(seed=SEED)[0]
    correctness = execute_correctness(
        candidate,
        source,
        generate_correctness_cases(seed=SEED, randomized_cases=1),
        output_root=tmp_path / "correctness",
        seed=SEED,
    )
    report = benchmark_candidate(
        candidate,
        source,
        correctness,
        output_root=tmp_path / "benchmark",
        config=_small_benchmark_config(),
    )
    validate_benchmark_report(report)
    first = report.regimes[0].model_copy(
        update={"improvement_percent": report.regimes[0].improvement_percent + 50.0}
    )
    tampered = report.model_copy(update={"regimes": (first, *report.regimes[1:])})
    with pytest.raises(ValueError, match="not derived"):
        validate_benchmark_report(tampered)


def test_microbenchmark_status_cannot_be_altered_to_change_decision(tmp_path: Path) -> None:
    candidate, source = generate_candidates(seed=SEED)[0]
    correctness = execute_correctness(
        candidate,
        source,
        generate_correctness_cases(seed=SEED, randomized_cases=1),
        output_root=tmp_path / "correctness",
        seed=SEED,
    )
    report = benchmark_candidate(
        candidate,
        source,
        correctness,
        output_root=tmp_path / "benchmark",
        config=_small_benchmark_config(),
    )
    regimes = tuple(
        item.model_copy(
            update={
                "status": (
                    LabStatus.FAILED if item.status is not LabStatus.FAILED else LabStatus.PASSED
                )
            }
        )
        if item.regime == "micro_batch_1"
        else item
        for item in report.regimes
    )
    inconclusive = report.model_copy(update={"status": LabStatus.INCONCLUSIVE, "regimes": regimes})
    with pytest.raises(ValueError, match="not derived"):
        decide_candidate(correctness, inconclusive)


def test_triton_adapter_is_fail_closed_and_never_claims_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLOFORGE_GENESIS_ALLOW_GPU", raising=False)
    report = triton_adapter_status()
    assert report.status in {AdapterStatus.UNAVAILABLE, AdapterStatus.UNEXERCISED}
    assert not report.exercised
    assert "unexercised" in report.reason or "not installed" in report.reason


def test_demo_writes_source_raw_negative_and_aggregate_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "generated-kernels"
    report = run_kernel_lab_demo(
        evidence=_evidence(tmp_path),
        output_root=output,
        seed=SEED,
        candidate_limit=1,
        benchmark_config=_small_benchmark_config(),
    )
    assert len(report.candidates) == 1
    candidate_id = report.candidates[0].candidate_id
    candidate_root = output / "candidates" / candidate_id
    assert (candidate_root / "correctness" / "source" / "candidate.py").is_file()
    assert (candidate_root / "raw_samples.jsonl").is_file()
    raw_lines = (candidate_root / "raw_samples.jsonl").read_text().splitlines()
    assert len(raw_lines) == len(report.benchmarks[0].raw_samples)
    assert (
        hashlib.sha256((candidate_root / "raw_samples.jsonl").read_bytes()).hexdigest()
        == report.benchmarks[0].raw_samples_sha256
    )
    assert json.loads((output / "kernel_lab_report.json").read_text())["evidence"]
    assert len(report.runtime_impacts) == 1
    impact = report.runtime_impacts[0]
    assert impact.measurement_scope == "cpu_generated_runtime_end_to_end_serving"
    assert impact.timing_boundary.endswith("through_state_release")
    assert impact.hardware_backed_gpu is False
    assert impact.output_exact_match
    assert impact.state_exact_match
    assert impact.statistics.analysis_method == "paired_trial_bootstrap_median_improvement"
    assert impact.validation is not None
    assert impact.validation.runtime_outputs_replayed
    assert impact.validation.state_semantics_replayed
    assert len(impact.samples) == 14
    assert {item.alternative for item in impact.samples} == {"reference", "candidate"}
    assert impact.source_package_hash != impact.patched_package_hash
    assert {item.package_hash for item in impact.runtime_bundles} == {
        impact.source_package_hash,
        impact.patched_package_hash,
    }
    assert (
        len(
            {
                impact.config.synthesis_seed,
                impact.config.runtime_generation_seed,
                impact.config.trace_seed,
                impact.config.trial_order_seed,
                impact.config.bootstrap_seed,
                impact.config.sandbox_seed,
            }
        )
        == 6
    )
    impact_root = candidate_root / "runtime-impact"
    validate_runtime_impact_report(impact, artifact_root=impact_root)

    package_link = tmp_path / "flagship-package-link"
    package_link.symlink_to(
        Path(__file__).resolve().parents[2] / "models/reference_tasks/hybrid_decoder",
        target_is_directory=True,
    )
    generated_candidate, generated_source = generate_candidates(seed=SEED)[0]
    with pytest.raises(ValueError, match="source package must not be a symlink"):
        benchmark_generated_runtime_impact(
            generated_candidate,
            generated_source,
            report.correctness[0],
            reference_package_root=package_link,
            output_root=tmp_path / "symlink-runtime-impact",
            config=derive_runtime_impact_config(SEED),
        )
    raw_runtime_path = Path(impact.raw_samples_path)
    raw_runtime = raw_runtime_path.read_bytes()
    raw_runtime_path.write_bytes(raw_runtime + b"{}\n")
    with pytest.raises(ValueError, match="raw samples changed"):
        validate_runtime_impact_report(impact, artifact_root=impact_root)
    raw_runtime_path.write_bytes(raw_runtime)
    validate_runtime_impact_report(impact, artifact_root=impact_root)

    escaped = impact.model_copy(update={"raw_samples_path": "/etc/hosts"})
    with pytest.raises(ValueError, match="escapes its trusted root"):
        validate_runtime_impact_report(escaped, artifact_root=impact_root)

    runner_path = Path(impact.runner_path)
    runner_source = runner_path.read_bytes()
    runner_path.write_bytes(runner_source + b"\n# forged runner\n")
    with pytest.raises(ValueError, match=r"installed trusted runner|trusted runner changed"):
        validate_runtime_impact_report(impact, artifact_root=impact_root)
    runner_path.write_bytes(runner_source)

    candidate_identity = next(
        item for item in impact.runtime_bundles if item.alternative == "candidate"
    )
    wrapper_path = Path(candidate_identity.package_root) / "reference.py"
    wrapper_source = wrapper_path.read_bytes()
    wrapper_path.write_bytes(wrapper_source + b"\n# forged wrapper\n")
    with pytest.raises(ValueError, match="package hash changed"):
        validate_runtime_impact_report(impact, artifact_root=impact_root)
    wrapper_path.write_bytes(wrapper_source)

    runtime_path = Path(candidate_identity.bundle_root) / "runtime.py"
    runtime_source = runtime_path.read_bytes()
    runtime_path.write_bytes(runtime_source + b"\n# forged runtime\n")
    with pytest.raises(ValueError, match="generated runtime artifact changed"):
        validate_runtime_impact_report(impact, artifact_root=impact_root)
    runtime_path.write_bytes(runtime_source)

    symlink_backup = raw_runtime_path.with_name("raw-runtime-samples.backup")
    raw_runtime_path.rename(symlink_backup)
    raw_runtime_path.symlink_to(symlink_backup.name)
    with pytest.raises(ValueError, match="contains a symlink"):
        validate_runtime_impact_report(impact, artifact_root=impact_root)
    raw_runtime_path.unlink()
    symlink_backup.rename(raw_runtime_path)

    measured_path = Path(impact.runner_output_path)
    measured_source = measured_path.read_bytes()
    measured = json.loads(measured_source)
    measured["samples"][0]["output_sha256"] = "0" * 64
    measured_path.write_text(
        json.dumps(measured, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    forged_sample = impact.samples[0].model_copy(update={"output_sha256": "0" * 64})
    forged_samples = (forged_sample, *impact.samples[1:])
    forged_raw = b"".join(
        json.dumps(item.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for item in forged_samples
    )
    raw_runtime_path.write_bytes(forged_raw)
    forged_output = impact.model_copy(
        update={
            "samples": forged_samples,
            "raw_samples_sha256": hashlib.sha256(forged_raw).hexdigest(),
            "runner_output_sha256": hashlib.sha256(measured_path.read_bytes()).hexdigest(),
        }
    )
    with pytest.raises(ValueError, match="output digest is not observation-derived"):
        validate_runtime_impact_report(forged_output, artifact_root=impact_root)
    measured_path.write_bytes(measured_source)
    raw_runtime_path.write_bytes(raw_runtime)
    validate_runtime_impact_report(impact, artifact_root=impact_root)
    assert report.decisions[0].claim in {
        "no speedup claim; end-to-end generated-runtime acceptance gate did not pass",
        "CPU generated-runtime speedup accepted within the declared local evidence scope",
    }
    with pytest.raises(FileExistsError, match="must be empty"):
        run_kernel_lab_demo(
            evidence=_evidence(tmp_path),
            output_root=output,
            seed=SEED,
            candidate_limit=1,
            benchmark_config=_small_benchmark_config(),
        )
