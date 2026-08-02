from __future__ import annotations

import numpy as np
import pytest

from sloforge.genesis.verification import (
    BenchmarkContract,
    EvidenceStatus,
    MetricDirection,
    OperatorContract,
    QualityContract,
    ResourceContract,
    ResourceDemand,
    ShapeBound,
    VerificationError,
    analyze_resources,
    evaluate_performance,
    evaluate_quality,
    verify_operator,
)


def _operator_contract() -> OperatorContract:
    return OperatorContract(
        operator_id="state_update",
        shape=(ShapeBound(1, 5, "batch"), ShapeBound(2, 7, "hidden")),
        dtypes=("float32", "int32"),
        allow_non_contiguous=True,
        allow_aliasing=False,
        exact=True,
        maximum_absolute_error=0.0,
        maximum_relative_error=0.0,
        preserve_nan=True,
        preserve_infinity=True,
        preserve_signed_zero=True,
        deterministic=True,
        maximum_cases=24,
    )


def test_operator_verifier_passes_and_is_seeded() -> None:
    def reference(value: np.ndarray) -> np.ndarray:
        return value * 2

    result = verify_operator(reference, reference, _operator_contract(), seed=73129)
    assert result.status is EvidenceStatus.PASSED
    assert result.cases_executed == 24
    assert result == verify_operator(reference, reference, _operator_contract(), seed=73129)


def test_high_performing_but_wrong_candidate_gets_real_counterexample() -> None:
    def reference(value: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> np.ndarray:
        return value * 2

    def rare_shape_bug(value: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> np.ndarray:
        result = value * 2
        if value.shape[-1] == 7:
            result = result.copy()
            result.reshape(-1)[-1] = 0
        return result

    verification = verify_operator(reference, rare_shape_bug, _operator_contract(), seed=5)
    assert verification.status is EvidenceStatus.FAILED
    assert verification.counterexample is not None
    assert verification.counterexample.shape[-1] == 7
    assert verification.counterexample.violation == "exact_value"


def test_operator_verifier_records_candidate_exception_as_counterexample() -> None:
    def reference(value: np.ndarray) -> np.ndarray:
        return value * 2

    def candidate(value: np.ndarray) -> np.ndarray:
        if value.shape[-1] == 7:
            raise RuntimeError("rare shape")
        return value * 2

    verification = verify_operator(reference, candidate, _operator_contract(), seed=5)
    assert verification.status is EvidenceStatus.FAILED
    assert verification.counterexample is not None
    assert verification.counterexample.violation == "candidate_exception:RuntimeError"


def test_operator_verifier_detects_signed_zero_contract_violation() -> None:
    def reference(value: np.ndarray) -> np.ndarray:
        return value

    def loses_negative_zero(value: np.ndarray) -> np.ndarray:
        result = value.copy()
        result[(result == 0) & np.signbit(result)] = 0.0
        return result

    verification = verify_operator(reference, loses_negative_zero, _operator_contract(), seed=5)
    assert verification.status is EvidenceStatus.FAILED
    assert verification.counterexample is not None
    assert verification.counterexample.violation == "signed_zero_behavior"


def test_quality_contract_detects_distribution_and_token_regression() -> None:
    reference = np.asarray([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    candidate = np.asarray([[1.0, 4.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    evidence = evaluate_quality(
        reference,
        candidate,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 1], dtype=np.int64),
        QualityContract(1.0, 1.0, 2, 1.0, 0.01, 0.01, 0.1),
    )
    assert evidence.status is EvidenceStatus.FAILED
    assert {"exact_token_match", "top1_agreement", "kl_divergence"}.issubset(evidence.violations)


def test_quality_contract_rejects_nonfinite_logits_and_invalid_tokens() -> None:
    contract = QualityContract(1.0, 1.0, 1, 1.0, 0.1, 0.1, 0.1)
    with pytest.raises(VerificationError, match="finite"):
        evaluate_quality(
            np.asarray([[float("nan"), 0.0]], dtype=np.float64),
            np.asarray([[0.0, 0.0]], dtype=np.float64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            contract,
        )
    with pytest.raises(VerificationError, match="vocabulary"):
        evaluate_quality(
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            np.asarray([2], dtype=np.int64),
            np.asarray([2], dtype=np.int64),
            contract,
        )


def test_resource_gate_includes_challenger_conversion_and_fragmentation() -> None:
    evidence = analyze_resources(
        ResourceContract(1000, 1000, 0.1, 0.1, 4, 16, 128),
        ResourceDemand(300, 100, 20, 40, 40, 300, 200, 100, 2, 8, 32),
    )
    assert evidence.status is EvidenceStatus.FAILED
    assert "peak_device_memory" in evidence.violations


def test_performance_gate_requires_raw_repeated_significant_evidence() -> None:
    contract = BenchmarkContract(
        "kernel-state-update",
        "latency_us",
        "us",
        MetricDirection.LOWER_IS_BETTER,
        "workload-sha256",
        "hardware-sha256",
        "software-sha256",
        warmup_count=3,
        practical_significance_percent=5.0,
        noise_floor_percent=2.0,
        bootstrap_rounds=500,
        confidence=0.95,
    )
    accepted = evaluate_performance(
        contract,
        (100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8, 100.1, 99.9),
        (80.0, 81.0, 79.0, 80.5, 79.5, 80.2, 79.8, 80.1, 79.9),
        seed=41,
        run_order=("baseline", "candidate") * 9,
    )
    assert accepted.status is EvidenceStatus.PASSED
    assert accepted.interval_low_percent > 5
    noisy = evaluate_performance(
        contract,
        (100.0, 90.0, 110.0, 95.0, 105.0, 92.0, 108.0),
        (98.0, 91.0, 109.0, 94.0, 104.0, 93.0, 107.0),
        seed=41,
        run_order=("candidate", "baseline") * 7,
    )
    assert noisy.status is EvidenceStatus.INCONCLUSIVE
    with pytest.raises(VerificationError, match="seven"):
        evaluate_performance(
            contract,
            (1.0,),
            (0.5,),
            seed=1,
            run_order=("baseline", "candidate"),
        )


def test_performance_gate_rejects_fabricated_order_and_nonfinite_samples() -> None:
    contract = BenchmarkContract(
        "kernel-state-update",
        "latency_us",
        "us",
        MetricDirection.LOWER_IS_BETTER,
        "workload-sha256",
        "hardware-sha256",
        "software-sha256",
        warmup_count=1,
        practical_significance_percent=1.0,
        noise_floor_percent=0.5,
        bootstrap_rounds=100,
        confidence=0.95,
    )
    baseline = (10.0,) * 7
    candidate = (9.0,) * 7
    with pytest.raises(VerificationError, match="run order"):
        evaluate_performance(
            contract,
            baseline,
            candidate,
            seed=7,
            run_order=("baseline",) * 7 + ("candidate",) * 6,
        )
    with pytest.raises(VerificationError, match="finite and positive"):
        evaluate_performance(
            contract,
            (*baseline[:-1], float("nan")),
            candidate,
            seed=7,
            run_order=("baseline", "candidate") * 7,
        )
