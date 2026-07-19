from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.analysis.statistics import (
    MAX_BOOTSTRAP_DRAWS,
    ArtifactEvidence,
    BootstrapStatistic,
    RawSampleSeries,
    WarmupPolicy,
    bootstrap_confidence_interval,
    correlation,
    empirical_cdf,
    histogram,
    identify_mad_outliers,
    noise_floor,
    paired_effect_size,
    summarize,
)
from sloforge.helix.characterization.matrix import EvidenceClass


def _series(
    values: tuple[float, ...],
    *,
    series_id: str = "run-41.branch-latency",
    warmup: tuple[float, ...] = (),
    metric: str = "branch_latency",
    unit: str = "ms",
) -> RawSampleSeries:
    return RawSampleSeries(
        schema_version="sloforge.branchfabric.raw-sample-series/v1",
        series_id=series_id,
        metric=metric,
        unit=unit,
        provenance=ArtifactEvidence(
            schema_version="sloforge.branchfabric.artifact-evidence/v1",
            source_experiment="experiment-41",
            artifact_reference="artifacts/branchfabric/raw/experiment-41.jsonl",
            artifact_sha256="a" * 64,
            evidence_class=EvidenceClass.SYNTHETIC,
            sample_selector="events[type=BRANCH_READY].duration_ms",
            sample_count=len(warmup) + len(values),
            seed=41,
            repetition=0,
        ),
        warmup_policy=WarmupPolicy(
            method="fixed_count" if warmup else "none",
            declared_warmup_count=len(warmup),
            rationale="first fixture iteration is the declared warmup" if warmup else "no warmup",
        ),
        warmup_samples=warmup,
        samples=values,
    )


def test_summary_preserves_raw_samples_and_excludes_declared_warmup() -> None:
    series = _series((1.0, 2.0, 3.0, 4.0, 100.0), warmup=(9999.0,))

    result = summarize(series)

    assert series.warmup_samples == (9999.0,)
    assert series.samples == (1.0, 2.0, 3.0, 4.0, 100.0)
    assert result.warmup_count == 1
    assert result.sample_count == 5
    assert result.median == 3.0
    assert result.p95 == pytest.approx(80.8)
    assert result.maximum == 100.0
    assert result.outlier_policy == "none removed"
    assert result.provenance.artifact_sha256 == "a" * 64


def test_raw_series_rejects_provenance_or_warmup_count_mismatch() -> None:
    with pytest.raises(ValidationError, match="warmup policy count"):
        RawSampleSeries(
            schema_version="sloforge.branchfabric.raw-sample-series/v1",
            series_id="bad-series",
            metric="latency",
            unit="ms",
            provenance=ArtifactEvidence(
                schema_version="sloforge.branchfabric.artifact-evidence/v1",
                source_experiment="bad-experiment",
                artifact_reference="raw.jsonl",
                artifact_sha256="b" * 64,
                evidence_class=EvidenceClass.REPLAYED,
                sample_selector="duration_ms",
                sample_count=2,
                seed=7,
                repetition=0,
            ),
            warmup_policy=WarmupPolicy(
                method="fixed_count",
                declared_warmup_count=0,
                rationale="invalid test fixture",
            ),
            warmup_samples=(1.0,),
            samples=(2.0,),
        )


def test_bootstrap_is_seeded_reproducible_and_bounded() -> None:
    series = _series((1.0, 2.0, 3.0, 4.0, 5.0))

    first = bootstrap_confidence_interval(
        series, BootstrapStatistic.MEDIAN, seed=73, repetitions=500
    )
    second = bootstrap_confidence_interval(
        series, BootstrapStatistic.MEDIAN, seed=73, repetitions=500
    )

    assert first == second
    assert first.lower <= first.observed <= first.upper
    assert first.seed == 73
    assert first.method == "deterministic percentile bootstrap"

    bounded_series = _series(tuple(float(index) for index in range(1000)))
    repetitions = MAX_BOOTSTRAP_DRAWS // len(bounded_series.samples) + 1
    with pytest.raises(ValueError, match="draw safety bound"):
        bootstrap_confidence_interval(
            bounded_series, BootstrapStatistic.MEAN, seed=73, repetitions=repetitions
        )


def test_ecdf_and_histogram_preserve_all_measurement_samples() -> None:
    series = _series(tuple(float(index) for index in range(10)))

    exact = empirical_cdf(series, max_points=10)
    compressed = empirical_cdf(series, max_points=4)
    distribution = histogram(series, bins=3)

    assert exact.exact is True
    assert len(exact.points) == 10
    assert exact.points[-1].cumulative_probability == 1.0
    assert compressed.exact is False
    assert compressed.method == "deterministic rank grid"
    assert len(compressed.points) <= 4
    assert compressed.points[-1].cumulative_probability == 1.0
    assert sum(distribution.counts) == 10
    assert len(distribution.bin_edges) == 4


def test_constant_histogram_and_noise_floor_have_defined_behavior() -> None:
    series = _series((5.0, 5.0, 5.0))

    distribution = histogram(series, bins=100)
    noise = noise_floor(series)

    assert distribution.counts == (3,)
    assert distribution.bin_edges[0] < 5.0 < distribution.bin_edges[1]
    assert noise.median_absolute_deviation == 0.0
    assert noise.coefficient_of_variation == 0.0
    assert noise.relative_mad == 0.0


def test_mad_outliers_are_flagged_but_never_removed() -> None:
    series = _series((9.0, 10.0, 10.0, 11.0, 100.0))

    report = identify_mad_outliers(series)

    assert report.sample_count == len(series.samples)
    assert report.samples_removed == 0
    assert [flag.sample_index for flag in report.flags] == [4]
    assert report.flags[0].value == 100.0
    assert report.flags[0].modified_z_score is not None


def test_zero_mad_flags_nonmedian_values_without_infinite_scores() -> None:
    series = _series((10.0, 10.0, 10.0, 11.0))

    report = identify_mad_outliers(series)

    assert len(report.flags) == 1
    assert report.flags[0].modified_z_score is None
    assert report.flags[0].reason == "non-median value with zero MAD"


def test_paired_effect_size_and_correlation_preserve_pair_order() -> None:
    baseline = _series((10.0, 20.0, 30.0), series_id="baseline")
    candidate = _series((11.0, 22.0, 33.0), series_id="candidate")

    effect = paired_effect_size(baseline, candidate)
    association = correlation(baseline, candidate)

    assert effect.mean_difference == 2.0
    assert effect.median_difference == 2.0
    assert effect.paired_cohens_dz == 2.0
    assert effect.matched_pairs_rank_biserial == 1.0
    assert effect.median_relative_change == pytest.approx(0.1)
    assert effect.relative_denominator == "absolute baseline"
    assert effect.direction == "candidate minus baseline"
    assert association.pearson_r == pytest.approx(1.0)
    assert association.spearman_rho == pytest.approx(1.0)

    different_metric = _series((11.0, 22.0, 33.0), series_id="throughput", metric="throughput")
    with pytest.raises(ValueError, match="identical metrics"):
        paired_effect_size(baseline, different_metric)


def test_constant_series_correlation_is_explicitly_undefined() -> None:
    constant = _series((2.0, 2.0, 2.0), series_id="constant")
    varying = _series((1.0, 2.0, 3.0), series_id="varying")

    result = correlation(constant, varying)

    assert result.pearson_r is None
    assert result.spearman_rho is None
    assert result.constant_series_policy == "undefined correlation is null"


def test_nonfinite_samples_and_excluding_histogram_bounds_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _series((1.0, math.inf))
    series = _series((1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="cannot exclude"):
        histogram(series, bins=2, lower=1.5)
