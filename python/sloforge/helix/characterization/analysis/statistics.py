"""Deterministic, provenance-preserving characterization statistics.

The models in this module keep raw warmup and measurement values intact.  The
analysis functions are pure: they neither rewrite source artifacts nor remove
outliers.  Every result retains the evidence reference used to derive it.
"""

from __future__ import annotations

import bisect
import math
import random
import statistics
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.helix.characterization.matrix import EvidenceClass

MAX_SAMPLES = 1_000_000
MAX_ECDF_POINTS = 100_000
MAX_HISTOGRAM_BINS = 4096
MAX_BOOTSTRAP_REPETITIONS = 100_000
MAX_BOOTSTRAP_DRAWS = 20_000_000

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class StatisticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ArtifactEvidence(StatisticsModel):
    """An immutable reference to the raw samples behind one series."""

    schema_version: Literal["sloforge.branchfabric.artifact-evidence/v1"]
    source_experiment: Identifier
    artifact_reference: NonEmpty
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_class: EvidenceClass
    sample_selector: NonEmpty
    sample_count: int = Field(ge=1, le=MAX_SAMPLES)
    seed: int = Field(ge=0, le=2**64 - 1)
    repetition: int = Field(ge=0, le=1_000_000)
    raw_artifact_immutable: Literal[True] = True


class WarmupPolicy(StatisticsModel):
    method: Literal["none", "fixed_count", "external_phase_marker"]
    declared_warmup_count: int = Field(ge=0, le=MAX_SAMPLES)
    rationale: NonEmpty
    excluded_from_derived_statistics: Literal[True] = True

    @model_validator(mode="after")
    def none_has_no_warmup(self) -> Self:
        if self.method == "none" and self.declared_warmup_count:
            raise ValueError("warmup method 'none' requires a zero warmup count")
        return self


class RawSampleSeries(StatisticsModel):
    """Ordered raw values with an explicit warmup/measurement boundary."""

    schema_version: Literal["sloforge.branchfabric.raw-sample-series/v1"]
    series_id: Identifier
    metric: Identifier
    unit: Identifier
    provenance: ArtifactEvidence
    warmup_policy: WarmupPolicy
    warmup_samples: Annotated[tuple[FiniteFloat, ...], Field(max_length=MAX_SAMPLES)] = ()
    samples: Annotated[tuple[FiniteFloat, ...], Field(min_length=1, max_length=MAX_SAMPLES)]
    sample_order: Literal["preserved"] = "preserved"

    @model_validator(mode="after")
    def counts_match_provenance(self) -> Self:
        total = len(self.warmup_samples) + len(self.samples)
        if total > MAX_SAMPLES:
            raise ValueError(f"raw series exceeds {MAX_SAMPLES} total samples")
        if self.warmup_policy.declared_warmup_count != len(self.warmup_samples):
            raise ValueError("warmup policy count does not match preserved warmup samples")
        if self.provenance.sample_count != total:
            raise ValueError("artifact evidence count does not match preserved raw samples")
        return self


class SummaryStatistics(StatisticsModel):
    schema_version: Literal["sloforge.branchfabric.summary-statistics/v1"]
    series_id: Identifier
    metric: Identifier
    unit: Identifier
    provenance: ArtifactEvidence
    warmup_count: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    minimum: FiniteFloat
    mean: FiniteFloat
    median: FiniteFloat
    p90: FiniteFloat
    p95: FiniteFloat
    p99: FiniteFloat
    maximum: FiniteFloat
    percentile_method: Literal["Hyndman-Fan type 7"]
    outlier_policy: Literal["none removed"]


class BootstrapStatistic(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    P95 = "p95"


class BootstrapConfidenceInterval(StatisticsModel):
    schema_version: Literal["sloforge.branchfabric.bootstrap-confidence-interval/v1"]
    series_id: Identifier
    metric: Identifier
    unit: Identifier
    provenance: ArtifactEvidence
    statistic: BootstrapStatistic
    observed: FiniteFloat
    lower: FiniteFloat
    upper: FiniteFloat
    confidence_level: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    repetitions: int = Field(ge=1, le=MAX_BOOTSTRAP_REPETITIONS)
    seed: int = Field(ge=0, le=2**64 - 1)
    method: Literal["deterministic percentile bootstrap"]
    percentile_method: Literal["Hyndman-Fan type 7"]
    sample_count: int = Field(gt=0)


class EmpiricalCdfPoint(StatisticsModel):
    value: FiniteFloat
    cumulative_probability: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)


class EmpiricalCdf(StatisticsModel):
    schema_version: Literal["sloforge.branchfabric.empirical-cdf/v1"]
    series_id: Identifier
    metric: Identifier
    unit: Identifier
    provenance: ArtifactEvidence
    sample_count: int = Field(gt=0)
    points: tuple[EmpiricalCdfPoint, ...] = Field(min_length=1, max_length=MAX_ECDF_POINTS)
    exact: bool
    method: Literal["exact unique values", "deterministic rank grid"]
    max_points: int = Field(ge=2, le=MAX_ECDF_POINTS)
    warmup_excluded: Literal[True] = True


class Histogram(StatisticsModel):
    schema_version: Literal["sloforge.branchfabric.histogram/v1"]
    series_id: Identifier
    metric: Identifier
    unit: Identifier
    provenance: ArtifactEvidence
    sample_count: int = Field(gt=0)
    bin_edges: tuple[FiniteFloat, ...] = Field(min_length=2, max_length=MAX_HISTOGRAM_BINS + 1)
    counts: tuple[int, ...] = Field(min_length=1, max_length=MAX_HISTOGRAM_BINS)
    method: Literal["equal width"]
    warmup_excluded: Literal[True] = True

    @model_validator(mode="after")
    def bins_are_consistent(self) -> Self:
        if len(self.bin_edges) != len(self.counts) + 1:
            raise ValueError("histogram edge and count lengths are inconsistent")
        if any(
            right <= left for left, right in zip(self.bin_edges, self.bin_edges[1:], strict=False)
        ):
            raise ValueError("histogram edges must increase strictly")
        if any(count < 0 for count in self.counts) or sum(self.counts) != self.sample_count:
            raise ValueError("histogram counts must preserve every measurement sample")
        return self


class NoiseFloor(StatisticsModel):
    schema_version: Literal["sloforge.branchfabric.noise-floor/v1"]
    series_id: Identifier
    metric: Identifier
    unit: Identifier
    provenance: ArtifactEvidence
    sample_count: int = Field(gt=0)
    median: FiniteFloat
    median_absolute_deviation: FiniteFloat
    p95_absolute_deviation: FiniteFloat
    population_standard_deviation: FiniteFloat
    coefficient_of_variation: FiniteFloat | None
    relative_mad: FiniteFloat | None
    relative_p95_absolute_deviation: FiniteFloat | None
    method: Literal["all measurement samples; no outlier removal"]


class OutlierFlag(StatisticsModel):
    sample_index: int = Field(ge=0, lt=MAX_SAMPLES)
    value: FiniteFloat
    modified_z_score: FiniteFloat | None
    reason: Literal["modified z-score threshold", "non-median value with zero MAD"]


class MadOutlierReport(StatisticsModel):
    schema_version: Literal["sloforge.branchfabric.mad-outliers/v1"]
    series_id: Identifier
    metric: Identifier
    unit: Identifier
    provenance: ArtifactEvidence
    sample_count: int = Field(gt=0)
    median: FiniteFloat
    median_absolute_deviation: FiniteFloat
    threshold: float = Field(gt=0.0, allow_inf_nan=False)
    flags: tuple[OutlierFlag, ...]
    samples_removed: Literal[0] = 0
    method: Literal["MAD modified z-score"]


class PairedEffectSize(StatisticsModel):
    schema_version: Literal["sloforge.branchfabric.paired-effect-size/v1"]
    baseline_series_id: Identifier
    candidate_series_id: Identifier
    baseline_provenance: ArtifactEvidence
    candidate_provenance: ArtifactEvidence
    metric: Identifier
    unit: Identifier
    pair_count: int = Field(gt=0)
    mean_difference: FiniteFloat
    median_difference: FiniteFloat
    paired_cohens_dz: FiniteFloat | None
    matched_pairs_rank_biserial: FiniteFloat | None
    median_relative_change: FiniteFloat | None
    relative_pairs_used: int = Field(ge=0)
    zero_baseline_pairs: int = Field(ge=0)
    relative_denominator: Literal["absolute baseline"]
    direction: Literal["candidate minus baseline"]
    pair_order: Literal["preserved by sample index"]


class CorrelationResult(StatisticsModel):
    schema_version: Literal["sloforge.branchfabric.correlation/v1"]
    x_series_id: Identifier
    y_series_id: Identifier
    x_provenance: ArtifactEvidence
    y_provenance: ArtifactEvidence
    x_metric: Identifier
    y_metric: Identifier
    x_unit: Identifier
    y_unit: Identifier
    pair_count: int = Field(gt=0)
    pearson_r: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    spearman_rho: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    ties_present: bool
    pair_order: Literal["preserved by sample index"]
    constant_series_policy: Literal["undefined correlation is null"]


def _percentile(values: tuple[float, ...] | list[float], probability: float) -> float:
    """Hyndman-Fan type 7 quantile, including exact endpoints."""

    if not values:
        raise ValueError("a percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def summarize(series: RawSampleSeries) -> SummaryStatistics:
    values = series.samples
    return SummaryStatistics(
        schema_version="sloforge.branchfabric.summary-statistics/v1",
        series_id=series.series_id,
        metric=series.metric,
        unit=series.unit,
        provenance=series.provenance,
        warmup_count=len(series.warmup_samples),
        sample_count=len(values),
        minimum=min(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        p90=_percentile(values, 0.90),
        p95=_percentile(values, 0.95),
        p99=_percentile(values, 0.99),
        maximum=max(values),
        percentile_method="Hyndman-Fan type 7",
        outlier_policy="none removed",
    )


def _evaluate_statistic(values: tuple[float, ...] | list[float], kind: BootstrapStatistic) -> float:
    if kind is BootstrapStatistic.MEAN:
        return statistics.fmean(values)
    if kind is BootstrapStatistic.MEDIAN:
        return statistics.median(values)
    return _percentile(values, 0.95)


def bootstrap_confidence_interval(
    series: RawSampleSeries,
    statistic: BootstrapStatistic,
    *,
    seed: int,
    repetitions: int = 2000,
    confidence_level: float = 0.95,
) -> BootstrapConfidenceInterval:
    if not 0 <= seed <= 2**64 - 1:
        raise ValueError("bootstrap seed must be an unsigned 64-bit integer")
    if not 1 <= repetitions <= MAX_BOOTSTRAP_REPETITIONS:
        raise ValueError(f"bootstrap repetitions must be in [1, {MAX_BOOTSTRAP_REPETITIONS}]")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    values = series.samples
    if len(values) * repetitions > MAX_BOOTSTRAP_DRAWS:
        raise ValueError(
            f"bootstrap requires more than the {MAX_BOOTSTRAP_DRAWS} draw safety bound"
        )
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        replicate = tuple(values[generator.randrange(len(values))] for _ in values)
        estimates.append(_evaluate_statistic(replicate, statistic))
    tail = (1.0 - confidence_level) / 2.0
    return BootstrapConfidenceInterval(
        schema_version="sloforge.branchfabric.bootstrap-confidence-interval/v1",
        series_id=series.series_id,
        metric=series.metric,
        unit=series.unit,
        provenance=series.provenance,
        statistic=statistic,
        observed=_evaluate_statistic(values, statistic),
        lower=_percentile(estimates, tail),
        upper=_percentile(estimates, 1.0 - tail),
        confidence_level=confidence_level,
        repetitions=repetitions,
        seed=seed,
        method="deterministic percentile bootstrap",
        percentile_method="Hyndman-Fan type 7",
        sample_count=len(values),
    )


def empirical_cdf(series: RawSampleSeries, *, max_points: int = 4096) -> EmpiricalCdf:
    if not 2 <= max_points <= MAX_ECDF_POINTS:
        raise ValueError(f"max_points must be in [2, {MAX_ECDF_POINTS}]")
    ordered = sorted(series.samples)
    unique: list[float] = []
    for value in ordered:
        if not unique or value != unique[-1]:
            unique.append(value)
            if len(unique) > max_points:
                break
    exact = len(unique) <= max_points
    method: Literal["exact unique values", "deterministic rank grid"]
    if exact:
        values = unique
        method = "exact unique values"
    else:
        indices = {
            round(ordinal * (len(ordered) - 1) / (max_points - 1)) for ordinal in range(max_points)
        }
        values = []
        for index in sorted(indices):
            value = ordered[index]
            if not values or value != values[-1]:
                values.append(value)
        method = "deterministic rank grid"
    points = tuple(
        EmpiricalCdfPoint(
            value=value,
            cumulative_probability=bisect.bisect_right(ordered, value) / len(ordered),
        )
        for value in values
    )
    return EmpiricalCdf(
        schema_version="sloforge.branchfabric.empirical-cdf/v1",
        series_id=series.series_id,
        metric=series.metric,
        unit=series.unit,
        provenance=series.provenance,
        sample_count=len(ordered),
        points=points,
        exact=exact,
        method=method,
        max_points=max_points,
    )


def histogram(
    series: RawSampleSeries,
    *,
    bins: int = 20,
    lower: float | None = None,
    upper: float | None = None,
) -> Histogram:
    if not 1 <= bins <= MAX_HISTOGRAM_BINS:
        raise ValueError(f"bins must be in [1, {MAX_HISTOGRAM_BINS}]")
    values = series.samples
    minimum = min(values)
    maximum = max(values)
    if lower is not None and not math.isfinite(lower):
        raise ValueError("lower histogram bound must be finite")
    if upper is not None and not math.isfinite(upper):
        raise ValueError("upper histogram bound must be finite")
    low = minimum if lower is None else lower
    high = maximum if upper is None else upper
    if low > minimum or high < maximum:
        raise ValueError("histogram bounds cannot exclude raw measurement samples")
    edges: tuple[float, ...]
    counts: tuple[int, ...]
    if low >= high:
        if minimum != maximum or lower is not None or upper is not None:
            raise ValueError("histogram lower bound must be less than upper bound")
        padding = max(0.5, abs(minimum) * 0.5)
        edges = (minimum - padding, minimum + padding)
        counts = (len(values),)
    else:
        width = (high - low) / bins
        edge_values = [low + width * index for index in range(bins + 1)]
        edge_values[-1] = high
        count_values = [0] * bins
        for value in values:
            index = bins - 1 if value == high else int((value - low) / width)
            index = max(0, min(index, bins - 1))
            count_values[index] += 1
        edges = tuple(edge_values)
        counts = tuple(count_values)
    return Histogram(
        schema_version="sloforge.branchfabric.histogram/v1",
        series_id=series.series_id,
        metric=series.metric,
        unit=series.unit,
        provenance=series.provenance,
        sample_count=len(values),
        bin_edges=edges,
        counts=counts,
        method="equal width",
    )


def noise_floor(series: RawSampleSeries) -> NoiseFloor:
    values = series.samples
    center = statistics.median(values)
    deviations = tuple(abs(value - center) for value in values)
    mad = statistics.median(deviations)
    p95_deviation = _percentile(deviations, 0.95)
    stddev = statistics.pstdev(values)
    magnitude = abs(center)
    mean_magnitude = abs(statistics.fmean(values))
    return NoiseFloor(
        schema_version="sloforge.branchfabric.noise-floor/v1",
        series_id=series.series_id,
        metric=series.metric,
        unit=series.unit,
        provenance=series.provenance,
        sample_count=len(values),
        median=center,
        median_absolute_deviation=mad,
        p95_absolute_deviation=p95_deviation,
        population_standard_deviation=stddev,
        coefficient_of_variation=(stddev / mean_magnitude if mean_magnitude else None),
        relative_mad=(mad / magnitude if magnitude else None),
        relative_p95_absolute_deviation=(p95_deviation / magnitude if magnitude else None),
        method="all measurement samples; no outlier removal",
    )


def identify_mad_outliers(series: RawSampleSeries, *, threshold: float = 3.5) -> MadOutlierReport:
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("MAD threshold must be finite and positive")
    values = series.samples
    center = statistics.median(values)
    mad = statistics.median(abs(value - center) for value in values)
    flags: list[OutlierFlag] = []
    for index, value in enumerate(values):
        if mad == 0.0:
            if value != center:
                flags.append(
                    OutlierFlag(
                        sample_index=index,
                        value=value,
                        modified_z_score=None,
                        reason="non-median value with zero MAD",
                    )
                )
            continue
        score = 0.6744897501960817 * (value - center) / mad
        if abs(score) > threshold:
            flags.append(
                OutlierFlag(
                    sample_index=index,
                    value=value,
                    modified_z_score=score,
                    reason="modified z-score threshold",
                )
            )
    return MadOutlierReport(
        schema_version="sloforge.branchfabric.mad-outliers/v1",
        series_id=series.series_id,
        metric=series.metric,
        unit=series.unit,
        provenance=series.provenance,
        sample_count=len(values),
        median=center,
        median_absolute_deviation=mad,
        threshold=threshold,
        flags=tuple(flags),
        method="MAD modified z-score",
    )


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for offset in range(start, end):
            ranks[ordered[offset][0]] = average_rank
        start = end
    return tuple(ranks)


def _pearson(x_values: tuple[float, ...], y_values: tuple[float, ...]) -> float | None:
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    x_centered = tuple(value - x_mean for value in x_values)
    y_centered = tuple(value - y_mean for value in y_values)
    denominator = math.sqrt(
        math.fsum(value * value for value in x_centered)
        * math.fsum(value * value for value in y_centered)
    )
    if denominator == 0.0:
        return None
    value = math.fsum(x * y for x, y in zip(x_centered, y_centered, strict=True)) / denominator
    return max(-1.0, min(1.0, value))


def paired_effect_size(baseline: RawSampleSeries, candidate: RawSampleSeries) -> PairedEffectSize:
    if baseline.metric != candidate.metric:
        raise ValueError("paired effect sizes require identical metrics")
    if baseline.unit != candidate.unit:
        raise ValueError("paired effect sizes require identical units")
    if len(baseline.samples) != len(candidate.samples):
        raise ValueError("paired effect sizes require equal sample counts")
    differences = tuple(
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(baseline.samples, candidate.samples, strict=True)
    )
    difference_stddev = statistics.stdev(differences) if len(differences) > 1 else 0.0
    nonzero = tuple(value for value in differences if value != 0.0)
    if nonzero:
        ranks = _average_ranks(tuple(abs(value) for value in nonzero))
        rank_biserial = math.fsum(
            rank if difference > 0.0 else -rank
            for difference, rank in zip(nonzero, ranks, strict=True)
        ) / math.fsum(ranks)
    else:
        rank_biserial = None
    relative = tuple(
        (candidate_value - baseline_value) / abs(baseline_value)
        for baseline_value, candidate_value in zip(baseline.samples, candidate.samples, strict=True)
        if baseline_value != 0.0
    )
    return PairedEffectSize(
        schema_version="sloforge.branchfabric.paired-effect-size/v1",
        baseline_series_id=baseline.series_id,
        candidate_series_id=candidate.series_id,
        baseline_provenance=baseline.provenance,
        candidate_provenance=candidate.provenance,
        metric=baseline.metric,
        unit=baseline.unit,
        pair_count=len(differences),
        mean_difference=statistics.fmean(differences),
        median_difference=statistics.median(differences),
        paired_cohens_dz=(
            statistics.fmean(differences) / difference_stddev if difference_stddev else None
        ),
        matched_pairs_rank_biserial=rank_biserial,
        median_relative_change=(statistics.median(relative) if relative else None),
        relative_pairs_used=len(relative),
        zero_baseline_pairs=len(differences) - len(relative),
        relative_denominator="absolute baseline",
        direction="candidate minus baseline",
        pair_order="preserved by sample index",
    )


def correlation(x_series: RawSampleSeries, y_series: RawSampleSeries) -> CorrelationResult:
    if len(x_series.samples) != len(y_series.samples):
        raise ValueError("correlation requires equal sample counts")
    x_values = x_series.samples
    y_values = y_series.samples
    x_ranks = _average_ranks(x_values)
    y_ranks = _average_ranks(y_values)
    return CorrelationResult(
        schema_version="sloforge.branchfabric.correlation/v1",
        x_series_id=x_series.series_id,
        y_series_id=y_series.series_id,
        x_provenance=x_series.provenance,
        y_provenance=y_series.provenance,
        x_metric=x_series.metric,
        y_metric=y_series.metric,
        x_unit=x_series.unit,
        y_unit=y_series.unit,
        pair_count=len(x_values),
        pearson_r=_pearson(x_values, y_values),
        spearman_rho=_pearson(x_ranks, y_ranks),
        ties_present=(len(set(x_values)) < len(x_values) or len(set(y_values)) < len(y_values)),
        pair_order="preserved by sample index",
        constant_series_policy="undefined correlation is null",
    )


__all__ = [
    "MAX_BOOTSTRAP_DRAWS",
    "MAX_BOOTSTRAP_REPETITIONS",
    "MAX_ECDF_POINTS",
    "MAX_HISTOGRAM_BINS",
    "MAX_SAMPLES",
    "ArtifactEvidence",
    "BootstrapConfidenceInterval",
    "BootstrapStatistic",
    "CorrelationResult",
    "EmpiricalCdf",
    "Histogram",
    "MadOutlierReport",
    "NoiseFloor",
    "PairedEffectSize",
    "RawSampleSeries",
    "SummaryStatistics",
    "WarmupPolicy",
    "bootstrap_confidence_interval",
    "correlation",
    "empirical_cdf",
    "histogram",
    "identify_mad_outliers",
    "noise_floor",
    "paired_effect_size",
    "summarize",
]
