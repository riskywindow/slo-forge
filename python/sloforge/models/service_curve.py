from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sloforge.profiler.core import ProfileBundle, RawMeasurement
from sloforge.util import percentile


class CurvePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    median_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    sample_count: int = Field(ge=1)


class MonotonicServiceCurve(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Literal["prompt_tokens", "active_sequences"]
    points: list[CurvePoint]
    residual_p95_ms: float = Field(ge=0)
    conformal_radius_ms: float = Field(ge=0)
    calibration_sample_count: int = Field(ge=0)
    nominal_coverage: float = Field(default=0.95, gt=0, lt=1)

    def predict(self, value: float) -> tuple[float, float, float]:
        if value < 0:
            raise ValueError("service-curve inputs must be nonnegative")
        if not self.points:
            raise RuntimeError("cannot predict with an empty service curve")
        if len(self.points) == 1 or value <= self.points[0].x:
            center = self.points[0].median_ms
        elif value >= self.points[-1].x:
            left, right = self.points[-2], self.points[-1]
            slope = (right.median_ms - left.median_ms) / max(right.x - left.x, 1e-9)
            center = right.median_ms + max(slope, 0.0) * (value - right.x)
        else:
            center = self.points[-1].median_ms
            for left, right in zip(self.points, self.points[1:], strict=False):
                if left.x <= value <= right.x:
                    fraction = (value - left.x) / max(right.x - left.x, 1e-9)
                    center = left.median_ms + fraction * (right.median_ms - left.median_ms)
                    break
        radius = self.conformal_radius_ms
        return center, max(0.0, center - radius), center + radius


class CandidateModels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    prefill: MonotonicServiceCurve
    decode: MonotonicServiceCurve
    startup_median_ms: float
    startup_p95_ms: float
    held_out_mape: float
    interval_coverage: float
    decode_held_out_mape: float
    decode_interval_coverage: float
    held_out_sample_count: int


class CalibratedModels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.models/v1"] = "sloforge.models/v1"
    seed: int
    candidates: list[CandidateModels]
    limitations: list[str]


def _finite_sample_radius(residuals: list[float], nominal_coverage: float) -> float:
    if not residuals:
        return 0.0
    ordered = sorted(residuals)
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * nominal_coverage))
    return ordered[rank - 1]


def _curve(
    measurements: list[RawMeasurement],
    dimension: Literal["prompt_tokens", "active_sequences"],
    rng: random.Random,
) -> MonotonicServiceCurve:
    grouped: defaultdict[float, list[float]] = defaultdict(list)
    for item in measurements:
        raw_x = item.prompt_tokens if dimension == "prompt_tokens" else item.active_sequences
        if raw_x is not None and not item.warmup and not item.failed:
            grouped[float(raw_x)].append(item.latency_ms)
    training: dict[float, list[float]] = {}
    calibration: dict[float, list[float]] = {}
    for coordinate, samples in grouped.items():
        shuffled = samples.copy()
        rng.shuffle(shuffled)
        calibration_count = max(1, len(shuffled) // 4) if len(shuffled) > 1 else 0
        training[coordinate] = shuffled[calibration_count:] or shuffled
        calibration[coordinate] = shuffled[:calibration_count]
    points: list[CurvePoint] = []
    running = 0.0
    for coordinate, samples in sorted(training.items()):
        median = max(running, percentile(samples, 0.50))
        running = median
        points.append(
            CurvePoint(
                x=coordinate,
                median_ms=median,
                p95_ms=max(median, percentile(samples, 0.95)),
                sample_count=len(samples),
            )
        )
    if not points:
        raise ValueError(f"no usable {dimension} measurements")
    residuals: list[float] = []
    lookup = {point.x: point.median_ms for point in points}
    for coordinate, samples in calibration.items():
        residuals.extend(abs(sample - lookup[coordinate]) for sample in samples)
    nominal_coverage = 0.95
    radius = _finite_sample_radius(residuals, nominal_coverage)
    return MonotonicServiceCurve(
        dimension=dimension,
        points=points,
        residual_p95_ms=radius,
        conformal_radius_ms=radius,
        calibration_sample_count=len(residuals),
        nominal_coverage=nominal_coverage,
    )


def fit_service_curves(profile: ProfileBundle, *, seed: int) -> CalibratedModels:
    rng = random.Random(seed)
    models: list[CandidateModels] = []
    by_candidate: defaultdict[str, list[RawMeasurement]] = defaultdict(list)
    for measurement in profile.raw_measurements:
        by_candidate[measurement.candidate_id].append(measurement)
    for candidate in profile.candidates:
        if not candidate.feasible:
            continue
        raw = by_candidate[candidate.candidate.candidate_id]
        prefill = _curve([item for item in raw if item.stage == "prefill"], "prompt_tokens", rng)
        decode = _curve([item for item in raw if item.stage == "decode"], "active_sequences", rng)
        startup = [item.latency_ms for item in raw if item.stage == "startup" and not item.warmup]
        representative_load = [item for item in raw if item.stage == "load" and not item.failed]
        rng.shuffle(representative_load)
        load_calibration_count = (
            max(1, len(representative_load) // 2) if len(representative_load) > 1 else 0
        )
        load_calibration = representative_load[:load_calibration_count]
        held_out = representative_load[load_calibration_count:]
        # Stage-D prefill samples exclude HTTP/runtime queue overhead that is present in Stage-E
        # TTFT. Calibrate the interval for the quantity the optimizer constrains, then evaluate it
        # on a disjoint test half. Reusing all Stage-E samples as "held out" gave tiny service-only
        # radii and severe undercoverage for workload TTFT.
        load_prefill_residuals: list[float] = []
        load_decode_residuals: list[float] = []
        for item in load_calibration:
            if item.prompt_tokens is not None and item.ttft_ms is not None:
                center, _, _ = prefill.predict(float(item.prompt_tokens))
                load_prefill_residuals.append(abs(center - item.ttft_ms))
            if item.active_sequences is not None and item.itl_ms is not None:
                center, _, _ = decode.predict(float(item.active_sequences))
                load_decode_residuals.append(abs(center - item.itl_ms))
        workload_radius = _finite_sample_radius(load_prefill_residuals, prefill.nominal_coverage)
        combined_prefill_radius = max(prefill.conformal_radius_ms, workload_radius)
        prefill = prefill.model_copy(
            update={
                "residual_p95_ms": combined_prefill_radius,
                "conformal_radius_ms": combined_prefill_radius,
                "calibration_sample_count": (
                    prefill.calibration_sample_count + len(load_prefill_residuals)
                ),
            }
        )
        workload_decode_radius = _finite_sample_radius(
            load_decode_residuals, decode.nominal_coverage
        )
        combined_decode_radius = max(decode.conformal_radius_ms, workload_decode_radius)
        decode = decode.model_copy(
            update={
                "residual_p95_ms": combined_decode_radius,
                "conformal_radius_ms": combined_decode_radius,
                "calibration_sample_count": (
                    decode.calibration_sample_count + len(load_decode_residuals)
                ),
            }
        )
        errors: list[float] = []
        covered = 0
        decode_errors: list[float] = []
        decode_covered = 0
        for item in held_out:
            if item.prompt_tokens is None or item.active_sequences is None or item.ttft_ms is None:
                continue
            prefill_center, prefill_low, prefill_high = prefill.predict(float(item.prompt_tokens))
            errors.append(abs(prefill_center - item.ttft_ms) / max(item.ttft_ms, 1e-6))
            if prefill_low <= item.ttft_ms <= prefill_high:
                covered += 1
            if item.itl_ms is not None:
                decode_center, decode_low, decode_high = decode.predict(
                    float(item.active_sequences)
                )
                decode_errors.append(abs(decode_center - item.itl_ms) / max(item.itl_ms, 1e-6))
                if decode_low <= item.itl_ms <= decode_high:
                    decode_covered += 1
        models.append(
            CandidateModels(
                candidate_id=candidate.candidate.candidate_id,
                prefill=prefill,
                decode=decode,
                startup_median_ms=percentile(startup, 0.50),
                startup_p95_ms=percentile(startup, 0.95),
                held_out_mape=float(np.mean(errors)) if errors else math.nan,
                interval_coverage=covered / len(errors) if errors else 0.0,
                decode_held_out_mape=(float(np.mean(decode_errors)) if decode_errors else math.nan),
                decode_interval_coverage=(
                    decode_covered / len(decode_errors) if decode_errors else 0.0
                ),
                held_out_sample_count=len(errors),
            )
        )
    if not models:
        raise ValueError("profile contains no feasible candidates to calibrate")
    return CalibratedModels(
        seed=seed,
        candidates=models,
        limitations=[
            "Intervals are calibrated only on measured mock or hardware configurations.",
            "Conformal radii use deterministic coordinate and representative-load calibration splits with disjoint load test samples.",
            "No claim is made about unmeasured hardware generalization.",
        ],
    )
