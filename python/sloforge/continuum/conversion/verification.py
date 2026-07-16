"""Independent equivalence checks and measured backend selection."""

from __future__ import annotations

import random
import statistics
import time
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sloforge.continuum.compatibility import ExactnessClass

from .compiler import canonical_convert, direct_convert
from .layouts import KVLayout, PhysicalKVState, decode_logical


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ConversionBackend(StrEnum):
    CANONICAL_CPU = "canonical_cpu"
    DIRECT_CPU = "direct_cpu"


class ConversionVerificationEvidence(EvidenceModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    source_hash: str = Field(min_length=1)
    canonical_hash: str = Field(min_length=1)
    direct_hash: str = Field(min_length=1)
    element_count: int = Field(ge=0)
    maximum_absolute_error: float = Field(ge=0.0, allow_inf_nan=False)
    source_to_destination_maximum_absolute_error: float = Field(ge=0.0, allow_inf_nan=False)
    declared_exactness: ExactnessClass
    numeric_tolerance: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    numeric_contract_satisfied: bool
    comparison_scope: Literal["direct_vs_canonical_destination_and_source_vs_destination"] = (
        "direct_vs_canonical_destination_and_source_vs_destination"
    )
    exact: bool
    canonical_integrity_valid: bool
    direct_integrity_valid: bool


class ConversionMeasurement(EvidenceModel):
    backend: ConversionBackend
    iteration: int = Field(ge=0)
    elapsed_ns: int = Field(gt=0)
    source_bytes: int = Field(ge=0)
    destination_bytes: int = Field(ge=0)
    seed: int
    clock: Literal["perf_counter_ns"] = "perf_counter_ns"


class ConversionSelection(EvidenceModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    selected_backend: ConversionBackend
    canonical_median_ns: int = Field(gt=0)
    direct_median_ns: int = Field(gt=0)
    measurements: tuple[ConversionMeasurement, ...]
    verification: ConversionVerificationEvidence
    selection_rule: Literal["lowest_measured_median_after_equivalence"] = (
        "lowest_measured_median_after_equivalence"
    )


class QualityBoundedConversionEvidence(EvidenceModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    exactness_class: Literal[ExactnessClass.QUALITY_BOUNDED] = ExactnessClass.QUALITY_BOUNDED
    quality_metric: Literal["source_to_destination_maximum_absolute_error"] = (
        "source_to_destination_maximum_absolute_error"
    )
    source_dtype: str
    destination_dtype: str
    quality_budget: float = Field(gt=0.0, allow_inf_nan=False)
    observed_quality_loss: float = Field(ge=0.0, allow_inf_nan=False)
    element_count: int = Field(ge=0)
    contract_satisfied: bool
    reference_verification: ConversionVerificationEvidence


def quality_bounded_convert(
    source: PhysicalKVState,
    destination: KVLayout,
    *,
    maximum_temporary_bytes: int,
    maximum_absolute_error_budget: float,
) -> tuple[PhysicalKVState, QualityBoundedConversionEvidence]:
    """Execute a lossy floating dtype conversion only under an explicit measured budget."""

    if maximum_absolute_error_budget <= 0 or not np.isfinite(maximum_absolute_error_budget):
        raise ValueError("quality budget must be a positive finite maximum absolute error")
    source_dtype = np.dtype(source.layout.dtype)
    destination_dtype = np.dtype(destination.dtype)
    if (
        source_dtype.kind != "f"
        or destination_dtype.kind != "f"
        or destination_dtype.itemsize >= source_dtype.itemsize
    ):
        raise ValueError("quality-bounded conversion requires a lower-precision floating target")
    verification = verify_direct_against_canonical(
        source,
        destination,
        maximum_temporary_bytes=maximum_temporary_bytes,
        numeric_tolerance=maximum_absolute_error_budget,
    )
    if not verification.exact:
        raise ValueError("lossy direct conversion differs from the trusted canonical converter")
    if not verification.numeric_contract_satisfied:
        raise ValueError("observed conversion loss exceeds the declared quality budget")
    converted = direct_convert(
        source,
        destination,
        maximum_temporary_bytes=maximum_temporary_bytes,
    )
    return converted, QualityBoundedConversionEvidence(
        source_dtype=source_dtype.name,
        destination_dtype=destination_dtype.name,
        quality_budget=maximum_absolute_error_budget,
        observed_quality_loss=verification.source_to_destination_maximum_absolute_error,
        element_count=verification.element_count,
        contract_satisfied=True,
        reference_verification=verification,
    )


def verify_direct_against_canonical(
    source: PhysicalKVState,
    destination: KVLayout,
    *,
    maximum_temporary_bytes: int,
    numeric_tolerance: float | None = None,
) -> ConversionVerificationEvidence:
    """Compare an optimized conversion with an independently implemented fallback."""

    source.verify_integrity()
    canonical = canonical_convert(source, destination)
    direct = direct_convert(
        source,
        destination,
        maximum_temporary_bytes=maximum_temporary_bytes,
    )
    canonical.verify_integrity()
    direct.verify_integrity()
    canonical_key, canonical_value = decode_logical(canonical)
    direct_key, direct_value = decode_logical(direct)
    source_key, source_value = decode_logical(source)
    compared = canonical_key.size + canonical_value.size
    if compared:
        key_error = float(
            np.max(np.abs(canonical_key.astype(np.float64) - direct_key.astype(np.float64)))
        )
        value_error = float(
            np.max(np.abs(canonical_value.astype(np.float64) - direct_value.astype(np.float64)))
        )
        maximum_error = max(key_error, value_error)
    else:
        maximum_error = 0.0
    if compared:
        source_key_error = float(
            np.max(np.abs(source_key.astype(np.float64) - canonical_key.astype(np.float64)))
        )
        source_value_error = float(
            np.max(np.abs(source_value.astype(np.float64) - canonical_value.astype(np.float64)))
        )
        source_to_destination_error = max(source_key_error, source_value_error)
    else:
        source_to_destination_error = 0.0
    declared_exactness = (
        ExactnessClass.EXACT_SEMANTIC
        if np.can_cast(np.dtype(source.layout.dtype), np.dtype(destination.dtype), casting="safe")
        else ExactnessClass.NUMERICALLY_EQUIVALENT
    )
    numeric_contract_satisfied = declared_exactness is ExactnessClass.EXACT_SEMANTIC or (
        numeric_tolerance is not None and source_to_destination_error <= numeric_tolerance
    )
    exact = (
        canonical.content_hash == direct.content_hash
        and np.array_equal(canonical_key, direct_key)
        and np.array_equal(canonical_value, direct_value)
    )
    return ConversionVerificationEvidence(
        source_hash=source.content_hash,
        canonical_hash=canonical.content_hash,
        direct_hash=direct.content_hash,
        element_count=compared,
        maximum_absolute_error=maximum_error,
        source_to_destination_maximum_absolute_error=source_to_destination_error,
        declared_exactness=declared_exactness,
        numeric_tolerance=numeric_tolerance,
        numeric_contract_satisfied=numeric_contract_satisfied,
        exact=exact,
        canonical_integrity_valid=True,
        direct_integrity_valid=True,
    )


def measure_and_select_converter(
    source: PhysicalKVState,
    destination: KVLayout,
    *,
    maximum_temporary_bytes: int,
    repetitions: int = 5,
    seed: int,
) -> ConversionSelection:
    """Select a backend from raw timings only after independent equivalence passes."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    verification = verify_direct_against_canonical(
        source,
        destination,
        maximum_temporary_bytes=maximum_temporary_bytes,
    )
    if not verification.exact:
        # Direct conversion remains untrusted; record no misleading benchmark choice.
        raise ValueError("direct converter failed canonical equivalence verification")

    rng = random.Random(seed)
    measurements: list[ConversionMeasurement] = []
    backends = [ConversionBackend.CANONICAL_CPU, ConversionBackend.DIRECT_CPU]
    for iteration in range(repetitions):
        order = list(backends)
        rng.shuffle(order)
        for backend in order:
            started = time.perf_counter_ns()
            if backend is ConversionBackend.CANONICAL_CPU:
                converted = canonical_convert(source, destination)
            else:
                converted = direct_convert(
                    source,
                    destination,
                    maximum_temporary_bytes=maximum_temporary_bytes,
                )
            elapsed = max(1, time.perf_counter_ns() - started)
            converted.verify_integrity()
            measurements.append(
                ConversionMeasurement(
                    backend=backend,
                    iteration=iteration,
                    elapsed_ns=elapsed,
                    source_bytes=source.layout.physical_nbytes,
                    destination_bytes=destination.physical_nbytes,
                    seed=seed,
                )
            )

    canonical_median = int(
        statistics.median(
            measurement.elapsed_ns
            for measurement in measurements
            if measurement.backend is ConversionBackend.CANONICAL_CPU
        )
    )
    direct_median = int(
        statistics.median(
            measurement.elapsed_ns
            for measurement in measurements
            if measurement.backend is ConversionBackend.DIRECT_CPU
        )
    )
    selected = (
        ConversionBackend.DIRECT_CPU
        if direct_median < canonical_median
        else ConversionBackend.CANONICAL_CPU
    )
    return ConversionSelection(
        selected_backend=selected,
        canonical_median_ns=canonical_median,
        direct_median_ns=direct_median,
        measurements=tuple(measurements),
        verification=verification,
    )
