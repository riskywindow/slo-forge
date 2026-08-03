"""Bottleneck-evidence validation and focused target selection."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import AttributionScope, BottleneckEvidence, EvidenceSource, RawBottleneckRecord


class BottleneckEvidenceError(ValueError):
    """Raised when kernel targeting lacks immutable causal evidence."""


def validate_bottleneck_evidence(
    evidence: BottleneckEvidence, *, minimum_fraction: float = 0.05
) -> None:
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum bottleneck fraction must be in (0, 1]")
    if evidence.operator_id != "quantized-state-update":
        raise BottleneckEvidenceError("evidence does not identify the focused operator")
    if evidence.observed_fraction < minimum_fraction:
        raise BottleneckEvidenceError("operator contribution is below the synthesis threshold")
    if evidence.source is EvidenceSource.CPU_PROFILE_MEASURED:
        if evidence.causal_attribution:
            raise BottleneckEvidenceError("CPU profiling cannot claim Autopsy causality")
        if evidence.attribution_scope is AttributionScope.SYNTHETIC_OPERATOR_MICROPROBE:
            if not evidence.synthetic:
                raise BottleneckEvidenceError("the CPU microprobe must remain synthetic")
        elif evidence.attribution_scope is AttributionScope.REFERENCE_WORKLOAD_TRACE_PROFILE:
            if evidence.synthetic:
                raise BottleneckEvidenceError("the reference workload trace cannot be synthetic")
        else:
            raise BottleneckEvidenceError("CPU profile attribution scope is unsupported")
    path = Path(evidence.raw_evidence_path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise BottleneckEvidenceError("raw bottleneck evidence is unavailable") from error
    if hashlib.sha256(payload).hexdigest() != evidence.raw_evidence_sha256:
        raise BottleneckEvidenceError("raw bottleneck evidence digest mismatch")
    try:
        raw = RawBottleneckRecord.model_validate_json(payload, strict=True)
    except ValueError as error:
        raise BottleneckEvidenceError("raw bottleneck evidence schema is invalid") from error
    if raw.operator_id != evidence.operator_id:
        raise BottleneckEvidenceError("raw evidence operator does not match its typed reference")
    if raw.seed != evidence.deterministic_seed:
        raise BottleneckEvidenceError("raw evidence seed does not match its typed reference")
    if len(raw.inclusive_cpu_time_ns) != evidence.sample_count:
        raise BottleneckEvidenceError(
            "raw evidence sample count does not match its typed reference"
        )
    if raw.attribution_scope != evidence.attribution_scope.value:
        raise BottleneckEvidenceError("raw evidence attribution scope does not match")
    if raw.workload_fingerprint != (
        evidence.workload_fingerprint
        if evidence.attribution_scope is AttributionScope.REFERENCE_WORKLOAD_TRACE_PROFILE
        else None
    ):
        raise BottleneckEvidenceError("raw evidence workload fingerprint does not match")
    if raw.attribution_scope == AttributionScope.REFERENCE_WORKLOAD_TRACE_PROFILE.value:
        trace = Path(str(raw.workload_trace_path))
        raw_parent = path.resolve(strict=True).parent
        if trace.is_symlink():
            raise BottleneckEvidenceError("raw workload trace must not be a symlink")
        try:
            trace = trace.resolve(strict=True)
            trace.relative_to(raw_parent)
        except (OSError, ValueError) as error:
            raise BottleneckEvidenceError(
                "raw workload trace must be retained beside its profile"
            ) from error
        if hashlib.sha256(trace.read_bytes()).hexdigest() != raw.workload_trace_sha256:
            raise BottleneckEvidenceError("raw workload trace digest mismatch")
        if hashlib.sha256(trace.read_bytes()).hexdigest() != raw.workload_fingerprint:
            raise BottleneckEvidenceError("raw workload fingerprint is not trace-derived")
    if abs(raw.operator_probe_fraction - evidence.observed_fraction) > 1e-12:
        raise BottleneckEvidenceError("raw evidence fraction does not match its typed reference")
