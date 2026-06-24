"""Bottleneck-evidence validation and focused target selection."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import BottleneckEvidence, RawBottleneckRecord


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
    if abs(raw.token_loop_fraction - evidence.observed_fraction) > 1e-12:
        raise BottleneckEvidenceError("raw evidence fraction does not match its typed reference")
    if evidence.synthetic and evidence.source.value.endswith("measured"):
        raise BottleneckEvidenceError("measured evidence cannot be marked synthetic")
