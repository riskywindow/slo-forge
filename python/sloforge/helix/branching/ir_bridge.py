"""Late-bound construction of canonical environment state-reuse evidence."""

from __future__ import annotations

import importlib
from typing import Any

from sloforge.helix.capture.models import canonical_digest


def build_ir_state_reuse_report(
    *,
    source_capsule: object,
    target_environment_id: str,
    target_policy_epoch: object,
    target_compatibility_fingerprint: object,
    mode: str,
    compatible: bool,
    reused: bool,
    conversion_evidence: object | None,
    reason: str,
    assessed_at: str,
    lineage: tuple[object, ...],
) -> object:
    """Build canonical ``helix.ir.StateReuseReport`` without an import cycle."""

    ir: Any = importlib.import_module("sloforge.helix.ir")
    state_reuse_mode = ir.StateReuseMode(mode)
    provisional = ir.StateReuseReport(
        report_id="pending",
        source_capsule=source_capsule,
        target_environment_id=target_environment_id,
        target_policy_epoch=target_policy_epoch,
        target_compatibility_fingerprint=target_compatibility_fingerprint,
        mode=state_reuse_mode,
        compatible=compatible,
        reused=reused,
        conversion_evidence=conversion_evidence,
        reason=reason,
        assessed_at=assessed_at,
        lineage=lineage,
    )
    report_id = canonical_digest(provisional.model_dump(mode="json", exclude={"report_id"}))
    return provisional.model_copy(update={"report_id": report_id})
