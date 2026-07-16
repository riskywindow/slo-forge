"""Translate internal compatibility decisions to the single canonical wire report."""

from __future__ import annotations

import hashlib

from sloforge.continuum.ir import (
    CompatibilityCheck,
    CompatibilityReport,
    Digest,
    ExactnessClass,
    QualityContract,
    RecomputeRequirement,
    RequiredConversion,
    RuntimeIdentity,
    canonical_hash,
)

from .models import CompatibilityDecision, ReasonSeverity


def _digest_text(value: str) -> Digest:
    return Digest(value=hashlib.sha256(value.encode("utf-8")).hexdigest())


def to_canonical_report(
    decision: CompatibilityDecision,
    *,
    source_capsule_id: str,
    destination_runtime: RuntimeIdentity,
    destination_physical_plan: Digest,
    quality_contracts: tuple[QualityContract, ...] = (),
) -> CompatibilityReport:
    """Create the artifact-facing ABI report from an engine-internal decision."""

    checks = tuple(
        CompatibilityCheck(
            check_id=reason.code.lower(),
            subject=reason.component,
            result=(
                "fail"
                if reason.severity is ReasonSeverity.BLOCKING
                else "requires_recomputation"
                if decision.compatibility_class is ExactnessClass.RECOMPUTATION_ASSISTED
                and reason.severity is ReasonSeverity.REQUIREMENT
                else "pass"
            ),
            explanation=reason.message,
            evidence=tuple(_digest_text(item) for item in reason.evidence),
        )
        for reason in decision.reasons
    )
    if decision.quality_implications and not quality_contracts:
        raise ValueError("quality-bounded decisions require typed canonical quality contracts")
    canonical = CompatibilityReport(
        report_id="pending",
        source_capsule_id=source_capsule_id,
        destination_runtime=destination_runtime,
        destination_physical_plan=destination_physical_plan,
        compatibility_class=decision.compatibility_class,
        checks=checks,
        rejected_classes=tuple(item.exactness_class for item in decision.rejected_classes),
        required_conversions=tuple(
            RequiredConversion(
                component_id="portable_state",
                operation=operation,
                exactness_class=decision.compatibility_class,
            )
            for operation in decision.required_conversion
        ),
        required_recomputation=tuple(
            RecomputeRequirement(
                component_id=component,
                from_component_ids=("state.token_history",),
                reason=(
                    "dependency analysis requires regeneration from the declared "
                    f"{requirement.source}"
                ),
            )
            for requirement in decision.required_recomputation
            for component in requirement.state_components
        ),
        unsupported_state=decision.unsupported_state,
        quality_implications=quality_contracts,
        verification_obligations=tuple(
            f"{item.obligation_id}: {item.method}" for item in decision.verification_obligations
        ),
        migration_restrictions=decision.migration_restrictions,
    )
    identifier = canonical_hash(canonical.model_copy(update={"report_id": "pending"}))
    return canonical.model_copy(update={"report_id": f"continuum-compatibility-{identifier[:24]}"})
