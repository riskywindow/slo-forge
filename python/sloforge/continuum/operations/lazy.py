"""Legality checks for carefully constrained lazy state availability."""

from __future__ import annotations

from sloforge.continuum.ir import AccessPatternKind, ExecutionStateCapsule

from .models import (
    LazyMigrationAssessment,
    LazyMigrationRejected,
    LazyReason,
    LazySegmentDeclaration,
)


def assess_lazy_migration(
    capsule: ExecutionStateCapsule,
    declarations: tuple[LazySegmentDeclaration, ...],
) -> LazyMigrationAssessment:
    """Prove declared delayed segments legal; undeclared segments remain pre-resume state."""

    if not declarations:
        return LazyMigrationAssessment(
            legal=False,
            legal_segment_ids=(),
            rejected_segment_ids=(),
            obligations=(),
            reasons=("lazy migration requires at least one explicit segment declaration",),
        )
    if len(declarations) > 4096:
        raise ValueError("lazy declaration bound exceeds 4096")
    declaration_by_id = {item.segment_id: item for item in declarations}
    if len(declaration_by_id) != len(declarations):
        raise ValueError("lazy segment declarations must be unique")
    physical = capsule.physical_state
    segments = {segment.segment_id: segment for segment in physical.segments}
    patterns = {item.access_pattern_id: item for item in physical.access_patterns}
    dense_attention = capsule.logical_state.attention is not None and any(
        "dense" in layer.attention_window_semantics.lower()
        or "full-context" in layer.attention_window_semantics.lower()
        for layer in capsule.logical_state.attention.layers
    )
    legal: list[str] = []
    rejected: list[str] = []
    reasons: list[str] = []
    obligations: list[str] = []
    for segment_id, declaration in sorted(declaration_by_id.items()):
        segment = segments.get(segment_id)
        if segment is None:
            rejected.append(segment_id)
            reasons.append(f"{segment_id}: segment is absent from the physical layout")
            continue
        pattern = patterns[segment.access_pattern_id]
        if pattern.required_before_resume:
            rejected.append(segment_id)
            reasons.append(f"{segment_id}: state is required before resume")
            continue
        if segment.logical_state_reference == "state/attention-kv" and dense_attention:
            rejected.append(segment_id)
            reasons.append(f"{segment_id}: dense full-attention state cannot be lazily unavailable")
            continue
        allowed = False
        if declaration.reason is LazyReason.SLIDING_WINDOW:
            allowed = (
                pattern.kind is AccessPatternKind.SLIDING_WINDOW
                and pattern.streamable_before_use
                and declaration.explicit_stall_on_miss
            )
        elif declaration.reason is LazyReason.LAYER_SEQUENTIAL:
            allowed = (
                pattern.kind is AccessPatternKind.LAYER_SEQUENTIAL
                and pattern.streamable_before_use
                and declaration.available_before_first_use
            )
        elif declaration.reason is LazyReason.COLD:
            allowed = (
                pattern.kind in {AccessPatternKind.READ_ONLY, AccessPatternKind.SPARSE_ACCESS}
                and declaration.explicit_stall_on_miss
            )
        elif declaration.reason is LazyReason.RECOMPUTABLE:
            allowed = pattern.recomputable and declaration.explicit_stall_on_miss
        if not allowed:
            rejected.append(segment_id)
            reasons.append(
                f"{segment_id}: declared {declaration.reason.value} contract is not proven by "
                f"access pattern {pattern.kind.value}"
            )
            continue
        legal.append(segment_id)
        obligations.append(
            f"{segment_id}: enforce {declaration.failure_behavior}; checksum before first use"
        )
    return LazyMigrationAssessment(
        legal=not rejected and bool(legal),
        legal_segment_ids=tuple(legal),
        rejected_segment_ids=tuple(rejected),
        obligations=tuple(obligations),
        reasons=tuple(reasons),
    )


def require_lazy_migration(
    capsule: ExecutionStateCapsule,
    declarations: tuple[LazySegmentDeclaration, ...],
) -> LazyMigrationAssessment:
    """Return a proof obligation bundle or raise one typed rejection."""

    assessment = assess_lazy_migration(capsule, declarations)
    if not assessment.legal:
        detail = "; ".join(assessment.reasons) or "no segment satisfied lazy legality"
        raise LazyMigrationRejected(detail)
    return assessment
