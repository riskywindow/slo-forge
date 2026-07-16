"""Dependency-aware semantic compatibility analysis."""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    CompatibilityDecision,
    CompatibilityReason,
    CompatibilityRequest,
    ExactnessClass,
    ReasonSeverity,
    RecomputationRequirement,
    RejectedCompatibilityClass,
    VerificationKind,
    VerificationObligation,
)


@dataclass(frozen=True)
class _Decision:
    exactness: ExactnessClass
    reasons: tuple[CompatibilityReason, ...]
    conversions: tuple[str, ...] = ()
    recomputation: tuple[RecomputationRequirement, ...] = ()
    quality: tuple[str, ...] = ()
    obligations: tuple[VerificationObligation, ...] = ()
    restrictions: tuple[str, ...] = ()


def _reason(
    code: str,
    severity: ReasonSeverity,
    component: str,
    message: str,
    *evidence: str,
) -> CompatibilityReason:
    return CompatibilityReason(
        code=code,
        severity=severity,
        component=component,
        message=message,
        evidence=tuple(evidence),
    )


def _obligation(
    obligation_id: str,
    kind: VerificationKind,
    component: str,
    method: str,
    tolerance: float | None = None,
) -> VerificationObligation:
    return VerificationObligation(
        obligation_id=obligation_id,
        kind=kind,
        component=component,
        method=method,
        tolerance=tolerance,
    )


def _incompatible(
    reasons: tuple[CompatibilityReason, ...],
    *,
    unsupported: tuple[str, ...] = (),
) -> CompatibilityDecision:
    return CompatibilityDecision(
        compatibility_class=ExactnessClass.INCOMPATIBLE,
        safe=False,
        reasons=reasons,
        rejected_classes=tuple(
            RejectedCompatibilityClass(
                exactness_class=exactness,
                reason_codes=tuple(reason.code for reason in reasons),
            )
            for exactness in ExactnessClass
            if exactness is not ExactnessClass.INCOMPATIBLE
        ),
        required_conversion=(),
        required_recomputation=(),
        unsupported_state=unsupported,
        quality_implications=(),
        verification_obligations=(),
        migration_restrictions=("destination activation prohibited",),
    )


def _required_exactness_allows(required: ExactnessClass, achieved: ExactnessClass) -> bool:
    if required is ExactnessClass.INCOMPATIBLE:
        return False
    if achieved is ExactnessClass.EXACT_BITWISE:
        return True
    if achieved is ExactnessClass.EXACT_SEMANTIC:
        return required in {
            ExactnessClass.EXACT_SEMANTIC,
            ExactnessClass.NUMERICALLY_EQUIVALENT,
            ExactnessClass.QUALITY_BOUNDED,
        }
    if achieved is ExactnessClass.NUMERICALLY_EQUIVALENT:
        return required in {ExactnessClass.NUMERICALLY_EQUIVALENT, ExactnessClass.QUALITY_BOUNDED}
    if achieved is ExactnessClass.QUALITY_BOUNDED:
        return required is ExactnessClass.QUALITY_BOUNDED
    return required is ExactnessClass.RECOMPUTATION_ASSISTED


def _semantic_mismatches(request: CompatibilityRequest) -> tuple[CompatibilityReason, ...]:
    source = request.source
    destination = request.destination
    checks = (
        ("architecture", source.architecture, destination.architecture),
        ("tokenizer", source.tokenizer_hash, destination.tokenizer_hash),
        ("special_tokens", source.special_tokens_hash, destination.special_tokens_hash),
        ("positional_encoding", source.positional_encoding, destination.positional_encoding),
        ("rope", source.rope_fingerprint, destination.rope_fingerprint),
        ("attention_mask", source.attention_mask_semantics, destination.attention_mask_semantics),
        ("sliding_window", str(source.sliding_window), str(destination.sliding_window)),
        ("layer_count", str(source.layer_count), str(destination.layer_count)),
        ("head_count", str(source.head_count), str(destination.head_count)),
        ("kv_head_count", str(source.kv_head_count), str(destination.kv_head_count)),
        ("head_dim", str(source.head_dim), str(destination.head_dim)),
        (
            "recurrent_update",
            str(source.recurrent_update_fingerprint),
            str(destination.recurrent_update_fingerprint),
        ),
        ("sampler", source.sampler_algorithm, destination.sampler_algorithm),
    )
    return tuple(
        _reason(
            f"{name.upper()}_MISMATCH",
            ReasonSeverity.BLOCKING,
            name,
            f"{name.replace('_', ' ')} semantics differ",
            left,
            right,
        )
        for name, left, right in checks
        if left != right
    )


def _weight_decision(request: CompatibilityRequest) -> _Decision | CompatibilityDecision | None:
    source = request.source
    destination = request.destination
    adapter_changed = source.adapter_hash != destination.adapter_hash
    if source.weights_hash == destination.weights_hash:
        inconsistent = []
        if source.state_producing_weights_hash != destination.state_producing_weights_hash:
            inconsistent.append("state_producing_weights_hash")
        if source.output_head_hash != destination.output_head_hash:
            inconsistent.append("output_head_hash")
        if inconsistent:
            return _incompatible(
                (
                    _reason(
                        "MODEL_FINGERPRINT_INCONSISTENT",
                        ReasonSeverity.BLOCKING,
                        "model_weights",
                        "equal full weight hashes cannot have different derived weight fingerprints",
                        *inconsistent,
                    ),
                )
            )
        if not adapter_changed:
            return None

    evidence = request.dependency_evidence
    state_weights_unchanged = (
        source.state_producing_weights_hash == destination.state_producing_weights_hash
    )
    output_head_changed = source.output_head_hash != destination.output_head_hash
    if adapter_changed and (evidence is None or "adapter" not in evidence.changed_components):
        return _incompatible(
            (
                _reason(
                    "ADAPTER_DEPENDENCY_EVIDENCE_INCOMPLETE",
                    ReasonSeverity.BLOCKING,
                    "adapter",
                    "an adapter identity change requires dependency evidence naming the adapter",
                    str(source.adapter_hash),
                    str(destination.adapter_hash),
                ),
            )
        )
    if output_head_changed and (
        evidence is None
        or not evidence.output_head_is_state_sink
        or (state_weights_unchanged and "output_head" not in evidence.changed_components)
    ):
        return _incompatible(
            (
                _reason(
                    "OUTPUT_HEAD_DEPENDENCY_EVIDENCE_INCOMPLETE",
                    ReasonSeverity.BLOCKING,
                    "output_head",
                    "an output-head change requires dependency evidence naming the output head",
                ),
            )
        )
    if (
        state_weights_unchanged
        and output_head_changed
        and evidence is not None
        and evidence.output_head_only
        and evidence.output_head_is_state_sink
        and not evidence.affected_state_components
    ):
        return _Decision(
            exactness=ExactnessClass.EXACT_SEMANTIC,
            reasons=(
                _reason(
                    "OUTPUT_HEAD_ONLY_SAFE",
                    ReasonSeverity.INFO,
                    "model_weights",
                    "dependency evidence proves the changed output head does not produce stored state",
                    evidence.dependency_graph_hash,
                ),
            ),
            obligations=(
                _obligation(
                    "verify-output-head-dependency",
                    VerificationKind.CONTINUATION,
                    "model_weights",
                    "validate dependency graph and compare bounded continuation at the output-head boundary",
                ),
            ),
        )

    if (
        state_weights_unchanged
        and evidence is not None
        and evidence.changed_components
        and not evidence.affected_state_components
        and not (set(evidence.changed_components) & set(evidence.state_producing_components))
    ):
        return _Decision(
            exactness=ExactnessClass.EXACT_SEMANTIC,
            reasons=(
                _reason(
                    "DEPENDENCY_PROVEN_STATE_UNAFFECTED",
                    ReasonSeverity.INFO,
                    "model_revision",
                    "dependency evidence proves changed model or adapter components do not affect stored state",
                    evidence.dependency_graph_hash,
                ),
            ),
            obligations=(
                _obligation(
                    "verify-unaffected-state-dependencies",
                    VerificationKind.CONTINUATION,
                    "model_revision",
                    "validate dependency graph and compare bounded continuation",
                ),
            ),
        )

    affected = (
        evidence.affected_state_components
        if evidence is not None and evidence.affected_state_components
        else request.required_state_types
    )
    can_recompute = (
        request.allow_recomputation
        and request.destination_runtime.can_recompute_from_token_history
        and evidence is not None
        and evidence.token_history_available
        and bool(affected)
        and set(affected).issubset(evidence.recomputable_state_components)
    )
    if can_recompute and evidence is not None:
        return _Decision(
            exactness=ExactnessClass.RECOMPUTATION_ASSISTED,
            reasons=(
                _reason(
                    "STATE_WEIGHTS_CHANGED_RECOMPUTE",
                    ReasonSeverity.REQUIREMENT,
                    "model_weights",
                    "model- or adapter-derived state cannot be reused and must be regenerated from token history",
                    source.state_producing_weights_hash,
                    destination.state_producing_weights_hash,
                    evidence.dependency_graph_hash,
                ),
            ),
            recomputation=(
                RecomputationRequirement(
                    state_components=affected,
                    source="token_history",
                    dependency_graph_hash=evidence.dependency_graph_hash,
                ),
            ),
            obligations=(
                _obligation(
                    "verify-recomputed-state",
                    VerificationKind.RECOMPUTATION,
                    "weight_derived_state",
                    "recompute from the declared token history and validate bounded continuation",
                ),
            ),
            restrictions=("direct reuse of model- or adapter-derived source state prohibited",),
        )

    return _incompatible(
        (
            _reason(
                "STATE_PRODUCING_DEPENDENCY_CHANGED",
                ReasonSeverity.BLOCKING,
                "model_weights",
                "matching shapes do not make state derived from different weights or adapters reusable",
                source.state_producing_weights_hash,
                destination.state_producing_weights_hash,
                str(source.adapter_hash),
                str(destination.adapter_hash),
            ),
        )
    )


def _representation_decision(request: CompatibilityRequest) -> _Decision:
    source = request.source
    destination = request.destination
    dtype_changed = source.state_dtype != destination.state_dtype
    quantization_changed = source.quantization != destination.quantization
    layout_changed = request.source_layout_fingerprint != request.destination_layout_fingerprint
    runtime_changed = (
        request.source_runtime.runtime_name != request.destination_runtime.runtime_name
        or request.source_runtime.runtime_version != request.destination_runtime.runtime_version
        or request.source_runtime.adapter_version != request.destination_runtime.adapter_version
    )

    if quantization_changed:
        quality = request.quality_evidence
        if quality is None or not quality.within_budget:
            return _Decision(
                exactness=ExactnessClass.INCOMPATIBLE,
                reasons=(
                    _reason(
                        "QUANTIZATION_BUDGET_UNPROVEN",
                        ReasonSeverity.BLOCKING,
                        "quantization",
                        "lossy state quantization requires measured evidence within an explicit budget",
                    ),
                ),
            )
        return _Decision(
            exactness=ExactnessClass.QUALITY_BOUNDED,
            reasons=(
                _reason(
                    "QUANTIZATION_QUALITY_BOUNDED",
                    ReasonSeverity.REQUIREMENT,
                    "quantization",
                    "state representation changes through a measured lossy conversion",
                    quality.artifact_hash,
                ),
            ),
            conversions=("dequantize_or_quantize", "layout_convert" if layout_changed else "copy"),
            quality=(
                f"{quality.metric} observed loss {quality.observed_loss} <= {quality.maximum_loss} "
                f"over {quality.sample_count} samples",
            ),
            obligations=(
                _obligation(
                    "verify-quality-budget",
                    VerificationKind.QUALITY,
                    "quantized_state",
                    f"re-evaluate {quality.metric} on destination continuation",
                    quality.maximum_loss,
                ),
            ),
        )

    if dtype_changed:
        source_dtype = source.state_dtype.lower()
        destination_dtype = destination.state_dtype.lower()
        float_dtypes = {"fp8", "float8", "float16", "bfloat16", "float32", "float64"}
        integer_bits = {
            "bool": (False, 1),
            "uint8": (False, 8),
            "uint16": (False, 16),
            "uint32": (False, 32),
            "uint64": (False, 64),
            "int8": (True, 8),
            "int16": (True, 16),
            "int32": (True, 32),
            "int64": (True, 64),
        }
        source_integer = integer_bits.get(source_dtype)
        destination_integer = integer_bits.get(destination_dtype)
        integer_widening = False
        if source_integer is not None and destination_integer is not None:
            source_signed, source_bits = source_integer
            destination_signed, destination_bits = destination_integer
            integer_widening = (
                source_signed == destination_signed and destination_bits >= source_bits
            ) or (not source_signed and destination_signed and destination_bits > source_bits)
        if integer_widening:
            return _Decision(
                exactness=ExactnessClass.EXACT_SEMANTIC,
                reasons=(
                    _reason(
                        "DTYPE_CONVERSION_LOSSLESS",
                        ReasonSeverity.REQUIREMENT,
                        "state_dtype",
                        "integer widening preserves every logical state value",
                        source.state_dtype,
                        destination.state_dtype,
                    ),
                ),
                conversions=("dtype_convert", "layout_convert" if layout_changed else "copy"),
                obligations=(
                    _obligation(
                        "verify-lossless-dtype-conversion",
                        VerificationKind.CONVERSION_EQUIVALENCE,
                        "converted_state",
                        "compare destination values with the trusted canonical converter",
                    ),
                ),
            )
        if source_dtype not in float_dtypes or destination_dtype not in float_dtypes:
            quality = request.quality_evidence
            if quality is None or not quality.within_budget:
                return _Decision(
                    exactness=ExactnessClass.INCOMPATIBLE,
                    reasons=(
                        _reason(
                            "DTYPE_CONVERSION_UNPROVEN",
                            ReasonSeverity.BLOCKING,
                            "state_dtype",
                            "potentially lossy integer/float dtype conversion requires measured quality evidence",
                            source.state_dtype,
                            destination.state_dtype,
                        ),
                    ),
                )
            return _Decision(
                exactness=ExactnessClass.QUALITY_BOUNDED,
                reasons=(
                    _reason(
                        "DTYPE_CONVERSION_QUALITY_BOUNDED",
                        ReasonSeverity.REQUIREMENT,
                        "state_dtype",
                        "a measured quality contract bounds the lossy dtype conversion",
                        quality.artifact_hash,
                    ),
                ),
                conversions=("dtype_convert", "layout_convert" if layout_changed else "copy"),
                quality=(
                    f"{quality.metric} observed loss {quality.observed_loss} <= "
                    f"{quality.maximum_loss} over {quality.sample_count} samples",
                ),
                obligations=(
                    _obligation(
                        "verify-dtype-quality-budget",
                        VerificationKind.QUALITY,
                        "converted_state",
                        f"re-evaluate {quality.metric} on destination continuation",
                        quality.maximum_loss,
                    ),
                ),
            )
        return _Decision(
            exactness=ExactnessClass.NUMERICALLY_EQUIVALENT,
            reasons=(
                _reason(
                    "DTYPE_CONVERSION_NUMERIC",
                    ReasonSeverity.REQUIREMENT,
                    "state_dtype",
                    "dtype conversion is not bitwise or semantically exact",
                    source.state_dtype,
                    destination.state_dtype,
                ),
            ),
            conversions=("dtype_convert", "layout_convert" if layout_changed else "copy"),
            obligations=(
                _obligation(
                    "verify-numeric-continuation",
                    VerificationKind.CONTINUATION,
                    "converted_state",
                    "compare logits, recurrent state, and emitted tokens over a bounded horizon",
                    request.numeric_tolerance,
                ),
            ),
        )

    if layout_changed or runtime_changed:
        reasons = []
        conversions = []
        if layout_changed:
            reasons.append(
                _reason(
                    "LAYOUT_ONLY_EXACT",
                    ReasonSeverity.INFO,
                    "physical_layout",
                    "physical layout differs while logical state semantics remain unchanged",
                    request.source_layout_fingerprint,
                    request.destination_layout_fingerprint,
                )
            )
            conversions.append("layout_convert")
        if runtime_changed:
            reasons.append(
                _reason(
                    "RUNTIME_BOUNDARY_SEMANTIC",
                    ReasonSeverity.INFO,
                    "runtime",
                    "runtime boundary prevents a bitwise-environment claim",
                    request.source_runtime.runtime_name,
                    request.destination_runtime.runtime_name,
                )
            )
        return _Decision(
            exactness=ExactnessClass.EXACT_SEMANTIC,
            reasons=tuple(reasons),
            conversions=tuple(conversions or ["copy"]),
            obligations=(
                _obligation(
                    "verify-exact-conversion",
                    VerificationKind.CONVERSION_EQUIVALENCE,
                    "portable_state",
                    "compare optimized conversion with the trusted canonical converter",
                ),
            ),
        )

    return _Decision(
        exactness=ExactnessClass.EXACT_BITWISE,
        reasons=(
            _reason(
                "IDENTICAL_COMPATIBILITY_DOMAIN",
                ReasonSeverity.INFO,
                "execution_state",
                "runtime adapter, dtype, quantization, and layout fingerprints match",
            ),
        ),
        conversions=("copy",),
        obligations=(
            _obligation(
                "verify-bitwise-state",
                VerificationKind.STRUCTURAL,
                "portable_state",
                "compare canonical state hashes before activation",
            ),
        ),
    )


def analyze_compatibility(request: CompatibilityRequest) -> CompatibilityDecision:
    """Return the smallest safe state action supported by supplied evidence."""

    missing_state = tuple(
        sorted(
            set(request.required_state_types)
            - set(request.destination_runtime.supported_state_types)
        )
    )
    unsupported_dtype = (
        request.destination.state_dtype not in request.destination_runtime.supported_dtypes
    )
    unsupported_quantization = (
        request.destination.quantization not in request.destination_runtime.supported_quantizations
    )
    runtime_contract_mismatch = (
        request.source_runtime.logical_state_contract
        != request.destination_runtime.logical_state_contract
    )
    capability_reasons: list[CompatibilityReason] = []
    if missing_state:
        capability_reasons.append(
            _reason(
                "UNSUPPORTED_STATE_TYPE",
                ReasonSeverity.BLOCKING,
                "runtime_capabilities",
                "destination runtime cannot import every required logical state type",
                *missing_state,
            )
        )
    if unsupported_dtype:
        capability_reasons.append(
            _reason(
                "UNSUPPORTED_DTYPE",
                ReasonSeverity.BLOCKING,
                "runtime_capabilities",
                "destination runtime does not support the declared state dtype",
                request.destination.state_dtype,
            )
        )
    if unsupported_quantization:
        capability_reasons.append(
            _reason(
                "UNSUPPORTED_QUANTIZATION",
                ReasonSeverity.BLOCKING,
                "runtime_capabilities",
                "destination runtime does not support the declared quantization",
                request.destination.quantization,
            )
        )
    if runtime_contract_mismatch:
        capability_reasons.append(
            _reason(
                "RUNTIME_LOGICAL_STATE_CONTRACT_MISMATCH",
                ReasonSeverity.BLOCKING,
                "runtime_capabilities",
                "runtime adapters do not declare the same logical-state semantics contract",
                request.source_runtime.logical_state_contract,
                request.destination_runtime.logical_state_contract,
            )
        )
    if capability_reasons:
        return _incompatible(tuple(capability_reasons), unsupported=missing_state)

    semantic_reasons = _semantic_mismatches(request)
    if semantic_reasons:
        return _incompatible(semantic_reasons)

    weight_decision = _weight_decision(request)
    if isinstance(weight_decision, CompatibilityDecision):
        return weight_decision
    representation = _representation_decision(request)
    if representation.exactness is ExactnessClass.INCOMPATIBLE:
        return _incompatible(representation.reasons)

    decision = representation
    if weight_decision is not None:
        if weight_decision.exactness is ExactnessClass.RECOMPUTATION_ASSISTED:
            combined_exactness = ExactnessClass.RECOMPUTATION_ASSISTED
        elif representation.exactness in {
            ExactnessClass.NUMERICALLY_EQUIVALENT,
            ExactnessClass.QUALITY_BOUNDED,
        }:
            combined_exactness = representation.exactness
        else:
            combined_exactness = ExactnessClass.EXACT_SEMANTIC
        decision = _Decision(
            exactness=combined_exactness,
            reasons=weight_decision.reasons + representation.reasons,
            conversions=representation.conversions,
            recomputation=weight_decision.recomputation,
            quality=representation.quality,
            obligations=weight_decision.obligations + representation.obligations,
            restrictions=weight_decision.restrictions + representation.restrictions,
        )

    if not _required_exactness_allows(request.required_exactness, decision.exactness):
        return _incompatible(
            (
                *decision.reasons,
                _reason(
                    "EXACTNESS_REQUIREMENT_UNSATISFIED",
                    ReasonSeverity.BLOCKING,
                    "exactness_contract",
                    "best known conversion is weaker than the requested exactness contract",
                    request.required_exactness.value,
                    decision.exactness.value,
                ),
            )
        )

    rejected: list[RejectedCompatibilityClass] = []
    if decision.exactness is not ExactnessClass.EXACT_BITWISE:
        rejected.append(
            RejectedCompatibilityClass(
                exactness_class=ExactnessClass.EXACT_BITWISE,
                reason_codes=tuple(reason.code for reason in decision.reasons),
            )
        )
    if decision.exactness not in {ExactnessClass.EXACT_BITWISE, ExactnessClass.EXACT_SEMANTIC}:
        rejected.append(
            RejectedCompatibilityClass(
                exactness_class=ExactnessClass.EXACT_SEMANTIC,
                reason_codes=tuple(reason.code for reason in decision.reasons),
            )
        )
    return CompatibilityDecision(
        compatibility_class=decision.exactness,
        safe=True,
        reasons=decision.reasons,
        rejected_classes=tuple(rejected),
        required_conversion=decision.conversions,
        required_recomputation=decision.recomputation,
        unsupported_state=(),
        quality_implications=decision.quality,
        verification_obligations=decision.obligations,
        migration_restrictions=decision.restrictions,
    )
