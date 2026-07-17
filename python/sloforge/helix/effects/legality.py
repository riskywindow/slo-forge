"""Fail-closed legality checks for real and speculative effects."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Effect, EffectClass


class IllegalEffectError(PermissionError):
    """An effect violates speculative execution or external-effect policy."""


@dataclass(frozen=True, slots=True)
class EffectLegalityPolicy:
    speculative: bool = False
    external_side_effects_enabled: bool = False
    expected_tenant_id: str | None = None


class EffectLegalityChecker:
    def check(self, effect: Effect, policy: EffectLegalityPolicy) -> tuple[str, ...]:
        reasons: list[str] = []
        effect.verify_identity()
        if policy.expected_tenant_id is not None and effect.tenant_id != policy.expected_tenant_id:
            reasons.append("effect tenant does not match the execution tenant")
        if effect.classification is EffectClass.READ_ONLY and not effect.stable_read:
            reasons.append("unstable reads cannot be replayed deterministically")
        if effect.classification is EffectClass.IDEMPOTENT_WRITE and not effect.idempotency_key:
            reasons.append("idempotent writes require an idempotency key")
        if effect.classification is EffectClass.COMPENSATABLE_WRITE and not effect.compensation:
            reasons.append("compensatable writes require a compensation recipe")
        if (
            effect.real_external
            and effect.classification
            not in {
                EffectClass.PURE,
                EffectClass.READ_ONLY,
            }
            and not policy.external_side_effects_enabled
        ):
            reasons.append("real external side effects are disabled")
        if (
            policy.speculative
            and effect.real_external
            and effect.classification
            in {EffectClass.IRREVERSIBLE_WRITE, EffectClass.EXTERNAL_UNKNOWN}
        ):
            reasons.append("speculation cannot execute irreversible or unknown real effects")
        return tuple(reasons)

    def require(self, effect: Effect, policy: EffectLegalityPolicy) -> None:
        reasons = self.check(effect, policy)
        if reasons:
            raise IllegalEffectError("; ".join(reasons))


def check_effect_legality(
    effect: Effect,
    *,
    speculative: bool,
    external_side_effects_enabled: bool = False,
    expected_tenant_id: str | None = None,
) -> tuple[str, ...]:
    return EffectLegalityChecker().check(
        effect,
        EffectLegalityPolicy(
            speculative=speculative,
            external_side_effects_enabled=external_side_effects_enabled,
            expected_tenant_id=expected_tenant_id,
        ),
    )


def require_effect_legal(
    effect: Effect,
    *,
    speculative: bool,
    external_side_effects_enabled: bool = False,
    expected_tenant_id: str | None = None,
) -> None:
    EffectLegalityChecker().require(
        effect,
        EffectLegalityPolicy(
            speculative=speculative,
            external_side_effects_enabled=external_side_effects_enabled,
            expected_tenant_id=expected_tenant_id,
        ),
    )
