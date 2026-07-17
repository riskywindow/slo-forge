"""Capability-gated wrappers for optional external Helix rollout runtimes.

The wrappers deliberately do not discover private engine state or import optional
packages during module import.  Continuum probes gate the public surfaces, while a
caller-supplied bounded executor owns model-specific generation.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.continuum.adapters.external import (
    AdapterProbe,
    CapabilityName,
    IntegrationStatus,
    RuntimePackageView,
)
from sloforge.continuum.adapters.genesis import (
    GenesisRuntimeBinding,
    GenesisRuntimeDescriptor,
)
from sloforge.continuum.adapters.pytorch import PyTorchRuntimeBinding, probe_pytorch
from sloforge.continuum.adapters.sdk import UnsupportedCapabilityError
from sloforge.continuum.adapters.sglang import SglangRuntimeBinding, probe_sglang
from sloforge.continuum.adapters.vllm import VllmRuntimeBinding, probe_vllm

_U64_MAX = 2**64 - 1
_MAX_TOKENS = 4096
_MAX_ACTIONS = 256
_MAX_TIMEOUT_SECONDS = 600.0

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,255}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0, le=_U64_MAX)]
LogProbability = Annotated[float, Field(le=0.0, allow_inf_nan=False)]


class ExternalRolloutContractError(ValueError):
    """External output or requested reuse violates the Helix rollout contract."""


class _ExternalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ExternalRuntimeKind(StrEnum):
    PYTORCH = "pytorch"
    GENESIS = "genesis"
    VLLM = "vllm"
    SGLANG = "sglang"


class StateReuseStrategy(StrEnum):
    FRESH = "fresh"
    EXACT_PORTABLE = "exact_portable"
    RECOMPUTE_FROM_TOKEN_HISTORY = "recompute_from_token_history"


class RuntimeRolloutDescriptor(_ExternalModel):
    schema_version: Literal["sloforge.helix.external-rollout-adapter/v1"] = (
        "sloforge.helix.external-rollout-adapter/v1"
    )
    runtime: ExternalRuntimeKind
    runtime_version: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None
    continuum_adapter_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    integration_status: IntegrationStatus
    ready: bool
    probe_exercised: bool
    public_api_only: Literal[True] = True
    capabilities: Annotated[tuple[CapabilityName, ...], Field(max_length=64)]
    required_rollout_capabilities: Annotated[
        tuple[CapabilityName, ...], Field(min_length=1, max_length=16)
    ]
    public_api_evidence: Annotated[tuple[str, ...], Field(min_length=1, max_length=64)]
    missing_requirements: Annotated[tuple[str, ...], Field(max_length=64)]
    probe_build_hash: Digest
    execution_contract: Literal["caller_supplied_bounded_executor"] = (
        "caller_supplied_bounded_executor"
    )
    portable_exact_state_export: Literal[False] = False
    state_reuse_requirement: Literal["explicit_recomputation"] = "explicit_recomputation"

    @model_validator(mode="after")
    def consistent_probe(self) -> Self:
        required = set(self.required_rollout_capabilities)
        expected_ready = self.integration_status is IntegrationStatus.READY and required.issubset(
            self.capabilities
        )
        if self.ready != expected_ready:
            raise ValueError("rollout descriptor readiness contradicts probed capabilities")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("rollout descriptor capabilities contain duplicates")
        return self


class ExternalRolloutConfig(_ExternalModel):
    seed: NonNegativeInt
    timeout_seconds: Annotated[float, Field(gt=0.0, le=_MAX_TIMEOUT_SECONDS, allow_inf_nan=False)]
    cancellation_grace_seconds: Annotated[float, Field(gt=0.0, le=30.0, allow_inf_nan=False)] = 1.0
    maximum_new_tokens: Annotated[int, Field(gt=0, le=_MAX_TOKENS)]
    maximum_actions: Annotated[int, Field(ge=0, le=_MAX_ACTIONS)] = 0

    @model_validator(mode="after")
    def bounded_cancellation(self) -> Self:
        if self.cancellation_grace_seconds > self.timeout_seconds:
            raise ValueError("cancellation grace cannot exceed the rollout timeout")
        return self


class PriorStateReference(_ExternalModel):
    source_runtime: ExternalRuntimeKind
    source_policy_epoch_id: Identifier
    source_state_digest: Digest
    token_history_digest: Digest | None = None
    compatibility_report_digest: Digest
    requested_strategy: Literal[
        StateReuseStrategy.EXACT_PORTABLE,
        StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY,
    ]
    recomputation_implementation_digest: Digest | None = None

    @model_validator(mode="after")
    def explicit_recomputation_inputs(self) -> Self:
        if self.requested_strategy is StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY:
            if (
                self.token_history_digest is None
                or self.recomputation_implementation_digest is None
            ):
                raise ValueError(
                    "state recomputation requires token history and an implementation digest"
                )
        elif (
            self.token_history_digest is not None
            or self.recomputation_implementation_digest is not None
        ):
            raise ValueError("exact state reuse cannot carry undeclared recomputation inputs")
        return self


class ExternalRolloutRequest(_ExternalModel):
    schema_version: Literal["sloforge.helix.external-rollout-request/v1"] = (
        "sloforge.helix.external-rollout-request/v1"
    )
    request_id: Identifier
    prompt: Annotated[str, StringConstraints(min_length=1, max_length=131_072)]
    policy_consistency: Literal["strict"] = "strict"
    policy_epoch_id: Identifier
    behavior_policy_epoch_id: Identifier
    require_behavior_log_probabilities: Literal[True] = True
    config: ExternalRolloutConfig
    prior_state: PriorStateReference | None = None

    @model_validator(mode="after")
    def strict_behavior_policy(self) -> Self:
        if self.behavior_policy_epoch_id != self.policy_epoch_id:
            raise ValueError("strict rollout behavior policy must equal the requested policy epoch")
        return self


class StateReuseDirective(_ExternalModel):
    strategy: StateReuseStrategy
    source_runtime: ExternalRuntimeKind | None = None
    source_policy_epoch_id: Identifier | None = None
    target_policy_epoch_id: Identifier
    source_state_digest: Digest | None = None
    token_history_digest: Digest | None = None
    compatibility_report_digest: Digest | None = None
    recomputation_implementation_digest: Digest | None = None
    recomputation_seed: NonNegativeInt | None = None
    portable_state_reused: Literal[False] = False
    recomputation_required: bool
    reason: Annotated[str, StringConstraints(min_length=1, max_length=1024)]

    @model_validator(mode="after")
    def complete_directive(self) -> Self:
        source_values = (
            self.source_runtime,
            self.source_policy_epoch_id,
            self.source_state_digest,
            self.compatibility_report_digest,
        )
        if self.strategy is StateReuseStrategy.FRESH:
            if any(value is not None for value in source_values) or self.recomputation_required:
                raise ValueError("fresh rollout cannot carry prior-state provenance")
            if any(
                value is not None
                for value in (
                    self.token_history_digest,
                    self.recomputation_implementation_digest,
                    self.recomputation_seed,
                )
            ):
                raise ValueError("fresh rollout cannot carry recomputation inputs")
        elif self.strategy is StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY:
            if any(value is None for value in source_values):
                raise ValueError("recomputation directive requires complete source provenance")
            if (
                self.token_history_digest is None
                or self.recomputation_implementation_digest is None
                or self.recomputation_seed is None
                or not self.recomputation_required
            ):
                raise ValueError("recomputation directive is incomplete")
        else:
            raise ValueError("external adapters cannot claim exact portable state reuse")
        return self


class ExternalRolloutPlan(_ExternalModel):
    schema_version: Literal["sloforge.helix.external-rollout-plan/v1"] = (
        "sloforge.helix.external-rollout-plan/v1"
    )
    plan_digest: Digest
    descriptor: RuntimeRolloutDescriptor
    request: ExternalRolloutRequest
    state_reuse: StateReuseDirective

    @model_validator(mode="after")
    def valid_plan(self) -> Self:
        if self.state_reuse.target_policy_epoch_id != self.request.policy_epoch_id:
            raise ValueError("state-reuse target does not match the rollout policy")
        expected = self.model_dump(mode="json", exclude={"plan_digest"})
        if _canonical_hash(expected) != self.plan_digest:
            raise ValueError("external rollout plan digest is invalid")
        return self


class ExternalTokenRecord(_ExternalModel):
    token_index: Annotated[int, Field(ge=0, lt=_MAX_TOKENS)]
    token_id: NonNegativeInt
    policy_epoch_id: Identifier
    behavior_log_probability: LogProbability


class ExternalActionRecord(_ExternalModel):
    action_index: Annotated[int, Field(ge=0, lt=_MAX_ACTIONS)]
    action: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    policy_epoch_id: Identifier
    behavior_log_probability: LogProbability


class StateRecomputationEvidence(_ExternalModel):
    source_state_digest: Digest
    token_history_digest: Digest
    target_policy_epoch_id: Identifier
    implementation_digest: Digest
    seed: NonNegativeInt
    output_state_digest: Digest
    evidence_digest: Digest

    @model_validator(mode="after")
    def verify_evidence_digest(self) -> Self:
        expected = self.model_dump(mode="json", exclude={"evidence_digest"})
        if _canonical_hash(expected) != self.evidence_digest:
            raise ValueError("state recomputation evidence digest is invalid")
        return self


class StateReuseOutcome(_ExternalModel):
    strategy: StateReuseStrategy
    portable_state_reused: Literal[False] = False
    output_state_digest: Digest
    recomputation: StateRecomputationEvidence | None = None

    @model_validator(mode="after")
    def complete_outcome(self) -> Self:
        if self.strategy is StateReuseStrategy.FRESH:
            if self.recomputation is not None:
                raise ValueError("fresh rollout cannot claim state recomputation")
        elif self.strategy is StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY:
            if self.recomputation is None:
                raise ValueError("recomputed rollout requires explicit action evidence")
            if self.recomputation.output_state_digest != self.output_state_digest:
                raise ValueError("recomputation evidence does not match output state")
        else:
            raise ValueError("external rollout outcome cannot claim exact portable state reuse")
        return self


class ExternalRolloutResult(_ExternalModel):
    schema_version: Literal["sloforge.helix.external-rollout-result/v1"] = (
        "sloforge.helix.external-rollout-result/v1"
    )
    request_id: Identifier
    plan_digest: Digest
    runtime: ExternalRuntimeKind
    runtime_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    policy_consistency: Literal["strict"] = "strict"
    policy_epoch_id: Identifier
    seed: NonNegativeInt
    tokens: Annotated[tuple[ExternalTokenRecord, ...], Field(max_length=_MAX_TOKENS)] = ()
    actions: Annotated[tuple[ExternalActionRecord, ...], Field(max_length=_MAX_ACTIONS)] = ()
    state_reuse: StateReuseOutcome
    elapsed_seconds: Annotated[float, Field(ge=0.0, le=_MAX_TIMEOUT_SECONDS, allow_inf_nan=False)]
    terminal_status: Literal["completed"] = "completed"

    @model_validator(mode="after")
    def strict_provenance(self) -> Self:
        if not self.tokens and not self.actions:
            raise ValueError("external rollout must contain a token or action")
        if [item.token_index for item in self.tokens] != list(range(len(self.tokens))):
            raise ValueError("external token indexes must be contiguous from zero")
        if [item.action_index for item in self.actions] != list(range(len(self.actions))):
            raise ValueError("external action indexes must be contiguous from zero")
        if any(item.policy_epoch_id != self.policy_epoch_id for item in self.tokens):
            raise ValueError("strict external tokens contain mixed policy epochs")
        if any(item.policy_epoch_id != self.policy_epoch_id for item in self.actions):
            raise ValueError("strict external actions contain mixed policy epochs")
        return self


class ValidatedExternalRollout(_ExternalModel):
    validation_digest: Digest
    plan: ExternalRolloutPlan
    result: ExternalRolloutResult
    behavior_log_probability_provenance_complete: Literal[True] = True
    state_reuse_validated: Literal[True] = True
    training_eligible: Literal[True] = True

    @model_validator(mode="after")
    def verify_validation_digest(self) -> Self:
        expected = self.model_dump(mode="json", exclude={"validation_digest"})
        if _canonical_hash(expected) != self.validation_digest:
            raise ValueError("validated external rollout digest is invalid")
        return self


class BoundedRolloutExecutor(Protocol):
    """Model-specific executor that must enforce the supplied timeout and cancellation."""

    def execute(
        self,
        plan: ExternalRolloutPlan,
        *,
        timeout_seconds: float,
        cancellation_grace_seconds: float,
    ) -> ExternalRolloutResult: ...


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_recomputation_evidence(
    *,
    source_state_digest: str,
    token_history_digest: str,
    target_policy_epoch_id: str,
    implementation_digest: str,
    seed: int,
    output_state_digest: str,
) -> StateRecomputationEvidence:
    """Seal explicit recomputation action evidence without invoking a runtime."""

    body = {
        "source_state_digest": source_state_digest,
        "token_history_digest": token_history_digest,
        "target_policy_epoch_id": target_policy_epoch_id,
        "implementation_digest": implementation_digest,
        "seed": seed,
        "output_state_digest": output_state_digest,
    }
    return StateRecomputationEvidence(
        source_state_digest=source_state_digest,
        token_history_digest=token_history_digest,
        target_policy_epoch_id=target_policy_epoch_id,
        implementation_digest=implementation_digest,
        seed=seed,
        output_state_digest=output_state_digest,
        evidence_digest=_canonical_hash(body),
    )


def _descriptor(
    runtime: ExternalRuntimeKind,
    probe: AdapterProbe,
    required: tuple[CapabilityName, ...],
) -> RuntimeRolloutDescriptor:
    capabilities = tuple(sorted(probe.capabilities, key=lambda item: item.value))
    return RuntimeRolloutDescriptor(
        runtime=runtime,
        runtime_version=probe.runtime_version,
        continuum_adapter_version=probe.adapter_version,
        integration_status=probe.status,
        ready=(
            probe.status is IntegrationStatus.READY and set(required).issubset(probe.capabilities)
        ),
        probe_exercised=probe.exercised,
        capabilities=capabilities,
        required_rollout_capabilities=required,
        public_api_evidence=probe.evidence,
        missing_requirements=probe.missing_requirements,
        probe_build_hash=probe.build_hash,
    )


class _ExternalRolloutAdapter:
    def __init__(
        self,
        *,
        probe: AdapterProbe,
        descriptor: RuntimeRolloutDescriptor,
    ) -> None:
        self.probe = probe
        self.descriptor = descriptor

    def _require_capabilities(self) -> None:
        for capability in self.descriptor.required_rollout_capabilities:
            self.probe.require_capability(capability)

    def _require_portable_execution_state_export(self) -> None:
        raise ExternalRolloutContractError(
            "runtime does not declare portable execution-state export"
        )

    def prepare(self, request: ExternalRolloutRequest) -> ExternalRolloutPlan:
        self._require_capabilities()
        prior = request.prior_state
        if prior is None:
            directive = StateReuseDirective(
                strategy=StateReuseStrategy.FRESH,
                target_policy_epoch_id=request.policy_epoch_id,
                recomputation_required=False,
                reason="fresh external rollout has no prior execution state",
            )
        elif prior.requested_strategy is StateReuseStrategy.EXACT_PORTABLE:
            self._require_portable_execution_state_export()
            raise AssertionError("portable state export unexpectedly returned")
        else:
            if (
                prior.token_history_digest is None
                or prior.recomputation_implementation_digest is None
            ):
                raise ExternalRolloutContractError(
                    "explicit state recomputation inputs are incomplete"
                )
            directive = StateReuseDirective(
                strategy=StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY,
                source_runtime=prior.source_runtime,
                source_policy_epoch_id=prior.source_policy_epoch_id,
                target_policy_epoch_id=request.policy_epoch_id,
                source_state_digest=prior.source_state_digest,
                token_history_digest=prior.token_history_digest,
                compatibility_report_digest=prior.compatibility_report_digest,
                recomputation_implementation_digest=prior.recomputation_implementation_digest,
                recomputation_seed=request.config.seed,
                recomputation_required=True,
                reason=(
                    "portable exact state export is unavailable; recompute from explicit token "
                    "history under the target policy"
                ),
            )
        body = {
            "schema_version": "sloforge.helix.external-rollout-plan/v1",
            "descriptor": self.descriptor.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "state_reuse": directive.model_dump(mode="json"),
        }
        return ExternalRolloutPlan(
            plan_digest=_canonical_hash(body),
            descriptor=self.descriptor,
            request=request,
            state_reuse=directive,
        )

    def validate_result(
        self,
        plan: ExternalRolloutPlan,
        result: ExternalRolloutResult,
    ) -> ValidatedExternalRollout:
        try:
            checked_plan = ExternalRolloutPlan.model_validate(plan.model_dump(), strict=True)
        except ValueError as exc:
            raise ExternalRolloutContractError(
                "external rollout plan failed strict integrity validation"
            ) from exc
        plan = checked_plan
        request = plan.request
        if plan.descriptor != self.descriptor:
            raise ExternalRolloutContractError("rollout plan belongs to another adapter")
        if result.plan_digest != plan.plan_digest or result.request_id != request.request_id:
            raise ExternalRolloutContractError("external result does not match its rollout plan")
        if result.runtime is not self.descriptor.runtime:
            raise ExternalRolloutContractError("external runtime silently changed")
        if result.runtime_version != self.descriptor.runtime_version:
            raise ExternalRolloutContractError("external runtime version silently changed")
        if result.policy_epoch_id != request.policy_epoch_id:
            raise ExternalRolloutContractError("external result used a different policy epoch")
        if result.seed != request.config.seed:
            raise ExternalRolloutContractError(
                "external result used a different deterministic seed"
            )
        if result.elapsed_seconds > request.config.timeout_seconds:
            raise ExternalRolloutContractError("external rollout exceeded its bounded timeout")
        if len(result.tokens) > request.config.maximum_new_tokens:
            raise ExternalRolloutContractError("external rollout exceeded its token bound")
        if len(result.actions) > request.config.maximum_actions:
            raise ExternalRolloutContractError("external rollout exceeded its action bound")
        if result.state_reuse.strategy is not plan.state_reuse.strategy:
            raise ExternalRolloutContractError("external state-reuse strategy changed")

        if plan.state_reuse.strategy is StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY:
            evidence = result.state_reuse.recomputation
            if evidence is None:
                raise ExternalRolloutContractError("external recomputation evidence is missing")
            expected = (
                plan.state_reuse.source_state_digest,
                plan.state_reuse.token_history_digest,
                plan.state_reuse.target_policy_epoch_id,
                plan.state_reuse.recomputation_implementation_digest,
                plan.state_reuse.recomputation_seed,
            )
            actual = (
                evidence.source_state_digest,
                evidence.token_history_digest,
                evidence.target_policy_epoch_id,
                evidence.implementation_digest,
                evidence.seed,
            )
            if actual != expected:
                raise ExternalRolloutContractError(
                    "external recomputation evidence does not match the explicit plan"
                )
        elif result.state_reuse.recomputation is not None:
            raise ExternalRolloutContractError("fresh rollout contains hidden recomputation")

        try:
            checked_result = ExternalRolloutResult.model_validate(result.model_dump(), strict=True)
        except ValueError as exc:
            raise ExternalRolloutContractError(
                "external rollout result failed strict integrity validation"
            ) from exc
        result = checked_result

        body = {
            "plan": plan.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "behavior_log_probability_provenance_complete": True,
            "state_reuse_validated": True,
            "training_eligible": True,
        }
        return ValidatedExternalRollout(
            validation_digest=_canonical_hash(body),
            plan=plan,
            result=result,
        )

    def run_with(
        self,
        request: ExternalRolloutRequest,
        executor: BoundedRolloutExecutor,
    ) -> ValidatedExternalRollout:
        plan = self.prepare(request)
        result = executor.execute(
            plan,
            timeout_seconds=request.config.timeout_seconds,
            cancellation_grace_seconds=request.config.cancellation_grace_seconds,
        )
        return self.validate_result(plan, result)


class PyTorchRolloutAdapter(_ExternalRolloutAdapter):
    def __init__(self, view: RuntimePackageView | None = None) -> None:
        probe = probe_pytorch(view)
        self.binding = PyTorchRuntimeBinding(probe)
        super().__init__(
            probe=probe,
            descriptor=_descriptor(
                ExternalRuntimeKind.PYTORCH,
                probe,
                (CapabilityName.RUNTIME_INSPECTION, CapabilityName.RNG_STATE),
            ),
        )

    def _require_portable_execution_state_export(self) -> None:
        self.probe.require_ready(operation="portable_execution_state_export")
        raise UnsupportedCapabilityError(
            (
                "PyTorch exposes explicitly supplied tensors and RNG state, not a complete "
                "portable live-rollout state export"
            ),
            operation="portable_execution_state_export",
        )


class GenesisRolloutAdapter(_ExternalRolloutAdapter):
    def __init__(self, binding: GenesisRuntimeBinding) -> None:
        self.binding = binding
        probe = binding.probe
        super().__init__(
            probe=probe,
            descriptor=_descriptor(
                ExternalRuntimeKind.GENESIS,
                probe,
                (CapabilityName.BOUNDED_STREAMING, CapabilityName.CANCELLATION),
            ),
        )

    @classmethod
    def from_descriptor(cls, descriptor: GenesisRuntimeDescriptor) -> GenesisRolloutAdapter:
        return cls(GenesisRuntimeBinding(descriptor))

    @classmethod
    def from_config(cls, config_path: Path) -> GenesisRolloutAdapter:
        return cls(GenesisRuntimeBinding.from_config(config_path))

    def _require_portable_execution_state_export(self) -> None:
        self.binding.require_portable_execution_state_export()


class VllmRolloutAdapter(_ExternalRolloutAdapter):
    def __init__(self, view: RuntimePackageView | None = None) -> None:
        probe = probe_vllm(view)
        self.binding = VllmRuntimeBinding(probe)
        super().__init__(
            probe=probe,
            descriptor=_descriptor(
                ExternalRuntimeKind.VLLM,
                probe,
                (
                    CapabilityName.KV_TRANSFER_CONFIGURATION,
                    CapabilityName.KV_CONNECTOR_V1,
                ),
            ),
        )

    def _require_portable_execution_state_export(self) -> None:
        self.binding.require_portable_execution_state_export()


class SglangRolloutAdapter(_ExternalRolloutAdapter):
    def __init__(self, view: RuntimePackageView | None = None) -> None:
        probe = probe_sglang(view)
        self.binding = SglangRuntimeBinding(probe)
        super().__init__(
            probe=probe,
            descriptor=_descriptor(
                ExternalRuntimeKind.SGLANG,
                probe,
                (CapabilityName.PD_DISAGGREGATION_CONFIGURATION,),
            ),
        )

    def _require_portable_execution_state_export(self) -> None:
        self.binding.require_portable_execution_state_export()


VLLMRolloutAdapter = VllmRolloutAdapter
SGLangRolloutAdapter = SglangRolloutAdapter


__all__ = [
    "BoundedRolloutExecutor",
    "ExternalActionRecord",
    "ExternalRolloutConfig",
    "ExternalRolloutContractError",
    "ExternalRolloutPlan",
    "ExternalRolloutRequest",
    "ExternalRolloutResult",
    "ExternalRuntimeKind",
    "ExternalTokenRecord",
    "GenesisRolloutAdapter",
    "PriorStateReference",
    "PyTorchRolloutAdapter",
    "RuntimeRolloutDescriptor",
    "SGLangRolloutAdapter",
    "SglangRolloutAdapter",
    "StateRecomputationEvidence",
    "StateReuseDirective",
    "StateReuseOutcome",
    "StateReuseStrategy",
    "VLLMRolloutAdapter",
    "ValidatedExternalRollout",
    "VllmRolloutAdapter",
    "build_recomputation_evidence",
]
