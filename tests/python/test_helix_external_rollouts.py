from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, TypeAlias

import pytest
from pydantic import ValidationError

from sloforge.continuum.adapters.external import (
    ExternalRuntimeApiError,
    RuntimePackageView,
)
from sloforge.continuum.adapters.genesis import GenesisRuntimeDescriptor
from sloforge.continuum.adapters.pytorch import PYTORCH_REQUIREMENTS
from sloforge.continuum.adapters.sdk import UnsupportedCapabilityError
from sloforge.continuum.adapters.sglang import SGLANG_REQUIREMENTS
from sloforge.continuum.adapters.vllm import VLLM_REQUIREMENTS
from sloforge.helix.rollouts import (
    ExternalRolloutConfig,
    ExternalRolloutContractError,
    ExternalRolloutPlan,
    ExternalRolloutRequest,
    ExternalRolloutResult,
    ExternalRuntimeKind,
    ExternalTokenRecord,
    GenesisRolloutAdapter,
    PriorStateReference,
    PyTorchRolloutAdapter,
    SglangRolloutAdapter,
    StateReuseOutcome,
    StateReuseStrategy,
    ValidatedExternalRollout,
    VllmRolloutAdapter,
    build_recomputation_evidence,
)

Adapter: TypeAlias = (
    PyTorchRolloutAdapter | GenesisRolloutAdapter | VllmRolloutAdapter | SglangRolloutAdapter
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _view(runtime: str, version: str, requirements: tuple[str, ...]) -> RuntimePackageView:
    return RuntimePackageView(
        distribution_name=runtime,
        import_name=runtime,
        version=version,
        available_symbols=frozenset(requirements),
        source="static_fixture",
    )


def _adapters(tmp_path: Path) -> tuple[Adapter, ...]:
    genesis = GenesisRuntimeDescriptor(
        config_path=tmp_path / "runtime_config.json",
        schema_version="1.0.0",
        runtime_id=_digest("genesis-runtime"),
        package_hash=_digest("genesis-package"),
        generation_seed=77,
        state_allocator_layout="paged",
        page_bytes=4096,
    )
    return (
        PyTorchRolloutAdapter(_view("torch", "2.13.0", PYTORCH_REQUIREMENTS)),
        GenesisRolloutAdapter.from_descriptor(genesis),
        VllmRolloutAdapter(_view("vllm", "0.23.0", VLLM_REQUIREMENTS)),
        SglangRolloutAdapter(_view("sglang", "0.5.12", SGLANG_REQUIREMENTS)),
    )


def _request(
    *,
    prior_state: PriorStateReference | None = None,
    maximum_actions: int = 0,
) -> ExternalRolloutRequest:
    return ExternalRolloutRequest(
        request_id="rollout-1",
        prompt="Choose the safe bounded action.",
        policy_epoch_id="policy-main@7",
        behavior_policy_epoch_id="policy-main@7",
        config=ExternalRolloutConfig(
            seed=73129,
            timeout_seconds=5.0,
            cancellation_grace_seconds=0.5,
            maximum_new_tokens=4,
            maximum_actions=maximum_actions,
        ),
        prior_state=prior_state,
    )


def _prior(
    strategy: Literal[
        StateReuseStrategy.EXACT_PORTABLE,
        StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY,
    ],
) -> PriorStateReference:
    return PriorStateReference(
        source_runtime=ExternalRuntimeKind.VLLM,
        source_policy_epoch_id="policy-main@6",
        source_state_digest=_digest("source-state"),
        token_history_digest=(
            _digest("token-history")
            if strategy is StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY
            else None
        ),
        compatibility_report_digest=_digest("compatibility-report"),
        requested_strategy=strategy,
        recomputation_implementation_digest=(
            _digest("recompute-implementation")
            if strategy is StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY
            else None
        ),
    )


def test_static_package_views_describe_capabilities_without_optional_imports(
    tmp_path: Path,
) -> None:
    adapters = _adapters(tmp_path)
    assert {adapter.descriptor.runtime for adapter in adapters} == {
        ExternalRuntimeKind.PYTORCH,
        ExternalRuntimeKind.GENESIS,
        ExternalRuntimeKind.VLLM,
        ExternalRuntimeKind.SGLANG,
    }
    for adapter in adapters:
        descriptor = adapter.descriptor
        assert descriptor.ready
        assert not descriptor.portable_exact_state_export
        assert descriptor.state_reuse_requirement == "explicit_recomputation"
        assert descriptor.execution_contract == "caller_supplied_bounded_executor"
        assert descriptor.public_api_evidence
        if descriptor.runtime is not ExternalRuntimeKind.GENESIS:
            assert not descriptor.probe_exercised


def test_missing_public_capability_fails_before_a_rollout_is_planned() -> None:
    incomplete = _view("vllm", "0.23.0", VLLM_REQUIREMENTS[:-1])
    adapter = VllmRolloutAdapter(incomplete)
    assert not adapter.descriptor.ready
    with pytest.raises(ExternalRuntimeApiError, match="missing required public API"):
        adapter.prepare(_request())


def test_all_external_backends_reject_exact_portable_reuse_and_require_recomputation(
    tmp_path: Path,
) -> None:
    exact = _request(prior_state=_prior(StateReuseStrategy.EXACT_PORTABLE))
    recompute = _request(prior_state=_prior(StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY))

    for adapter in _adapters(tmp_path):
        with pytest.raises(UnsupportedCapabilityError):
            adapter.prepare(exact)
        plan = adapter.prepare(recompute)
        assert plan.state_reuse.strategy is StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY
        assert plan.state_reuse.recomputation_required
        assert not plan.state_reuse.portable_state_reused
        assert plan.state_reuse.recomputation_seed == recompute.config.seed
        assert plan.state_reuse.target_policy_epoch_id == recompute.policy_epoch_id


def test_strict_policy_and_behavior_logprob_provenance_are_mandatory() -> None:
    raw = _request().model_dump()
    raw["behavior_policy_epoch_id"] = "policy-main@other"
    with pytest.raises(ValidationError, match="behavior policy"):
        ExternalRolloutRequest.model_validate(raw, strict=True)

    token = ExternalTokenRecord(
        token_index=0,
        token_id=42,
        policy_epoch_id="policy-main@7",
        behavior_log_probability=-0.25,
    )
    token_raw = token.model_dump()
    del token_raw["behavior_log_probability"]
    with pytest.raises(ValidationError, match="behavior_log_probability"):
        ExternalTokenRecord.model_validate(token_raw, strict=True)

    with pytest.raises(ValidationError, match="mixed policy epochs"):
        ExternalRolloutResult(
            request_id="rollout-1",
            plan_digest=_digest("plan"),
            runtime=ExternalRuntimeKind.PYTORCH,
            runtime_version="2.13.0",
            policy_epoch_id="policy-main@7",
            seed=1,
            tokens=(token.model_copy(update={"policy_epoch_id": "policy-main@6"}),),
            state_reuse=StateReuseOutcome(
                strategy=StateReuseStrategy.FRESH,
                output_state_digest=_digest("fresh-state"),
            ),
            elapsed_seconds=0.1,
        )


class _FixtureExecutor:
    def execute(
        self,
        plan: ExternalRolloutPlan,
        *,
        timeout_seconds: float,
        cancellation_grace_seconds: float,
    ) -> ExternalRolloutResult:
        assert timeout_seconds == plan.request.config.timeout_seconds
        assert cancellation_grace_seconds == plan.request.config.cancellation_grace_seconds
        runtime_version = plan.descriptor.runtime_version
        assert runtime_version is not None
        return ExternalRolloutResult(
            request_id=plan.request.request_id,
            plan_digest=plan.plan_digest,
            runtime=plan.descriptor.runtime,
            runtime_version=runtime_version,
            policy_epoch_id=plan.request.policy_epoch_id,
            seed=plan.request.config.seed,
            tokens=(
                ExternalTokenRecord(
                    token_index=0,
                    token_id=17,
                    policy_epoch_id=plan.request.policy_epoch_id,
                    behavior_log_probability=-0.3,
                ),
            ),
            state_reuse=StateReuseOutcome(
                strategy=StateReuseStrategy.FRESH,
                output_state_digest=_digest("fresh-output-state"),
            ),
            elapsed_seconds=0.25,
        )


def test_bounded_executor_result_is_validated_without_importing_pytorch() -> None:
    adapter = PyTorchRolloutAdapter(_view("torch", "2.13.0", PYTORCH_REQUIREMENTS))
    result = adapter.run_with(_request(), _FixtureExecutor())

    assert isinstance(result, ValidatedExternalRollout)
    assert result.training_eligible
    assert result.behavior_log_probability_provenance_complete
    assert result.result.tokens[0].behavior_log_probability == -0.3
    assert result.result.seed == 73129


def test_recomputation_action_evidence_must_match_the_plan(tmp_path: Path) -> None:
    adapter = VllmRolloutAdapter(_view("vllm", "0.23.0", VLLM_REQUIREMENTS))
    request = _request(prior_state=_prior(StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY))
    plan = adapter.prepare(request)
    evidence = build_recomputation_evidence(
        source_state_digest=_digest("source-state"),
        token_history_digest=_digest("token-history"),
        target_policy_epoch_id="policy-main@7",
        implementation_digest=_digest("recompute-implementation"),
        seed=73129,
        output_state_digest=_digest("recomputed-state"),
    )
    result = ExternalRolloutResult(
        request_id=request.request_id,
        plan_digest=plan.plan_digest,
        runtime=ExternalRuntimeKind.VLLM,
        runtime_version="0.23.0",
        policy_epoch_id=request.policy_epoch_id,
        seed=request.config.seed,
        tokens=(
            ExternalTokenRecord(
                token_index=0,
                token_id=9,
                policy_epoch_id=request.policy_epoch_id,
                behavior_log_probability=-0.1,
            ),
        ),
        state_reuse=StateReuseOutcome(
            strategy=StateReuseStrategy.RECOMPUTE_FROM_TOKEN_HISTORY,
            output_state_digest=_digest("recomputed-state"),
            recomputation=evidence,
        ),
        elapsed_seconds=0.5,
    )
    assert adapter.validate_result(plan, result).state_reuse_validated

    wrong = result.model_copy(
        update={
            "state_reuse": result.state_reuse.model_copy(
                update={
                    "recomputation": evidence.model_copy(update={"seed": 1}),
                }
            )
        }
    )
    with pytest.raises(ExternalRolloutContractError, match="explicit plan"):
        adapter.validate_result(plan, wrong)


def test_seed_timeout_and_output_sizes_are_bounded() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 600"):
        ExternalRolloutConfig(seed=1, timeout_seconds=601.0, maximum_new_tokens=1)
    with pytest.raises(ValidationError, match="less than or equal to 184467"):
        ExternalRolloutConfig(
            seed=2**64,
            timeout_seconds=1.0,
            maximum_new_tokens=1,
        )
