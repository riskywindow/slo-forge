"""Deterministic CPU fault injection and bounded callback execution."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from .models import (
    FaultCampaignResult,
    FaultExecutionResult,
    FaultKind,
    FaultMutation,
    FaultObservation,
    FaultPlan,
    FaultPlanRequest,
    FaultResponse,
    FaultSpec,
    InjectedFault,
    MutationOperation,
    canonical_digest,
)

FaultCallback = Callable[[InjectedFault], FaultObservation]

_ASSUMPTIONS = (
    "Fault callbacks are CPU-only, invoked at most once per planned fault, and time bounded.",
    "A callback error, timeout, missing callback, or response mismatch fails the campaign closed.",
    "Evidence hashes are content-addressed input claims and are not fabricated measurements.",
)


def compile_fault_plan(request: FaultPlanRequest) -> FaultPlan:
    """Canonicalize and seal a fault matrix independently of its input ordering."""

    faults = tuple(
        sorted(
            request.faults,
            key=lambda fault: (
                fault.activation.start_step,
                fault.stage.value,
                fault.kind.value,
                fault.fault_id,
            ),
        )
    )
    request_payload = request.model_dump(mode="json")
    request_payload["faults"] = [fault.model_dump(mode="json") for fault in faults]
    request_digest = canonical_digest(request_payload)
    assumptions = tuple(dict.fromkeys((*_ASSUMPTIONS, *request.assumptions)))
    unsealed = FaultPlan.model_construct(
        plan_id="0" * 64,
        request_digest=request_digest,
        request_id=request.request_id,
        seed=request.seed,
        horizon_steps=request.horizon_steps,
        callback_timeout_ms=request.callback_timeout_ms,
        max_callbacks=request.max_callbacks,
        faults=faults,
        assumptions=assumptions,
    )
    plan_id = canonical_digest(unsealed.model_dump(mode="json", exclude={"plan_id"}))
    return FaultPlan(
        plan_id=plan_id,
        request_digest=request_digest,
        request_id=request.request_id,
        seed=request.seed,
        horizon_steps=request.horizon_steps,
        callback_timeout_ms=request.callback_timeout_ms,
        max_callbacks=request.max_callbacks,
        faults=faults,
        assumptions=assumptions,
    )


def _derived_seed(seed: int, fault_id: str) -> int:
    digest = hashlib.sha256(f"sloforge.helix.fault/v1\0{seed}\0{fault_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _mutation_for(spec: FaultSpec) -> tuple[FaultMutation, ...]:
    mapping: dict[FaultKind, tuple[str, MutationOperation, str | None]] = {
        FaultKind.INCONSISTENT_BOUNDARY: (
            "capture.environment_event_watermark",
            MutationOperation.SET,
            "boundary+1",
        ),
        FaultKind.ILLEGAL_EFFECT: (
            "rollout.effect_class",
            MutationOperation.SET,
            "IRREVERSIBLE_WRITE",
        ),
        FaultKind.MISSING_LOGPROB: (
            "rollout.behavior_log_probability",
            MutationOperation.DELETE,
            None,
        ),
        FaultKind.POLICY_MISMATCH: (
            "rollout.policy_epoch_id",
            MutationOperation.SET,
            "mismatched-policy",
        ),
        FaultKind.POLICY_STALENESS: (
            "rollout.staleness_updates",
            MutationOperation.SET,
            str(max(1, round(spec.magnitude))),
        ),
        FaultKind.REWARD_CORRUPTION: (
            "reward.total_score",
            MutationOperation.SCALE,
            repr(spec.magnitude),
        ),
        FaultKind.DUPLICATE_REWARD: (
            "reward.submission",
            MutationOperation.DUPLICATE,
            "true",
        ),
        FaultKind.CHECKPOINT_FAILURE: (
            "branching.checkpoint",
            MutationOperation.INTERRUPT,
            "unavailable",
        ),
        FaultKind.LINEAGE_FAILURE: (
            "training.lineage_hash",
            MutationOperation.DELETE,
            None,
        ),
        FaultKind.TRAINING_FAILURE: (
            "training.step",
            MutationOperation.INTERRUPT,
            "injected-failure",
        ),
        FaultKind.QUALITY_REJECT: (
            "evaluation.quality_passed",
            MutationOperation.SET,
            "false",
        ),
        FaultKind.SERVING_REJECT: (
            "evaluation.serving_passed",
            MutationOperation.SET,
            "false",
        ),
        FaultKind.COMPATIBILITY_REJECT: (
            "evaluation.compatibility_passed",
            MutationOperation.SET,
            "false",
        ),
        FaultKind.PARTIAL_CHAMPION_POINTER: (
            "promotion.champion_pointer",
            MutationOperation.INTERRUPT,
            "after-pointer-update",
        ),
        FaultKind.TRAFFIC_SPIKE: (
            "resource.traffic",
            MutationOperation.SCALE,
            repr(1.0 + spec.magnitude),
        ),
        FaultKind.GPU_LOSS: (
            "resource.gpu_capacity",
            MutationOperation.SCALE,
            repr(1.0 - spec.magnitude),
        ),
    }
    field, operation, value = mapping[spec.kind]
    return (FaultMutation(field=field, operation=operation, encoded_value=value),)


class DeterministicFaultInjector:
    """Build hash-bound mutation instructions; it never performs real side effects."""

    def inject(self, spec: FaultSpec, *, seed: int) -> InjectedFault:
        derived_seed = _derived_seed(seed, spec.fault_id)
        mutations = _mutation_for(spec)
        unsealed = InjectedFault.model_construct(
            injection_id="0" * 64,
            fault_id=spec.fault_id,
            ground_truth_label=spec.ground_truth_label,
            kind=spec.kind,
            stage=spec.stage,
            activation=spec.activation,
            seed=derived_seed,
            expected_response=spec.expected_response,
            evidence_sha256=spec.evidence_sha256,
            mutations=mutations,
        )
        injection_id = canonical_digest(unsealed.model_dump(mode="json", exclude={"injection_id"}))
        return InjectedFault(
            injection_id=injection_id,
            fault_id=spec.fault_id,
            ground_truth_label=spec.ground_truth_label,
            kind=spec.kind,
            stage=spec.stage,
            activation=spec.activation,
            seed=derived_seed,
            expected_response=spec.expected_response,
            evidence_sha256=spec.evidence_sha256,
            mutations=mutations,
        )


def _failure_observation(response: FaultResponse, detail: str) -> FaultObservation:
    bounded_detail = detail[:2048] or response.value
    return FaultObservation(
        actual_response=response,
        detail=bounded_detail,
        evidence_sha256=canonical_digest({"response": response.value, "detail": bounded_detail}),
    )


class FaultRunner:
    """Run each planned callback once with a hard wall-time bound."""

    def __init__(self, injector: DeterministicFaultInjector | None = None) -> None:
        self.injector = injector or DeterministicFaultInjector()

    @staticmethod
    def _call_bounded(
        callback: FaultCallback,
        injection: InjectedFault,
        timeout_ms: int,
    ) -> FaultObservation:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="helix-fault")
        future = executor.submit(callback, injection)
        try:
            observation = future.result(timeout=timeout_ms / 1000.0)
        except FutureTimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            return _failure_observation(
                FaultResponse.CALLBACK_TIMEOUT,
                f"fault callback exceeded {timeout_ms} ms",
            )
        except Exception as error:
            executor.shutdown(wait=True, cancel_futures=True)
            return _failure_observation(
                FaultResponse.CALLBACK_ERROR,
                f"{type(error).__name__}: {error}",
            )
        executor.shutdown(wait=True, cancel_futures=True)
        if not isinstance(observation, FaultObservation):
            return _failure_observation(
                FaultResponse.CALLBACK_ERROR,
                "fault callback returned an invalid observation type",
            )
        return observation

    def run(
        self,
        plan: FaultPlan,
        callbacks: Mapping[FaultKind, FaultCallback],
    ) -> FaultCampaignResult:
        results: list[FaultExecutionResult] = []
        callback_count = 0
        for spec in plan.faults:
            injection = self.injector.inject(spec, seed=plan.seed)
            callback = callbacks.get(spec.kind)
            if callback is None:
                observation = _failure_observation(
                    FaultResponse.CALLBACK_MISSING,
                    f"no callback registered for {spec.kind.value}",
                )
            elif callback_count >= plan.max_callbacks:
                observation = _failure_observation(
                    FaultResponse.CALLBACK_ERROR,
                    "fault callback bound exhausted",
                )
            else:
                callback_count += 1
                observation = self._call_bounded(
                    callback,
                    injection,
                    plan.callback_timeout_ms,
                )
            results.append(
                FaultExecutionResult(
                    fault_id=spec.fault_id,
                    ground_truth_label=spec.ground_truth_label,
                    kind=spec.kind,
                    stage=spec.stage,
                    activation=spec.activation,
                    expected_response=spec.expected_response,
                    actual_response=observation.actual_response,
                    passed=observation.actual_response is spec.expected_response,
                    evidence_sha256=spec.evidence_sha256,
                    observation_evidence_sha256=observation.evidence_sha256,
                    injection_id=injection.injection_id,
                    detail=observation.detail,
                )
            )
        result_tuple = tuple(results)
        failed = tuple(result.fault_id for result in result_tuple if not result.passed)
        unsealed = FaultCampaignResult.model_construct(
            result_id="0" * 64,
            plan_id=plan.plan_id,
            seed=plan.seed,
            passed=not failed,
            callback_count=callback_count,
            results=result_tuple,
            failed_fault_ids=failed,
            assumptions=plan.assumptions,
        )
        result_id = canonical_digest(unsealed.model_dump(mode="json", exclude={"result_id"}))
        return FaultCampaignResult(
            result_id=result_id,
            plan_id=plan.plan_id,
            seed=plan.seed,
            passed=not failed,
            callback_count=callback_count,
            results=result_tuple,
            failed_fault_ids=failed,
            assumptions=plan.assumptions,
        )


def run_fault_plan(
    plan: FaultPlan,
    callbacks: Mapping[FaultKind, FaultCallback],
) -> FaultCampaignResult:
    return FaultRunner().run(plan, callbacks)


__all__ = [
    "DeterministicFaultInjector",
    "FaultCallback",
    "FaultRunner",
    "compile_fault_plan",
    "run_fault_plan",
]
