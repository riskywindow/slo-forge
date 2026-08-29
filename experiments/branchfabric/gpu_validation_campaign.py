"""Deterministic same-container campaign orchestration for Experiment 003.

The module is deliberately standard-library only.  A CUDA-clean caller injects
``gpu_cow_controller.run_controller`` and ``sample_driver_gpu``; unit tests inject
local fakes.  The campaign owns an additional, immutable HBM cleanup threshold so
that a sequence of individually clean children cannot ratchet its baseline upward.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

INDEPENDENT_MODE = "independent"
SHARED_ROOT_MODE = "shared_root"
PAIR_MODES: tuple[str, str] = (INDEPENDENT_MODE, SHARED_ROOT_MODE)
TRACE_LEVELS = frozenset({"disabled", "minimal", "full"})
PRIMARY_SEEDS = (41, 73, 113)
BASELINE_SAMPLE_COUNT = 3
STABLE_RECOVERY_SAMPLE_COUNT = 3
HBM_RECOVERY_ALLOWANCE_MIB = 1024
ORDER_ALGORITHM = "sha256-lsb:sloforge.branchfabric.experiment-003-order/v1"

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

GpuSampler = Callable[[str | None], dict[str, Any]]
ControllerRunner = Callable[..., dict[str, Any]]
Sleeper = Callable[[float], None]


class CampaignError(RuntimeError):
    """A fail-closed campaign validation or lifecycle error."""

    def __init__(self, message: str, *, stage: str) -> None:
        super().__init__(message)
        self.stage = stage


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_clone(value: Any) -> Any:
    """Validate JSON serializability and detach injected mutable values."""

    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _require_identifier(value: str, *, field: str) -> None:
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a path-safe identifier")


@dataclass(frozen=True)
class CampaignConfig:
    """Configuration for the three-pair fanout-8 causal campaign."""

    campaign_id: str
    randomized_order_seed: int
    seeds: tuple[int, int, int] = PRIMARY_SEEDS
    fanout: int = 8
    prefix_length: int = 16_384
    suffix_length: int = 256
    tracing_level: str = "minimal"
    maximum_campaign_seconds: float = 2700.0
    baseline_sample_interval_seconds: float = 0.2
    cleanup_sample_interval_seconds: float = 1.0
    cleanup_timeout_seconds: float = 30.0
    maximum_cleanup_samples: int = 33
    controller_monitor_interval_seconds: float = 1.0
    hbm_recovery_allowance_mib: int = HBM_RECOVERY_ALLOWANCE_MIB

    def __post_init__(self) -> None:
        _require_identifier(self.campaign_id, field="campaign_id")
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("primary paired campaign requires exactly three unique seeds")
        if any(isinstance(seed, bool) or seed < 0 or seed >= 1 << 63 for seed in self.seeds):
            raise ValueError("seeds must be unsigned 63-bit integers")
        if (
            isinstance(self.randomized_order_seed, bool)
            or self.randomized_order_seed < 0
            or self.randomized_order_seed >= 1 << 63
        ):
            raise ValueError("randomized_order_seed must be an unsigned 63-bit integer")
        for int_field, int_value in (
            ("fanout", self.fanout),
            ("prefix_length", self.prefix_length),
            ("suffix_length", self.suffix_length),
            ("maximum_cleanup_samples", self.maximum_cleanup_samples),
            ("hbm_recovery_allowance_mib", self.hbm_recovery_allowance_mib),
        ):
            if isinstance(int_value, bool) or int_value <= 0:
                raise ValueError(f"{int_field} must be a positive integer")
        if self.maximum_cleanup_samples < STABLE_RECOVERY_SAMPLE_COUNT:
            raise ValueError("maximum_cleanup_samples cannot be smaller than the stable gate")
        for float_field, float_value in (
            ("maximum_campaign_seconds", self.maximum_campaign_seconds),
            ("baseline_sample_interval_seconds", self.baseline_sample_interval_seconds),
            ("cleanup_sample_interval_seconds", self.cleanup_sample_interval_seconds),
            ("cleanup_timeout_seconds", self.cleanup_timeout_seconds),
            ("controller_monitor_interval_seconds", self.controller_monitor_interval_seconds),
        ):
            if isinstance(float_value, bool) or float_value < 0:
                raise ValueError(f"{float_field} must be nonnegative")
        if self.maximum_campaign_seconds <= 0 or self.cleanup_timeout_seconds <= 0:
            raise ValueError("campaign and cleanup timeouts must be positive")
        if self.tracing_level not in TRACE_LEVELS:
            raise ValueError(f"tracing_level must be one of {sorted(TRACE_LEVELS)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sloforge.branchfabric.gpu-validation-campaign-config/v1",
            "campaign_id": self.campaign_id,
            "randomized_order_seed": self.randomized_order_seed,
            "seeds": list(self.seeds),
            "fanout": self.fanout,
            "prefix_length": self.prefix_length,
            "suffix_length": self.suffix_length,
            "tracing_level": self.tracing_level,
            "maximum_campaign_seconds": self.maximum_campaign_seconds,
            "baseline_sample_interval_seconds": self.baseline_sample_interval_seconds,
            "cleanup_sample_interval_seconds": self.cleanup_sample_interval_seconds,
            "cleanup_timeout_seconds": self.cleanup_timeout_seconds,
            "maximum_cleanup_samples": self.maximum_cleanup_samples,
            "controller_monitor_interval_seconds": self.controller_monitor_interval_seconds,
            "hbm_recovery_allowance_mib": self.hbm_recovery_allowance_mib,
        }


@dataclass(frozen=True)
class PairPlan:
    pair_id: str
    seed: int
    order_rule: str
    order_seed: int | None
    realized_order: tuple[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "seed": self.seed,
            "order_rule": self.order_rule,
            "order_seed": self.order_seed,
            "order_algorithm": ORDER_ALGORITHM if self.order_seed is not None else None,
            "realized_order": list(self.realized_order),
        }


def _randomized_pair_order(*, seed: int, order_seed: int) -> tuple[str, str]:
    material = (f"sloforge.branchfabric.experiment-003-order/v1:{order_seed}:{seed}").encode(
        "ascii"
    )
    bit = hashlib.sha256(material).digest()[0] & 1
    return PAIR_MODES if bit == 0 else (PAIR_MODES[1], PAIR_MODES[0])


def plan_primary_pairs(config: CampaignConfig) -> tuple[PairPlan, PairPlan, PairPlan]:
    """Resolve the required fixed, reversed, and seeded randomized pair orders."""

    first, second, third = config.seeds
    return (
        PairPlan(
            pair_id=f"seed-{first}",
            seed=first,
            order_rule="independent_then_shared",
            order_seed=None,
            realized_order=PAIR_MODES,
        ),
        PairPlan(
            pair_id=f"seed-{second}",
            seed=second,
            order_rule="shared_then_independent",
            order_seed=None,
            realized_order=(PAIR_MODES[1], PAIR_MODES[0]),
        ),
        PairPlan(
            pair_id=f"seed-{third}",
            seed=third,
            order_rule="deterministic_randomized",
            order_seed=config.randomized_order_seed,
            realized_order=_randomized_pair_order(
                seed=third,
                order_seed=config.randomized_order_seed,
            ),
        ),
    )


def _gpu_identity(sample: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    if sample.get("error") is not None:
        raise CampaignError(f"GPU sampling failed: {sample['error']}", stage=stage)
    gpu = sample.get("gpu")
    if not isinstance(gpu, Mapping):
        raise CampaignError("GPU sample did not contain a GPU record", stage=stage)
    try:
        identity: dict[str, Any] = {
            "index": int(gpu["index"]),
            "uuid": str(gpu["uuid"]),
            "name": str(gpu["name"]),
            "driver_version": str(gpu["driver_version"]),
            "memory_total_mib": int(gpu["memory_total_mib"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("GPU sample identity was incomplete", stage=stage) from error
    if "A100" not in str(identity["name"]) or int(identity["memory_total_mib"]) < 80_000:
        raise CampaignError(f"unsupported campaign GPU: {identity}", stage=stage)
    return identity


def _validated_sample(
    sampler: GpuSampler,
    *,
    expected_identity: Mapping[str, Any] | None,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_uuid = None if expected_identity is None else str(expected_identity["uuid"])
    try:
        raw_sample = sampler(expected_uuid)
    except Exception as error:
        raise CampaignError(
            f"GPU sampler raised {type(error).__name__}: {error}", stage=stage
        ) from error
    sample = _json_clone(raw_sample)
    identity = _gpu_identity(sample, stage=stage)
    if expected_identity is not None and identity != dict(expected_identity):
        raise CampaignError(
            f"campaign GPU identity changed from {dict(expected_identity)!r} to {identity!r}",
            stage=stage,
        )
    processes = sample.get("compute_processes")
    if not isinstance(processes, list):
        raise CampaignError("GPU sample compute-process inventory was unavailable", stage=stage)
    return sample, identity


def _initial_campaign_gate(
    *,
    sampler: GpuSampler,
    interval_seconds: float,
    allowance_mib: int,
    sleep: Sleeper,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    identity: dict[str, Any] | None = None
    for index in range(BASELINE_SAMPLE_COUNT):
        sample, observed_identity = _validated_sample(
            sampler,
            expected_identity=identity,
            stage="campaign_baseline",
        )
        identity = observed_identity
        if sample["compute_processes"]:
            raise CampaignError(
                f"campaign GPU was not exclusive: {sample['compute_processes']}",
                stage="campaign_baseline",
            )
        sample["campaign_baseline_sample_index"] = index
        samples.append(sample)
        if index + 1 < BASELINE_SAMPLE_COUNT:
            sleep(interval_seconds)
    assert identity is not None
    median_mib = int(statistics.median(int(sample["gpu"]["memory_used_mib"]) for sample in samples))
    return {
        "schema_version": "sloforge.branchfabric.campaign-hbm-baseline/v1",
        "passed": True,
        "gpu_identity": identity,
        "samples": samples,
        "baseline_median_mib": median_mib,
        "cleanup_threshold_mib": median_mib + allowance_mib,
        "allowance_mib": allowance_mib,
    }


def _pre_child_gate(
    *,
    sampler: GpuSampler,
    expected_identity: Mapping[str, Any],
    threshold_mib: int,
    interval_seconds: float,
    sleep: Sleeper,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for index in range(BASELINE_SAMPLE_COUNT):
        sample, _ = _validated_sample(
            sampler,
            expected_identity=expected_identity,
            stage="pre_child_gate",
        )
        memory_mib = int(sample["gpu"]["memory_used_mib"])
        if sample["compute_processes"] or memory_mib > threshold_mib:
            raise CampaignError(
                "pre-child GPU was not clean against the immutable campaign threshold",
                stage="pre_child_gate",
            )
        sample["pre_child_sample_index"] = index
        samples.append(sample)
        if index + 1 < BASELINE_SAMPLE_COUNT:
            sleep(interval_seconds)
    return {
        "schema_version": "sloforge.branchfabric.campaign-pre-child-gate/v1",
        "passed": True,
        "samples": samples,
        "required_samples": BASELINE_SAMPLE_COUNT,
        "cleanup_threshold_mib": threshold_mib,
    }


def _post_child_gate(
    *,
    sampler: GpuSampler,
    expected_identity: Mapping[str, Any],
    threshold_mib: int,
    interval_seconds: float,
    timeout_seconds: float,
    maximum_samples: int,
    sleep: Sleeper,
) -> dict[str, Any]:
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    consecutive = 0
    winning_start: int | None = None
    for index in range(maximum_samples):
        sample, _ = _validated_sample(
            sampler,
            expected_identity=expected_identity,
            stage="campaign_postflight_gate",
        )
        memory_mib = int(sample["gpu"]["memory_used_mib"])
        qualifies = memory_mib <= threshold_mib and sample["compute_processes"] == []
        sample["post_child_sample_index"] = index
        sample["qualifies"] = qualifies
        samples.append(sample)
        if qualifies:
            consecutive += 1
            if consecutive == 1:
                winning_start = index
            if consecutive >= STABLE_RECOVERY_SAMPLE_COUNT:
                return {
                    "schema_version": "sloforge.branchfabric.campaign-post-child-gate/v1",
                    "passed": True,
                    "samples": samples,
                    "required_consecutive_samples": STABLE_RECOVERY_SAMPLE_COUNT,
                    "winning_sample_start_index": winning_start,
                    "winning_sample_end_index": index,
                    "cleanup_threshold_mib": threshold_mib,
                }
        else:
            consecutive = 0
            winning_start = None
        if time.monotonic() - started >= timeout_seconds:
            break
        if index + 1 < maximum_samples:
            sleep(interval_seconds)
    return {
        "schema_version": "sloforge.branchfabric.campaign-post-child-gate/v1",
        "passed": False,
        "samples": samples,
        "required_consecutive_samples": STABLE_RECOVERY_SAMPLE_COUNT,
        "winning_sample_start_index": None,
        "winning_sample_end_index": None,
        "cleanup_threshold_mib": threshold_mib,
        "failure_reason": ("post-child HBM cleanup did not reach three consecutive clean samples"),
    }


def _assert_controller_result(
    result: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
) -> None:
    if result.get("status") != "succeeded":
        raise CampaignError("single-child controller failed", stage="child_controller")
    cleanup = result.get("cleanup_gate")
    if not isinstance(cleanup, Mapping) or cleanup.get("GPU_RUNTIME_LIFECYCLE_CLEAN") is not True:
        raise CampaignError("single-child cleanup gate did not pass", stage="child_controller")
    if cleanup.get("forced_gpu_process_kill_required") is True:
        raise CampaignError(
            "single-child controller required a forced kill", stage="child_controller"
        )
    actual_gpu = result.get("actual_gpu")
    if not isinstance(actual_gpu, Mapping):
        raise CampaignError(
            "single-child controller omitted GPU identity", stage="child_controller"
        )
    observed_identity = _gpu_identity({"gpu": actual_gpu}, stage="child_controller")
    if observed_identity != dict(expected_identity):
        raise CampaignError(
            "single-child controller reported a different GPU identity",
            stage="child_controller",
        )
    child_manifest = result.get("child_manifest")
    if not isinstance(child_manifest, Mapping):
        raise CampaignError(
            "single-child controller omitted its child manifest", stage="child_controller"
        )
    child_uuid = child_manifest.get("gpu_uuid")
    if child_uuid is not None and child_uuid != expected_identity["uuid"]:
        raise CampaignError(
            "child manifest GPU UUID disagreed with campaign", stage="child_controller"
        )


def _trial_payload(
    base_payload: Mapping[str, Any],
    *,
    config: CampaignConfig,
    pair: PairPlan,
    mode: str,
    order_position: int,
) -> dict[str, Any]:
    payload = cast(dict[str, Any], _json_clone(dict(base_payload)))
    attempt_id = f"{config.campaign_id}-{pair.pair_id}-{order_position:02d}-{mode}"
    payload.update(
        {
            "attempt_id": attempt_id,
            "campaign_id": config.campaign_id,
            "pair_id": pair.pair_id,
            "pair_order_position": order_position,
            "pair_order_rule": pair.order_rule,
            "pair_order_seed": pair.order_seed,
            "baseline_mode": mode,
            "fanout": config.fanout,
            "prefix_length": config.prefix_length,
            "suffix_length": config.suffix_length,
            "seed": pair.seed,
            "tracing_level": config.tracing_level,
        }
    )
    return payload


def run_campaign(
    *,
    config: CampaignConfig,
    base_config_payload: Mapping[str, Any],
    work_root: Path,
    worker_path: Path,
    model_snapshot: Path,
    controller_runner: ControllerRunner,
    gpu_sampler: GpuSampler,
    sleep: Sleeper = time.sleep,
) -> dict[str, Any]:
    """Run all paired children serially and return an immutable JSON manifest.

    ``controller_runner`` is expected to be
    :func:`gpu_cow_controller.run_controller`.  It is injected so this module
    stays CUDA-clean and its abort semantics can be tested without a GPU.
    """

    plan = plan_primary_pairs(config)
    base_payload = _json_clone(dict(base_config_payload))
    started_at_utc = _utc_now()
    started_ns = time.monotonic_ns()
    work_root.mkdir(parents=True, exist_ok=False)
    pairs_root = work_root / "pairs"
    pairs_root.mkdir()
    plan_payload = {
        "schema_version": "sloforge.branchfabric.gpu-validation-campaign-plan/v1",
        "config": config.to_dict(),
        "base_config_payload": base_payload,
        "pairs": [item.to_dict() for item in plan],
        "total_planned_children": sum(len(item.realized_order) for item in plan),
    }
    _write_json(work_root / "campaign-plan.json", plan_payload)

    campaign_baseline: dict[str, Any] | None = None
    pair_manifests: list[dict[str, Any]] = []
    error_record: dict[str, Any] | None = None
    started_children = 0
    completed_children = 0
    try:
        campaign_baseline = _initial_campaign_gate(
            sampler=gpu_sampler,
            interval_seconds=config.baseline_sample_interval_seconds,
            allowance_mib=config.hbm_recovery_allowance_mib,
            sleep=sleep,
        )
        expected_identity = campaign_baseline["gpu_identity"]
        threshold_mib = int(campaign_baseline["cleanup_threshold_mib"])
        for pair in plan:
            if (time.monotonic_ns() - started_ns) / 1e9 >= config.maximum_campaign_seconds:
                raise CampaignError("campaign wall-time bound reached", stage="campaign_deadline")
            pair_root = pairs_root / pair.pair_id
            pair_root.mkdir()
            pair_manifest: dict[str, Any] = {
                "schema_version": "sloforge.branchfabric.same-gpu-pair-manifest/v1",
                **pair.to_dict(),
                "campaign_id": config.campaign_id,
                "campaign_gpu_identity": expected_identity,
                "authoritative_cleanup_threshold_mib": threshold_mib,
                "tracing_level": config.tracing_level,
                "status": "running",
                "same_gpu_identity_verified": False,
                "trials": [],
                "error": None,
            }
            try:
                for order_position, mode in enumerate(pair.realized_order, start=1):
                    if (time.monotonic_ns() - started_ns) / 1e9 >= config.maximum_campaign_seconds:
                        raise CampaignError(
                            "campaign wall-time bound reached",
                            stage="campaign_deadline",
                        )
                    payload = _trial_payload(
                        base_payload,
                        config=config,
                        pair=pair,
                        mode=mode,
                        order_position=order_position,
                    )
                    trial_record: dict[str, Any] = {
                        "attempt_id": payload["attempt_id"],
                        "pair_id": pair.pair_id,
                        "seed": pair.seed,
                        "baseline_mode": mode,
                        "order_position": order_position,
                        "work_root": str(
                            Path("pairs") / pair.pair_id / f"{order_position:02d}-{mode}"
                        ),
                        "status": "running",
                        "pre_child_gate": None,
                        "controller": None,
                        "campaign_postflight_gate": None,
                        "error": None,
                    }
                    try:
                        trial_record["pre_child_gate"] = _pre_child_gate(
                            sampler=gpu_sampler,
                            expected_identity=expected_identity,
                            threshold_mib=threshold_mib,
                            interval_seconds=config.baseline_sample_interval_seconds,
                            sleep=sleep,
                        )
                        trial_root = pair_root / f"{order_position:02d}-{mode}"
                        started_children += 1
                        result = controller_runner(
                            config_payload=payload,
                            work_root=trial_root,
                            worker_path=worker_path,
                            model_snapshot=model_snapshot,
                            gpu_sampler=gpu_sampler,
                            postflight_timeout_s=config.cleanup_timeout_seconds,
                            baseline_interval_s=config.baseline_sample_interval_seconds,
                            monitor_interval_s=config.controller_monitor_interval_seconds,
                        )
                        _assert_controller_result(result, expected_identity=expected_identity)
                        trial_record["controller"] = _json_clone(
                            {
                                "status": result.get("status"),
                                "actual_gpu": result.get("actual_gpu"),
                                "summary": result.get("summary"),
                                "child_manifest": result.get("child_manifest"),
                                "cleanup_gate": result.get("cleanup_gate"),
                                "controller_manifest": result.get("controller_manifest"),
                                "error": result.get("error"),
                            }
                        )
                        trial_record["campaign_postflight_gate"] = _post_child_gate(
                            sampler=gpu_sampler,
                            expected_identity=expected_identity,
                            threshold_mib=threshold_mib,
                            interval_seconds=config.cleanup_sample_interval_seconds,
                            timeout_seconds=config.cleanup_timeout_seconds,
                            maximum_samples=config.maximum_cleanup_samples,
                            sleep=sleep,
                        )
                        if not trial_record["campaign_postflight_gate"]["passed"]:
                            raise CampaignError(
                                "post-child HBM cleanup did not reach three consecutive clean "
                                f"samples at or below immutable threshold {threshold_mib} MiB",
                                stage="campaign_postflight_gate",
                            )
                        trial_record["status"] = "succeeded"
                        completed_children += 1
                    except Exception as error:
                        trial_record["status"] = "failed"
                        trial_record["error"] = {
                            "type": type(error).__name__,
                            "message": str(error),
                            "stage": getattr(error, "stage", "trial"),
                        }
                        pair_manifest["trials"].append(trial_record)
                        raise
                    pair_manifest["trials"].append(trial_record)
                pair_manifest["status"] = "succeeded"
                pair_manifest["same_gpu_identity_verified"] = True
            except Exception as error:
                pair_manifest["status"] = "failed"
                pair_manifest["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "stage": getattr(error, "stage", "pair"),
                }
                _write_json(pair_root / "pair-manifest.json", pair_manifest)
                pair_manifests.append(pair_manifest)
                raise
            _write_json(pair_root / "pair-manifest.json", pair_manifest)
            pair_manifests.append(pair_manifest)
    except Exception as error:
        error_record = {
            "type": type(error).__name__,
            "message": str(error),
            "stage": getattr(error, "stage", "campaign"),
        }

    ended_ns = time.monotonic_ns()
    manifest = {
        "schema_version": "sloforge.branchfabric.gpu-validation-campaign-manifest/v1",
        "campaign_id": config.campaign_id,
        "status": "succeeded" if error_record is None else "failed",
        "started_at_utc": started_at_utc,
        "ended_at_utc": _utc_now(),
        "duration_seconds": (ended_ns - started_ns) / 1e9,
        "config": config.to_dict(),
        "plan_path": "campaign-plan.json",
        "campaign_hbm_baseline": campaign_baseline,
        "gpu_identity": None if campaign_baseline is None else campaign_baseline["gpu_identity"],
        "authoritative_cleanup_threshold_mib": (
            None if campaign_baseline is None else campaign_baseline["cleanup_threshold_mib"]
        ),
        "planned_children": sum(len(item.realized_order) for item in plan),
        "started_children": started_children,
        "completed_children": completed_children,
        "all_children_fresh_and_sequential": error_record is None,
        "pair_manifests": pair_manifests,
        "error": error_record,
    }
    _write_json(work_root / "campaign-manifest.json", manifest)
    return manifest


__all__ = [
    "BASELINE_SAMPLE_COUNT",
    "HBM_RECOVERY_ALLOWANCE_MIB",
    "INDEPENDENT_MODE",
    "ORDER_ALGORITHM",
    "PAIR_MODES",
    "PRIMARY_SEEDS",
    "SHARED_ROOT_MODE",
    "STABLE_RECOVERY_SAMPLE_COUNT",
    "TRACE_LEVELS",
    "CampaignConfig",
    "CampaignError",
    "PairPlan",
    "plan_primary_pairs",
    "run_campaign",
]
