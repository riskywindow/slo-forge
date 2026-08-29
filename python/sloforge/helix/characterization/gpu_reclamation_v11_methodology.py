"""Fail-closed methodology contracts for Experiment 004 v11 paid runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
V11_CHUNK_BYTES = 28 * 1024 * 1024
V11_MICRO_RESERVATION_WALL_SECONDS = 525.0
V11_MICRO_POST_CONTROLLER_RESERVE_SECONDS = 10.0
V11_INTEGRATED_RESERVATION_WALL_SECONDS = 588.0
V11_TARGETED_IDENTITY_RESERVATION_WALL_SECONDS = 300.0
V11_KILL_RECOMPUTE_RESERVATION_WALL_SECONDS = 660.0

_Attempt = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{7,95}$")]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_GitObject = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class Experiment004V11MicroConfig(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-v11-micro-config/v1"] = (
        "sloforge.branchfabric.experiment-004-v11-micro-config/v1"
    )
    pipeline_version: Literal["v11"] = "v11"
    mode: Literal["PRESERVE_OPTIMIZED"] = "PRESERVE_OPTIMIZED"
    execution_mode: Literal["state-pipeline-micro-v11"] = "state-pipeline-micro-v11"
    attempt_id: _Attempt
    seed: int = Field(ge=0, lt=1 << 63)
    model: Literal["Qwen/Qwen2.5-7B-Instruct"] = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    )
    tokenizer_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    )
    runtime: Literal["vllm"] = "vllm"
    runtime_version: Literal["0.23.0"] = "0.23.0"
    torch_version: Literal["2.11.0"] = "2.11.0"
    dtype: Literal["bfloat16"] = "bfloat16"
    requested_gpu: Literal["A100-80GB"] = "A100-80GB"
    gpu_count: Literal[1] = 1
    prefix_length: Literal[16384] = 16384
    fanout: Literal[8] = 8
    suffix_length: Literal[256] = 256
    continuation_tokens: Literal[8] = 8
    expected_block_size_tokens: Literal[16] = 16
    expected_shared_blocks: Literal[1024] = 1024
    expected_private_blocks: Literal[128] = 128
    expected_total_blocks: Literal[1152] = 1152
    maximum_chunk_bytes: Literal[29360128] = 29360128
    buffer_count: Literal[2] = 2
    tracing_level: Literal["minimal"] = "minimal"
    gpu_memory_utilization: float = Field(default=0.72, gt=0.0, le=0.85)
    initialization_timeout_seconds: int = Field(default=180, ge=120, le=240)
    transaction_timeout_seconds: int = Field(default=320, ge=180, le=360)
    cleanup_timeout_seconds: int = Field(default=20, ge=10, le=30)
    maximum_wall_seconds: float = Field(default=525.0, ge=525.0, le=525.0)
    offline_gate_manifest: str = Field(min_length=1)
    offline_gate_manifest_sha256: _Sha256
    budget_authorization: str = Field(min_length=1)
    budget_authorization_sha256: _Sha256
    ledger_sha256_before_reservation: _Sha256

    @model_validator(mode="after")
    def exact_state_and_bounds(self) -> Self:
        if self.expected_shared_blocks + self.expected_private_blocks != self.expected_total_blocks:
            raise ValueError("v11 exact block topology does not close")
        if self.prefix_length // self.expected_block_size_tokens != self.expected_shared_blocks:
            raise ValueError("v11 prefix length does not produce 1,024 shared blocks")
        if (
            self.fanout * self.suffix_length // self.expected_block_size_tokens
            != self.expected_private_blocks
        ):
            raise ValueError("v11 fanout/suffix does not produce 128 private blocks")
        bounded = (
            self.initialization_timeout_seconds
            + self.transaction_timeout_seconds
            + self.cleanup_timeout_seconds
        )
        if bounded > self.maximum_wall_seconds:
            raise ValueError("v11 micro inner timeouts exceed the paid wall bound")
        return self


class Experiment004V11IntegratedConfig(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-v11-integrated-config/v1"] = (
        "sloforge.branchfabric.experiment-004-v11-integrated-config/v1"
    )
    pipeline_version: Literal["v11"] = "v11"
    mode: Literal["PRESERVE_OPTIMIZED"] = "PRESERVE_OPTIMIZED"
    execution_mode: Literal["integrated-reclamation-v11"] = "integrated-reclamation-v11"
    attempt_id: _Attempt
    seed: int = Field(ge=0, lt=1 << 63)
    model: Literal["Qwen/Qwen2.5-7B-Instruct"] = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    )
    tokenizer_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    )
    runtime: Literal["vllm"] = "vllm"
    runtime_version: Literal["0.23.0"] = "0.23.0"
    torch_version: Literal["2.11.0"] = "2.11.0"
    dtype: Literal["bfloat16"] = "bfloat16"
    requested_gpu: Literal["A100-80GB"] = "A100-80GB"
    gpu_count: Literal[2] = 2
    prefix_length: Literal[16384] = 16384
    fanout: Literal[8] = 8
    suffix_length: Literal[256] = 256
    continuation_tokens: Literal[8] = 8
    expected_block_size_tokens: Literal[16] = 16
    expected_shared_blocks: Literal[1024] = 1024
    expected_private_blocks: Literal[128] = 128
    expected_total_blocks: Literal[1152] = 1152
    maximum_chunk_bytes: Literal[29360128] = 29360128
    buffer_count: Literal[2] = 2
    tracing_level: Literal["minimal"] = "minimal"
    lambda_1_rps: float = Field(default=12.0, ge=12.0, le=12.0)
    lambda_spike_rps: float = Field(default=15.0, ge=15.0, le=15.0)
    lambda_2_rps: float = Field(default=20.0, ge=20.0, le=20.0)
    control_request_rate_rps: float = Field(default=9.0, ge=9.0, le=9.0)
    restore_request_rate_rps: float = Field(default=9.0, ge=9.0, le=9.0)
    serving_methodology: Literal["v10-global-capacity"] = "v10-global-capacity"
    baseline_seconds: float = Field(default=5.0, ge=5.0, le=5.0)
    sanity_guard_measurement_seconds: float = Field(default=3.0, ge=3.0, le=3.0)
    overload_probe_seconds: float = Field(default=3.0, ge=3.0, le=3.0)
    recovery_evaluation_seconds: float = Field(default=1.0, ge=1.0, le=1.0)
    recovery_queue_threshold: Literal[4] = 4
    temporary_serving_seconds: float = Field(default=5.0, ge=5.0, le=5.0)
    producer_queue_capacity: Literal[256] = 256
    maximum_pending_requests: Literal[64] = 64
    restore_handoff_lead_requests: Literal[4] = 4
    probe_timeout_seconds: float = Field(default=15.0, ge=15.0, le=15.0)
    tail_drain_seconds: float = Field(default=1.0, ge=1.0, le=1.0)
    warmup_seconds: float = Field(default=1.0, ge=1.0, le=1.0)
    overload_queue_trigger: Literal[20] = 20
    overload_queue_abort: Literal[64] = 64
    serving_prompt_tokens: Literal[256] = 256
    serving_output_tokens: Literal[64] = 64
    serving_slo_ttft_seconds: float = Field(default=2.0, ge=2.0, le=2.0)
    serving_stability_seconds: float = Field(default=5.0, ge=5.0, le=5.0)
    gpu_memory_utilization: float = Field(default=0.72, gt=0.0, le=0.85)
    initialization_timeout_seconds: Literal[160] = 160
    cleanup_timeout_seconds: Literal[10] = 10
    maximum_wall_seconds: float = Field(default=588.0, ge=588.0, le=588.0)
    offline_gate_manifest: str = Field(min_length=1)
    offline_gate_manifest_sha256: _Sha256
    micro_validation_artifact: str = Field(min_length=1)
    micro_validation_sha256: _Sha256
    post_micro_review_manifest: str = Field(min_length=1)
    post_micro_review_manifest_sha256: _Sha256
    budget_authorization: str = Field(min_length=1)
    budget_authorization_sha256: _Sha256
    ledger_sha256_before_reservation: _Sha256

    @model_validator(mode="after")
    def exact_calibration(self) -> Self:
        if self.expected_shared_blocks + self.expected_private_blocks != self.expected_total_blocks:
            raise ValueError("v11 integrated exact block topology does not close")
        if self.prefix_length // self.expected_block_size_tokens != self.expected_shared_blocks:
            raise ValueError("v11 integrated prefix does not produce 1,024 shared blocks")
        if (
            self.fanout * self.suffix_length // self.expected_block_size_tokens
            != self.expected_private_blocks
        ):
            raise ValueError("v11 integrated suffixes do not produce 128 private blocks")
        if not self.control_request_rate_rps < self.lambda_1_rps < self.lambda_spike_rps:
            raise ValueError("v11 control/spike rates disagree with frozen calibration")
        if not self.lambda_spike_rps < self.lambda_2_rps:
            raise ValueError("v11 spike is not below measured two-GPU capacity")
        if self.restore_request_rate_rps >= self.lambda_1_rps:
            raise ValueError("v11 restore interference load must remain below lambda_1")
        return self


class Experiment004V11TargetedSourceIdentityConfig(Experiment004V11IntegratedConfig):
    """Exact integrated lifecycle truncated at the source identity gate."""

    schema_version: Literal[
        "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-config/v1"
    ] = "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-config/v1"  # type: ignore[assignment]
    execution_mode: Literal["targeted-source-identity-v11"] = "targeted-source-identity-v11"  # type: ignore[assignment]
    terminal_phase: Literal["SOURCE_ALLOCATION_IDENTITY_GATE"] = "SOURCE_ALLOCATION_IDENTITY_GATE"
    full_export_authorized: Literal[False] = False
    source_release_authorized: Literal[False] = False
    maximum_wall_seconds: float = Field(default=300.0, ge=300.0, le=300.0)


class Experiment004V11KillRecomputeConfig(Experiment004V11IntegratedConfig):
    """One post-freeze real-runtime kill/recompute comparison arm.

    It inherits every model, runtime, topology, and serving-methodology field
    from the frozen integrated contract.  The additional bindings make the
    successful K bundle and its freeze the sole authorization boundary.
    """

    schema_version: Literal["sloforge.branchfabric.experiment-004-kill-recompute-config/v1"] = (
        "sloforge.branchfabric.experiment-004-kill-recompute-config/v1"  # type: ignore[assignment]
    )
    mode: Literal["KILL_AND_RECOMPUTE"] = "KILL_AND_RECOMPUTE"  # type: ignore[assignment]
    execution_mode: Literal["integrated-kill-recompute-v11"] = "integrated-kill-recompute-v11"  # type: ignore[assignment]
    maximum_wall_seconds: float = Field(default=660.0, ge=660.0, le=660.0)
    integrated_k_status_artifact: str = Field(min_length=1)
    integrated_k_status_sha256: _Sha256
    integrated_k_remote_manifest: str = Field(min_length=1)
    integrated_k_remote_manifest_sha256: _Sha256
    integrated_k_remote_manifest_entries: Literal[507] = 507
    integrated_k_provider_cleanup_artifact: str = Field(min_length=1)
    integrated_k_provider_cleanup_sha256: _Sha256
    optimized_v11_freeze_artifact: str = Field(min_length=1)
    optimized_v11_freeze_sha256: _Sha256
    optimized_v11_freeze_tag_binding_artifact: str = Field(min_length=1)
    optimized_v11_freeze_tag_binding_sha256: _Sha256
    optimized_v11_freeze_tag: Literal["branchfabric-exp004-optimized-preserve-v11"] = (
        "branchfabric-exp004-optimized-preserve-v11"
    )
    optimized_v11_freeze_tag_object: _GitObject
    ledger_snapshot_artifact: str = Field(min_length=1)
    ledger_snapshot_sha256: _Sha256
    expected_submitted_prefix_tokens: Literal[131072] = 131072
    expected_computed_private_tokens: Literal[2048] = 2048
    expected_computed_boundary_tokens: Literal[133120] = 133120
    expected_computed_state_topology_tokens: Literal[18432] = 18432
    expected_live_uncomputed_tail_tokens: Literal[8] = 8
    expected_submitted_private_tokens: Literal[2056] = 2056
    expected_submitted_replay_tokens: Literal[133128] = 133128
    expected_unique_submitted_replay_history_tokens: Literal[18440] = 18440
    expected_lost_private_rollout_tokens: Literal[2048] = 2048

    @model_validator(mode="after")
    def exact_kill_recompute_workload(self) -> Self:
        submitted_prefix = self.fanout * self.prefix_length
        computed_private = self.fanout * self.suffix_length
        submitted_private = computed_private + self.expected_live_uncomputed_tail_tokens
        if submitted_prefix != self.expected_submitted_prefix_tokens:
            raise ValueError("kill/recompute submitted prefix count changed")
        if computed_private != self.expected_computed_private_tokens:
            raise ValueError("kill/recompute computed private count changed")
        if submitted_prefix + computed_private != self.expected_computed_boundary_tokens:
            raise ValueError("kill/recompute computed KV boundary changed")
        if submitted_private != self.expected_submitted_private_tokens:
            raise ValueError("kill/recompute submitted private count changed")
        if submitted_prefix + submitted_private != self.expected_submitted_replay_tokens:
            raise ValueError("kill/recompute submitted replay count changed")
        if self.prefix_length + computed_private != self.expected_computed_state_topology_tokens:
            raise ValueError("kill/recompute computed state topology changed")
        if (
            self.prefix_length + submitted_private
            != self.expected_unique_submitted_replay_history_tokens
        ):
            raise ValueError("kill/recompute submitted replay topology changed")
        if computed_private != self.expected_lost_private_rollout_tokens:
            raise ValueError("kill/recompute lost private rollout work changed")
        return self


def canonical_json_bytes(value: BaseModel | dict[str, Any]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bound_artifact(
    repository_root: Path,
    *,
    reference: str,
    expected_sha256: str,
) -> Path:
    path = (repository_root / reference).resolve(strict=True)
    root = repository_root.resolve(strict=True)
    if path != root and root not in path.parents:
        raise ValueError("v11 bound artifact escapes the repository")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"v11 bound artifact hash mismatch: {reference}")
    return path


def projected_gpu_seconds(*, gpu_count: int, predicted_wall_seconds: float, margin: float) -> float:
    if gpu_count not in {1, 2}:
        raise ValueError("v11 GPU count must be one or two")
    if not math.isfinite(predicted_wall_seconds) or predicted_wall_seconds <= 0:
        raise ValueError("v11 predicted wall time must be finite and positive")
    if not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("v11 GPU budget margin must be in [0, 1]")
    return gpu_count * predicted_wall_seconds * (1.0 + margin)


__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "V11_CHUNK_BYTES",
    "V11_INTEGRATED_RESERVATION_WALL_SECONDS",
    "V11_KILL_RECOMPUTE_RESERVATION_WALL_SECONDS",
    "V11_MICRO_RESERVATION_WALL_SECONDS",
    "V11_TARGETED_IDENTITY_RESERVATION_WALL_SECONDS",
    "Experiment004V11IntegratedConfig",
    "Experiment004V11KillRecomputeConfig",
    "Experiment004V11MicroConfig",
    "Experiment004V11TargetedSourceIdentityConfig",
    "canonical_json_bytes",
    "projected_gpu_seconds",
    "sha256_file",
    "validate_bound_artifact",
]
