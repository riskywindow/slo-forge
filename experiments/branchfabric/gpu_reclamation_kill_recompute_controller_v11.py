"""Fail-closed controller for the single Experiment 004 kill/recompute arm.

The mature K controller retains ownership of readiness, A/B replication, the
15-rps overload guard, two-process lifecycle cleanup, and GPU inventory.  This
module applies a process-local typed kill envelope, K-freeze verifier, 660s
deadline, and kill-specific result validator, restoring every patched binding
before return.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    V11_KILL_RECOMPUTE_RESERVATION_WALL_SECONDS,
    Experiment004V11KillRecomputeConfig,
    validate_bound_artifact,
)

_INTEGRATED_K_SETTLED_LEDGER_SHA256 = (
    "e9b3b521da9f0a6eb3ac2e69030b7ce5560e4f5de11504644cf8f707237d8a16"
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"kill/recompute {label} is not a JSON object")
    return value


def verify_kill_recompute_evidence(
    config: Experiment004V11KillRecomputeConfig,
    repository_root: Path,
) -> dict[str, Any]:
    """Verify K validity, 507-entry bundle commitment, cleanup, freeze, and ledger."""

    bindings = {
        "integrated_k_status": (
            config.integrated_k_status_artifact,
            config.integrated_k_status_sha256,
        ),
        "integrated_k_remote_manifest": (
            config.integrated_k_remote_manifest,
            config.integrated_k_remote_manifest_sha256,
        ),
        "integrated_k_provider_cleanup": (
            config.integrated_k_provider_cleanup_artifact,
            config.integrated_k_provider_cleanup_sha256,
        ),
        "optimized_v11_freeze": (
            config.optimized_v11_freeze_artifact,
            config.optimized_v11_freeze_sha256,
        ),
        "optimized_v11_freeze_tag_binding": (
            config.optimized_v11_freeze_tag_binding_artifact,
            config.optimized_v11_freeze_tag_binding_sha256,
        ),
        "budget_authorization": (
            config.budget_authorization,
            config.budget_authorization_sha256,
        ),
        "ledger_snapshot": (
            config.ledger_snapshot_artifact,
            config.ledger_snapshot_sha256,
        ),
    }
    paths = {
        label: validate_bound_artifact(
            repository_root, reference=reference, expected_sha256=expected
        )
        for label, (reference, expected) in bindings.items()
    }
    status = _load_object(paths["integrated_k_status"], label="K status")
    manifest = _load_object(paths["integrated_k_remote_manifest"], label="K remote manifest")
    provider = _load_object(paths["integrated_k_provider_cleanup"], label="K provider cleanup")
    freeze = _load_object(paths["optimized_v11_freeze"], label="v11 freeze")
    tag_binding = _load_object(paths["optimized_v11_freeze_tag_binding"], label="v11 tag binding")
    budget = _load_object(paths["budget_authorization"], label="budget authorization")
    ledger_snapshot = _load_object(paths["ledger_snapshot"], label="ledger snapshot")
    status_cleanup = status.get("cleanup")
    status_budget = status.get("budget")
    freeze_run = freeze.get("immutable_run_evidence")
    artifacts = manifest.get("artifacts")
    if (
        status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-status/v1"
        or status.get("status") != "SCIENTIFICALLY_VALID"
        or status.get("scientifically_valid") is not True
        or status.get("attempt_id") != "exp004-v11-integrated-s41-k"
        or status.get("freeze_authorized") is not True
        or status.get("kill_and_recompute_authorized_after_freeze") is not True
        or not isinstance(status_cleanup, dict)
        or not isinstance(status_cleanup.get("provider"), dict)
        or status_cleanup["provider"].get("artifact")
        != config.integrated_k_provider_cleanup_artifact
        or status_cleanup["provider"].get("sha256") != config.integrated_k_provider_cleanup_sha256
        or status_cleanup["provider"].get("status") != "PASS"
        or not isinstance(status_cleanup.get("in_function"), dict)
        or status_cleanup["in_function"].get("status") != "PASS"
        or not isinstance(status_budget, dict)
        or status_budget.get("settled_ledger_sha256") != _INTEGRATED_K_SETTLED_LEDGER_SHA256
        or status_budget.get("reservations_empty") is not True
        or manifest.get("attempt_id") != "exp004-v11-integrated-s41-k"
        or not isinstance(artifacts, list)
        or len(artifacts) != config.integrated_k_remote_manifest_entries
        or len({item.get("relative_path") for item in artifacts if isinstance(item, dict)})
        != config.integrated_k_remote_manifest_entries
        or any(
            not isinstance(item, dict)
            or set(item) != {"relative_path", "sha256", "bytes"}
            or not isinstance(item.get("relative_path"), str)
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            or type(item.get("bytes")) is not int
            or item["bytes"] < 0
            or Path(item["relative_path"]).is_absolute()
            or ".." in Path(item["relative_path"]).parts
            or Path(item["relative_path"]).as_posix() != item["relative_path"]
            for item in artifacts
        )
    ):
        raise RuntimeError("kill/recompute K scientific bundle commitment is invalid")
    manifest_root = paths["integrated_k_remote_manifest"].parent
    if not manifest_root.is_dir() or manifest_root.is_symlink():
        raise RuntimeError("kill/recompute K manifest root is not a regular directory")
    descendants = tuple(manifest_root.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise RuntimeError("kill/recompute K manifest tree contains a symlink")
    expected_rows = {str(item["relative_path"]): item for item in artifacts}
    actual_paths = {
        str(path.relative_to(manifest_root)): path
        for path in descendants
        if path.is_file() and path.name != "REMOTE_MANIFEST.json"
    }
    if set(actual_paths) != set(expected_rows):
        raise RuntimeError("kill/recompute K 507-file manifest set differs from disk")
    for relative_path, path in actual_paths.items():
        row = expected_rows[relative_path]
        if (
            path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or _sha256_file(path) != row["sha256"]
        ):
            raise RuntimeError(f"kill/recompute K manifest entry changed: {relative_path}")
    provider_observation = provider.get("provider_observation")
    provider_ledger = provider.get("ledger")
    if (
        provider.get("status") != "PASS"
        or provider.get("attempt_id") != "exp004-v11-integrated-s41-k"
        or not isinstance(provider_observation, dict)
        or any(
            type(provider_observation.get(field)) is not int or provider_observation[field] != 0
            for field in (
                "active_apps",
                "running_tasks",
                "running_containers",
                "endpoints",
                "queues",
                "dicts",
                "provider_reservations",
            )
        )
        or provider_observation.get("unexpected_persistent_volumes") != []
        or not isinstance(provider_ledger, dict)
        or provider_ledger.get("active_reservations") != 0
        or provider_ledger.get("sha256_after_settlement") != _INTEGRATED_K_SETTLED_LEDGER_SHA256
    ):
        raise RuntimeError("kill/recompute K cleanup/ledger evidence is invalid")
    scientific = freeze.get("scientific_validity")
    freeze_authorization = freeze.get("kill_and_recompute_authorization")
    runtime_closure = freeze.get("runtime_source_closure")
    expected_runtime_closure = {
        "integrated_controller": (
            "experiments/branchfabric/gpu_reclamation_integrated_controller_v11.py",
            "c7c07ea0444874a100aa6d6da273058edd4f8cee03548e2c08c62bca5ecae2fc",
        ),
        "integrated_worker": (
            "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py",
            "a9312a017116ffc86ac311b578da67f2afb97198815f30369c8ee3e0159c2002",
        ),
        "early_trigger_overlay": (
            "experiments/branchfabric/gpu_reclamation_integrated_trigger_v11.py",
            "9e81ab90ed6d7cb8f7f77e9fd63f2a0b7f3001422c53bcef84eb01344c83d2cc",
        ),
        "optimized_v11_worker": (
            "experiments/branchfabric/gpu_reclamation_worker_v11.py",
            "08449e5af89759847add7433054b2383eb9b1659785fb7e1184f25dc5e02de11",
        ),
        "v11_pipeline": (
            "python/sloforge/continuum/adapters/vllm_reclamation_v11.py",
            "697d9d9786fbd1759bc4e4242f5b507a7123d1f6daf1af42c12cffc790dde458",
        ),
        "source_identity_and_ownership": (
            "python/sloforge/continuum/adapters/vllm_reclamation_v11_ownership.py",
            "0ca553fbe21b431bc315553e792b7b4de523da3c44f84f36e95ff469ba3e6de0",
        ),
        "engine_step_binding": (
            "python/sloforge/continuum/adapters/vllm_reclamation_v11_sync.py",
            "05dc160d881f0b772d77db4f6b96298fe4290a7de81be9abcc876e83cada2820",
        ),
    }
    if (
        freeze.get("schema_version")
        != "sloforge.branchfabric.experiment-004-optimized-preserve-v11-freeze/v1"
        or freeze.get("status") != "FROZEN"
        or freeze.get("freeze_name") != config.optimized_v11_freeze_tag
        or not isinstance(scientific, dict)
        or scientific.get("artifact") != config.integrated_k_status_artifact
        or scientific.get("sha256") != config.integrated_k_status_sha256
        or scientific.get("scientifically_valid") is not True
        or not isinstance(freeze_authorization, dict)
        or freeze_authorization.get("authorized") is not True
        or freeze_authorization.get("maximum_invocations") != 1
        or freeze_authorization.get("maximum_wall_seconds")
        != V11_KILL_RECOMPUTE_RESERVATION_WALL_SECONDS
        or freeze_authorization.get("maximum_gpu_seconds") != 1320.0
        or freeze_authorization.get("requires_fresh_attempt_id") is not True
        or freeze_authorization.get("must_bind_this_freeze_record_sha256") is not True
        or not isinstance(freeze_run, dict)
        or freeze_run.get("remote_manifest_sha256") != config.integrated_k_remote_manifest_sha256
        or freeze_run.get("verified_entries") != 507
        or freeze_run.get("expected_entries") != 507
        or freeze_run.get("provider_cleanup_sha256") != config.integrated_k_provider_cleanup_sha256
        or freeze_run.get("settled_ledger_sha256") != _INTEGRATED_K_SETTLED_LEDGER_SHA256
        or freeze.get("frozen_properties", {}).get("post_freeze_mutation_permitted") is not False
        or not isinstance(runtime_closure, dict)
        or set(runtime_closure) != set(expected_runtime_closure)
    ):
        raise RuntimeError("kill/recompute optimized-v11 freeze is invalid")
    for label, (reference, expected_sha256) in expected_runtime_closure.items():
        binding = runtime_closure.get(label)
        if (
            not isinstance(binding, dict)
            or binding.get("artifact") != reference
            or binding.get("sha256") != expected_sha256
        ):
            raise RuntimeError(f"kill/recompute frozen runtime binding changed: {label}")
        validate_bound_artifact(
            repository_root,
            reference=reference,
            expected_sha256=expected_sha256,
        )
    if (
        tag_binding.get("schema_version")
        != "sloforge.branchfabric.experiment-004-optimized-preserve-v11-tag-binding/v1"
        or tag_binding.get("status") != "PASS"
        or tag_binding.get("tag") != config.optimized_v11_freeze_tag
        or tag_binding.get("annotated_tag_object") != config.optimized_v11_freeze_tag_object
        or tag_binding.get("freeze_record") != config.optimized_v11_freeze_artifact
        or tag_binding.get("freeze_record_sha256") != config.optimized_v11_freeze_sha256
        or tag_binding.get("scientific_status") != config.integrated_k_status_artifact
        or tag_binding.get("scientific_status_sha256") != config.integrated_k_status_sha256
        or tag_binding.get("tag_message_binds_freeze_and_scientific_status_hashes") is not True
    ):
        raise RuntimeError("kill/recompute optimized-v11 tag binding is invalid")
    if (
        budget.get("schema_version")
        != "sloforge.branchfabric.experiment-004-budget-authorization/v1"
        or budget.get("authorized_gpu_budget_usd") != 80.0
        or budget.get("authorized_cumulative_gpu_seconds") != 43_200.0
        or "one kill-and-recompute comparison only after integrated v11 passes and freezes"
        not in budget.get("authorized_scope", [])
        or ledger_snapshot.get("schema_version")
        != "sloforge.branchfabric.experiment-004-gpu-hours/v1"
        or ledger_snapshot.get("reservations") != []
        or config.ledger_snapshot_sha256 != config.ledger_sha256_before_reservation
    ):
        raise RuntimeError("kill/recompute budget authorization is not current and empty")
    return {
        "schema_version": "sloforge.branchfabric.kill-recompute-sealed-evidence/v1",
        "attempt_id": config.attempt_id,
        "integrated_k_attempt_id": "exp004-v11-integrated-s41-k",
        "integrated_k_remote_manifest_entries": len(artifacts),
        "integrated_k_scientifically_valid": True,
        "integrated_k_in_function_cleanup": "PASS",
        "integrated_k_provider_cleanup": "PASS",
        "optimized_v11_frozen": True,
        "maximum_invocations": 1,
        "bindings": {
            label: {"artifact": reference, "sha256": expected}
            for label, (reference, expected) in sorted(bindings.items())
        },
        "passed": True,
    }


def _kill_transaction_command(
    *,
    config: Experiment004V11KillRecomputeConfig,
    effective_config: Mapping[str, Any],
    selection_sha256: str,
    sanity_result_sha256: str,
) -> dict[str, Any]:
    from gpu_reclamation_integrated_worker_v11 import expanded_runtime_config

    if dict(effective_config) != expanded_runtime_config(config):
        raise RuntimeError("kill/recompute effective config changed before handoff")
    return {
        "schema_version": "sloforge.branchfabric.kill-recompute-transaction-command/v1",
        "effective_config": dict(effective_config),
        "selection_sha256": selection_sha256,
        "authorization_artifact_hash": config.budget_authorization_sha256,
        "sanity_result_sha256": sanity_result_sha256,
        "issued_at_monotonic_ns": __import__("time").monotonic_ns(),
    }


def _finite_positive_float(value: Any) -> bool:
    return type(value) is float and math.isfinite(value) and value > 0.0


def _validate_replay_history(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise RuntimeError("kill/recompute replay history is absent")
    rows = value.get("branches")
    expected_branches = [f"branch.{index}" for index in range(8)]
    if (
        value.get("schema_version") != "sloforge.branchfabric.kill-recompute-history/v2"
        or not isinstance(rows, list)
        or len(rows) != 8
        or value.get("analytical_independent_branch_expectation_tokens") != 133_120
        or value.get("computed_state_topology_tokens") != 18_432
        or value.get("lost_private_rollout_work_tokens") != 2_048
        or value.get("passed") is not True
    ):
        raise RuntimeError("kill/recompute replay history aggregate is invalid")
    submitted: dict[str, int] = {}
    common_prefix: tuple[int, ...] | None = None
    for expected_branch, row in zip(expected_branches, rows, strict=True):
        if not isinstance(row, Mapping):
            raise RuntimeError("kill/recompute replay history row is malformed")
        token_ids = row.get("token_ids")
        if (
            set(row)
            != {
                "logical_branch_id",
                "prefix_tokens_submitted",
                "private_tokens_submitted",
                "total_tokens_submitted",
                "computed_tokens",
                "computed_private_tokens",
                "uncomputed_tail_token_count",
                "uncomputed_tail_token_ids",
                "token_history_sha256",
                "computed_boundary_sha256",
                "token_ids",
            }
            or row.get("logical_branch_id") != expected_branch
            or row.get("prefix_tokens_submitted") != 16_384
            or row.get("computed_tokens") != 16_640
            or row.get("computed_private_tokens") != 256
            or type(row.get("uncomputed_tail_token_count")) is not int
            or row.get("uncomputed_tail_token_count") != 1
            or not isinstance(token_ids, list)
            or len(token_ids) != 16_640 + row.get("uncomputed_tail_token_count", -1)
            or row.get("private_tokens_submitted") != len(token_ids) - 16_384
            or row.get("total_tokens_submitted") != len(token_ids)
            or row.get("uncomputed_tail_token_ids") != token_ids[16_640:]
            or any(type(token) is not int or token < 0 for token in token_ids)
            or row.get("token_history_sha256")
            != hashlib.sha256(_canonical_bytes(token_ids)).hexdigest()
            or row.get("computed_boundary_sha256")
            != hashlib.sha256(_canonical_bytes(token_ids[:16_640])).hexdigest()
        ):
            raise RuntimeError("kill/recompute replay history row is invalid")
        prefix = tuple(token_ids[:16_384])
        if common_prefix is None:
            common_prefix = prefix
        elif prefix != common_prefix:
            raise RuntimeError("kill/recompute replay history prefix is not shared")
        submitted[expected_branch] = len(token_ids)
    live_tail_tokens = sum(length - 16_640 for length in submitted.values())
    submitted_private = sum(length - 16_384 for length in submitted.values())
    submitted_total = sum(submitted.values())
    if (
        live_tail_tokens != 8
        or value.get("computed_boundary_tokens") != 133_120
        or value.get("live_uncomputed_tail_tokens") != live_tail_tokens
        or value.get("submitted_prefix_tokens") != 131_072
        or value.get("submitted_private_tokens") != submitted_private
        or value.get("submitted_replay_tokens") != submitted_total
        or value.get("submitted_vs_analytical_delta_tokens") != live_tail_tokens
        or value.get("unique_submitted_replay_history_tokens") != 18_432 + live_tail_tokens
        or submitted_private != 2_048 + live_tail_tokens
        or submitted_total != 133_120 + live_tail_tokens
    ):
        raise RuntimeError("kill/recompute replay history aggregate is inconsistent")
    return submitted


def _validate_kill_worker_results(
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected: Mapping[str, tuple[int, str]],
    attempt_id: str,
    targeted: bool,
) -> tuple[dict[str, Any], ...]:
    """Validate both real workers and the measured kill/recompute contract."""

    if targeted or len(payloads) != 2 or set(expected) != {"serving", "rollout"}:
        raise RuntimeError("kill/recompute requires one exact two-worker result pair")
    validated = tuple(dict(item) for item in payloads)
    by_role = {str(item.get("role")): item for item in validated}
    if set(by_role) != {"serving", "rollout"}:
        raise RuntimeError("kill/recompute worker roles are incomplete")
    for role, payload in by_role.items():
        pid, gpu_uuid = expected[role]
        compilation = payload.get("measured_transaction_compilation_observation")
        salt = payload.get("cache_salt_evidence")
        if (
            payload.get("status") != "succeeded"
            or payload.get("mode") != "KILL_AND_RECOMPUTE"
            or payload.get("attempt_id") != attempt_id
            or payload.get("pid") != pid
            or payload.get("physical_gpu_uuid") != gpu_uuid
            or not isinstance(compilation, Mapping)
            or set(compilation)
            != {
                "schema_version",
                "source",
                "role",
                "interval_start_ns",
                "interval_end_ns",
                "events",
                "capture_buffer_valid",
                "no_deferred_compilation_event",
                "passed",
            }
            or compilation.get("schema_version")
            != "sloforge.branchfabric.measured-transaction-compilation-observation/v1"
            or compilation.get("source") != "bounded-python-logging-handler"
            or compilation.get("role") != role
            or type(compilation.get("interval_start_ns")) is not int
            or type(compilation.get("interval_end_ns")) is not int
            or compilation["interval_start_ns"] <= 0
            or compilation["interval_start_ns"] >= compilation["interval_end_ns"]
            or compilation.get("events") != []
            or compilation.get("capture_buffer_valid") is not True
            or compilation.get("passed") is not True
            or compilation.get("no_deferred_compilation_event") is not True
            or not isinstance(salt, Mapping)
            or salt.get("passed") is not True
        ):
            raise RuntimeError(f"kill/recompute {role} retained-engine result is invalid")
    serving = by_role["serving"]
    rollout = by_role["rollout"]
    serving_gates = serving.get("scientific_gates")
    restore_interference = serving.get("gpu0_restore_interference")
    expected_serving_gates = {
        "control_stability_pass",
        "gpu0_overload_pass",
        "bounded_backlog_pass",
        "two_gpu_service_gt_offered_pass",
        "queue_drain_pass",
        "slo_restoration_pass",
        "slo_stability_pass",
        "gpu0_active_during_restore_pass",
    }
    if (
        serving.get("schema_version") != "sloforge.branchfabric.kill-recompute-gpu0-result/v1"
        or not isinstance(serving_gates, Mapping)
        or set(serving_gates) != expected_serving_gates
        or any(value is not True for value in serving_gates.values())
        or not isinstance(restore_interference, Mapping)
        or restore_interference.get("active_during_restore_pass") is not True
    ):
        raise RuntimeError("kill/recompute GPU0 serving/recompute interference is invalid")
    correctness = rollout.get("correctness")
    discard = rollout.get("state_discard")
    recompute = rollout.get("recompute")
    continuation = rollout.get("continuation")
    timings = rollout.get("timings")
    proof = discard.get("post_free_ownership") if isinstance(discard, Mapping) else None
    proof_records = proof.get("records") if isinstance(proof, Mapping) else None
    proof_lifetimes: set[tuple[int, int]] = set()
    proof_blocks: set[int] = set()
    proof_runtime_blocks: set[str] = set()
    proof_physical_bytes = 0
    owner_shape_counts: dict[tuple[str, ...], int] = {}
    runtime_owner_by_branch: dict[str, str] = {}
    expected_branches = {f"branch.{index}" for index in range(8)}
    expected_record_fields = {
        "allocation_epoch",
        "allocator_available",
        "allocator_epoch_tombstoned",
        "block_hash_present",
        "block_id",
        "native_request_tables_absent",
        "no_live_branch_reference",
        "old_session_inaccessible",
        "physical_bytes",
        "post_release_owner",
        "post_release_refcount",
        "pre_release_owner",
        "pre_release_refcount",
        "pre_release_runtime_owner",
        "release_operation",
        "runtime_block_id",
    }
    if isinstance(proof_records, list):
        for row in proof_records:
            owners = row.get("pre_release_owner") if isinstance(row, Mapping) else None
            runtime_owners = (
                row.get("pre_release_runtime_owner") if isinstance(row, Mapping) else None
            )
            if (
                not isinstance(row, Mapping)
                or set(row) != expected_record_fields
                or type(row.get("block_id")) is not int
                or type(row.get("allocation_epoch")) is not int
                or row["block_id"] < 0
                or row["allocation_epoch"] <= 0
                or type(row.get("physical_bytes")) is not int
                or row.get("physical_bytes") != 917_504
                or not isinstance(owners, list)
                or not owners
                or len(owners) != len(set(owners))
                or not set(owners).issubset(expected_branches)
                or not isinstance(runtime_owners, list)
                or len(runtime_owners) != len(owners)
                or len(runtime_owners) != len(set(runtime_owners))
                or any(not isinstance(owner, str) or not owner for owner in runtime_owners)
                or row.get("pre_release_refcount") != len(owners)
                or row.get("post_release_owner") != []
                or row.get("post_release_refcount") != 0
                or row.get("allocator_available") is not True
                or row.get("block_hash_present") is not False
                or row.get("no_live_branch_reference") is not True
                or row.get("old_session_inaccessible") is not True
                or row.get("allocator_epoch_tombstoned") is not True
                or row.get("native_request_tables_absent") is not True
                or row.get("release_operation")
                != "LLMEngine.abort_request->KVCacheManager.free->BlockPool.reset_prefix_cache"
                or row.get("runtime_block_id") != f"vllm:kv-group:0:block:{row.get('block_id')}"
            ):
                raise RuntimeError("kill/recompute post-free ownership row is invalid")
            proof_lifetimes.add((int(row["block_id"]), int(row["allocation_epoch"])))
            proof_blocks.add(int(row["block_id"]))
            proof_runtime_blocks.add(str(row["runtime_block_id"]))
            proof_physical_bytes += int(row["physical_bytes"])
            owner_tuple = tuple(str(owner) for owner in owners)
            owner_shape_counts[owner_tuple] = owner_shape_counts.get(owner_tuple, 0) + 1
            for owner, runtime_owner in zip(owners, runtime_owners, strict=True):
                prior_runtime_owner = runtime_owner_by_branch.setdefault(owner, runtime_owner)
                if prior_runtime_owner != runtime_owner:
                    raise RuntimeError("kill/recompute runtime ownership identity changed")
    scheduler = recompute.get("scheduler_accounting") if isinstance(recompute, Mapping) else None
    raw_schedule = scheduler.get("raw_schedule_rows") if isinstance(scheduler, Mapping) else None
    raw_cache = (
        scheduler.get("raw_cache_admission_rows") if isinstance(scheduler, Mapping) else None
    )
    recomputed_scheduled = {branch: 0 for branch in expected_branches}
    if isinstance(raw_schedule, list):
        for row in raw_schedule:
            if not isinstance(row, Mapping) or any(key not in expected_branches for key in row):
                raise RuntimeError("kill/recompute raw scheduler row is malformed")
            for branch, count in row.items():
                if type(count) is not int or count < 0:
                    raise RuntimeError("kill/recompute raw scheduler count is malformed")
                recomputed_scheduled[str(branch)] += count
    recomputed_cache: dict[str, int] = {}
    runtime_request_ids: set[str] = set()
    if isinstance(raw_cache, list):
        for row in raw_cache:
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "logical_branch_id",
                    "runtime_request_id",
                    "runtime_cache_reused_tokens",
                }
                or row.get("logical_branch_id") not in expected_branches
                or not isinstance(row.get("runtime_request_id"), str)
                or not row["runtime_request_id"]
                or type(row.get("runtime_cache_reused_tokens")) is not int
                or row["runtime_cache_reused_tokens"] < 0
                or row["logical_branch_id"] in recomputed_cache
                or row["runtime_request_id"] in runtime_request_ids
            ):
                raise RuntimeError("kill/recompute raw cache admission is malformed")
            recomputed_cache[str(row["logical_branch_id"])] = int(
                row["runtime_cache_reused_tokens"]
            )
            runtime_request_ids.add(str(row["runtime_request_id"]))
    observed_first = (
        continuation.get("observed_first_tokens") if isinstance(continuation, Mapping) else None
    )
    expected_first = (
        continuation.get("independent_recompute_first_tokens")
        if isinstance(continuation, Mapping)
        else None
    )
    source_build = (
        recompute.get("discarded_source_build_measurement")
        if isinstance(recompute, Mapping)
        else None
    )
    discard_history = discard.get("history") if isinstance(discard, Mapping) else None
    recompute_history = recompute.get("history") if isinstance(recompute, Mapping) else None
    submitted_by_branch = _validate_replay_history(discard_history)
    if discard_history != recompute_history:
        raise RuntimeError("kill/recompute discard/replay histories differ")
    _validate_replay_history(recompute_history)
    trigger_timeline = rollout.get("trigger_timeline")
    serving_runtime = serving.get("serving")
    serving_trigger = (
        serving_runtime.get("reclamation_trigger_evidence")
        if isinstance(serving_runtime, Mapping)
        else None
    )
    serving_trigger_events = (
        serving_trigger.get("events") if isinstance(serving_trigger, Mapping) else None
    )
    ordered_timing_fields = (
        "reclaim_trigger_ns",
        "rollout_admission_stop_ns",
        "state_quiescence_ns",
        "state_discard_started_ns",
        "state_discard_completed_ns",
        "hbm_reclaim_confirmed_ns",
        "gpu1_serving_ready_ns",
        "restore_trigger_ns",
        "replay_started_ns",
        "first_resumed_token_ns",
        "all_branches_resumed_ns",
        "continuation_complete_ns",
    )
    timing_values = (
        [timings.get(field) for field in ordered_timing_fields]
        if isinstance(timings, Mapping)
        else []
    )
    reclaim_timing_ns = timings.get("reclaim_trigger_ns") if isinstance(timings, Mapping) else None
    source_started_ns = (
        source_build.get("started_ns") if isinstance(source_build, Mapping) else None
    )
    source_completed_ns = (
        source_build.get("completed_ns") if isinstance(source_build, Mapping) else None
    )
    private_started_ns = (
        source_build.get("private_rollout_started_ns")
        if isinstance(source_build, Mapping)
        else None
    )
    private_completed_ns = (
        source_build.get("private_rollout_completed_ns")
        if isinstance(source_build, Mapping)
        else None
    )
    trigger_events = (
        trigger_timeline.get("events") if isinstance(trigger_timeline, Mapping) else None
    )
    shared_owners = tuple(f"branch.{index}" for index in range(8))
    expected_owner_shapes = {shared_owners: 1024}
    expected_owner_shapes.update({(branch,): 16 for branch in shared_owners})
    numeric_fields = (
        "prefill_wall_seconds",
        "prefill_gpu_seconds",
        "uncached_prefill_tokens_per_second_wall",
        "uncached_prefill_tokens_per_gpu_second",
        "restore_trigger_to_first_resumed_token_seconds",
        "restore_trigger_to_all_branches_resumed_seconds",
        "restore_trigger_to_all_branches_complete_seconds",
    )
    if (
        rollout.get("schema_version") != "sloforge.branchfabric.kill-recompute-gpu1-result/v1"
        or not isinstance(correctness, Mapping)
        or set(correctness)
        != {
            "state_discard_pass",
            "no_checkpoint_export_pass",
            "post_free_ownership_pass",
            "allocator_hbm_agreement_pass",
            "gpu1_serving_after_release_pass",
            "actual_recompute_accounting_pass",
            "all_branches_resumed_pass",
            "first_token_exact_8_of_8_pass",
        }
        or any(value is not True for value in correctness.values())
        or not isinstance(discard, Mapping)
        or discard.get("export_started") is not False
        or discard.get("checkpoint_materialized") is not False
        or discard.get("source_block_count") != 1152
        or discard.get("kv_assigned_bytes_before") != 1_056_964_608
        or discard.get("kv_assigned_bytes_after") != 0
        or discard.get("allocator_hbm_agreement") is not True
        or discard.get("passed") is not True
        or not isinstance(proof, Mapping)
        or proof.get("schema_version") != "sloforge.continuum.vllm-v11-post-free-ownership-proof/v1"
        or proof.get("passed") is not True
        or proof.get("source_block_count") != 1152
        or proof.get("source_physical_bytes") != 1_056_964_608
        or proof.get("kv_assigned_bytes_before") != 1_056_964_608
        or proof.get("kv_assigned_bytes_after") != 0
        or proof.get("exact_source_coverage") is not True
        or proof.get("all_old_sessions_destroyed") is not True
        or proof.get("root_reference_released") is not True
        or proof.get("old_session_layout_inaccessible") is not True
        or proof.get("no_live_branch_reference") is not True
        or proof.get("native_manager_request_tables_absent") is not True
        or proof.get("all_allocator_epochs_tombstoned") is not True
        or proof.get("allocator_hbm_agreement") is not True
        or proof.get("full_free_pool_recovered") is not True
        or not isinstance(proof_records, list)
        or len(proof_records) != 1152
        or len(proof_lifetimes) != 1152
        or len(proof_blocks) != 1152
        or len(proof_runtime_blocks) != 1152
        or proof_physical_bytes != 1_056_964_608
        or owner_shape_counts != expected_owner_shapes
        or set(runtime_owner_by_branch) != expected_branches
        or proof.get("runtime") != "vllm"
        or proof.get("runtime_version") != "0.23.0"
        or proof.get("device") != "cuda:0"
        or proof.get("root_session_id") != f"{attempt_id}-root"
        or not isinstance(proof.get("root_reference_id"), str)
        or not proof["root_reference_id"].startswith(f"vllm-root:{attempt_id}-root:")
        or proof.get("branch_session_ids") != [f"branch.{index}" for index in range(8)]
        or proof.get("pool_free_block_count") != 48_475
        or proof.get("pool_usable_block_count") != 48_475
        or proof.get("kv_pool_reserved_bytes_before") != 44_476_923_904
        or proof.get("kv_pool_reserved_bytes_after") != 44_476_923_904
        or proof.get("kv_unassigned_bytes_before") != 43_419_959_296
        or proof.get("kv_unassigned_bytes_after") != 44_476_923_904
        or any(
            type(proof.get(field)) is not int or proof[field] < 0
            for field in (
                "kv_assigned_bytes_before",
                "kv_assigned_bytes_after",
                "kv_pool_reserved_bytes_before",
                "kv_pool_reserved_bytes_after",
                "kv_unassigned_bytes_before",
                "kv_unassigned_bytes_after",
                "pool_free_block_count",
                "pool_usable_block_count",
            )
        )
        or proof.get("pool_free_block_count") <= 0
        or (proof.get("pool_usable_block_count") + 1) * 917_504
        != proof.get("kv_pool_reserved_bytes_after")
        or proof.get("kv_pool_reserved_bytes_after") != proof.get("kv_pool_reserved_bytes_before")
        or proof.get("kv_unassigned_bytes_before") + proof.get("kv_assigned_bytes_before")
        != proof.get("kv_pool_reserved_bytes_before")
        or proof.get("kv_unassigned_bytes_after") - proof.get("kv_unassigned_bytes_before")
        != proof.get("kv_assigned_bytes_before")
        or type(proof.get("release_started_at_monotonic_ns")) is not int
        or type(proof.get("release_completed_at_monotonic_ns")) is not int
        or not (
            discard.get("state_discard_started_ns")
            <= proof["release_started_at_monotonic_ns"]
            <= proof["release_completed_at_monotonic_ns"]
            <= discard.get("state_discard_completed_ns")
        )
        or not isinstance(recompute, Mapping)
        or not isinstance(scheduler, Mapping)
        or scheduler.get("schema_version")
        != "sloforge.branchfabric.vllm-replay-scheduler-accounting/v1"
        or not isinstance(raw_schedule, list)
        or len(raw_schedule) != scheduler.get("scheduler_schedule_calls")
        or not isinstance(raw_cache, list)
        or len(raw_cache) != 8
        or set(recomputed_cache) != expected_branches
        or scheduler.get("raw_schedule_rows_sha256")
        != hashlib.sha256(_canonical_bytes(raw_schedule)).hexdigest()
        or scheduler.get("raw_cache_admission_rows_sha256")
        != hashlib.sha256(_canonical_bytes(raw_cache)).hexdigest()
        or scheduler.get("runtime_uncached_scheduled_tokens_by_branch")
        != dict(sorted(recomputed_scheduled.items()))
        or scheduler.get("runtime_cache_reused_tokens_by_branch")
        != dict(sorted(recomputed_cache.items()))
        or scheduler.get("runtime_uncached_scheduled_tokens") != sum(recomputed_scheduled.values())
        or scheduler.get("runtime_cache_reused_tokens") != sum(recomputed_cache.values())
        or scheduler.get("submitted_tokens_by_branch") != dict(sorted(submitted_by_branch.items()))
        or any(
            recomputed_scheduled[branch] + recomputed_cache[branch] != submitted_by_branch[branch]
            for branch in expected_branches
        )
        or scheduler.get("accounting_conserves_submitted_history") is not True
        or scheduler.get("passed") is not True
        or recompute.get("actual_submitted_prefix_tokens") != 131_072
        or recompute.get("actual_submitted_private_tokens") != 2_056
        or recompute.get("actual_submitted_replay_tokens") != 133_128
        or recompute.get("actual_computed_boundary_tokens") != 133_120
        or recompute.get("actual_live_uncomputed_tail_tokens") != 8
        or recompute.get("submitted_vs_analytical_delta_tokens") != 8
        or recompute.get("unique_submitted_replay_history_tokens") != 18_440
        or recompute.get("computed_state_topology_tokens") != 18_432
        or recompute.get("lost_private_rollout_work_tokens") != 2_048
        or recompute.get("analytical_independent_branch_expectation_tokens") != 133_120
        or type(recompute.get("actual_runtime_cache_reused_tokens")) is not int
        or recompute["actual_runtime_cache_reused_tokens"] < 0
        or type(recompute.get("actual_runtime_uncached_recompute_tokens")) is not int
        or recompute["actual_runtime_uncached_recompute_tokens"] <= 0
        or recompute["actual_runtime_cache_reused_tokens"]
        + recompute["actual_runtime_uncached_recompute_tokens"]
        != recompute["actual_submitted_replay_tokens"]
        or recompute.get("actual_runtime_cache_reused_tokens")
        != scheduler.get("runtime_cache_reused_tokens")
        or recompute.get("actual_runtime_uncached_recompute_tokens")
        != scheduler.get("runtime_uncached_scheduled_tokens")
        or recompute.get("gpu_computed_tokens_derived_from_scheduler")
        != scheduler.get("runtime_uncached_scheduled_tokens")
        or recompute.get("gpu_computed_token_provenance")
        != (
            "vllm SchedulerOutput.num_scheduled_tokens through each replay request's "
            "first token; CUDA events provide time, not token counts"
        )
        or any(
            type(recompute.get(field)) is not float
            or not math.isfinite(recompute[field])
            or recompute[field] <= 0.0
            for field in numeric_fields
        )
        or not math.isclose(
            recompute["uncached_prefill_tokens_per_second_wall"],
            recompute["actual_runtime_uncached_recompute_tokens"]
            / recompute["prefill_wall_seconds"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            recompute["uncached_prefill_tokens_per_gpu_second"],
            recompute["actual_runtime_uncached_recompute_tokens"]
            / recompute["prefill_gpu_seconds"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not isinstance(source_build, Mapping)
        or source_build.get("schema_version")
        != "sloforge.branchfabric.kill-recompute-source-build/v1"
        or source_build.get("unique_prefix_tokens") != 16_384
        or source_build.get("private_rollout_tokens") != 2_048
        or source_build.get("unique_source_tokens") != 18_432
        or type(source_started_ns) is not int
        or type(source_completed_ns) is not int
        or type(private_started_ns) is not int
        or type(private_completed_ns) is not int
        or type(reclaim_timing_ns) is not int
        or not (
            0
            < source_started_ns
            <= private_started_ns
            < private_completed_ns
            <= source_completed_ns
            <= reclaim_timing_ns
        )
        or type(source_build.get("wall_seconds")) is not float
        or not math.isfinite(source_build["wall_seconds"])
        or source_build["wall_seconds"] <= 0.0
        or type(source_build.get("gpu_seconds")) is not float
        or not math.isfinite(source_build["gpu_seconds"])
        or source_build["gpu_seconds"] <= 0.0
        or source_build["gpu_seconds"] > source_build["wall_seconds"] + 0.001
        or not math.isclose(
            source_build["wall_seconds"],
            (source_completed_ns - source_started_ns) / 1_000_000_000.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not _finite_positive_float(source_build.get("lost_private_rollout_wall_seconds"))
        or not _finite_positive_float(source_build.get("lost_private_rollout_gpu_seconds"))
        or source_build["lost_private_rollout_gpu_seconds"]
        > source_build["lost_private_rollout_wall_seconds"] + 0.001
        or source_build["lost_private_rollout_wall_seconds"] > source_build["wall_seconds"]
        or source_build["lost_private_rollout_gpu_seconds"] > source_build["gpu_seconds"] + 0.001
        or not math.isclose(
            source_build["lost_private_rollout_wall_seconds"],
            (private_completed_ns - private_started_ns) / 1_000_000_000.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or source_build.get("lost_private_rollout_gpu_time_provenance")
        != (
            "CUDA events enclosing only vLLM run_concurrent_branches for the "
            "eight 256-token private trajectories"
        )
        or source_build.get("passed") is not True
        or recompute.get("passed") is not True
        or not isinstance(continuation, Mapping)
        or not isinstance(observed_first, Mapping)
        or not isinstance(expected_first, Mapping)
        or set(observed_first) != expected_branches
        or set(expected_first) != expected_branches
        or dict(observed_first) != dict(expected_first)
        or any(type(value) is not int or value < 0 for value in observed_first.values())
        or continuation.get("exact_matches") != 8
        or continuation.get("minimum_tokens_per_branch", 0) < 8
        or continuation.get("branch_count") != 8
        or continuation.get("passed") is not True
        or not isinstance(timings, Mapping)
        or len(timing_values) != len(ordered_timing_fields)
        or any(type(value) is not int or value <= 0 for value in timing_values)
        or any(left > right for left, right in pairwise(timing_values))
        or recompute["prefill_gpu_seconds"] > recompute["prefill_wall_seconds"] + 0.001
        or timings.get("state_discard_started_ns") != discard.get("state_discard_started_ns")
        or timings.get("state_discard_completed_ns") != discard.get("state_discard_completed_ns")
        or timings.get("hbm_reclaim_confirmed_ns") != discard.get("hbm_reclaim_confirmed_ns")
        or not math.isclose(
            recompute["prefill_wall_seconds"],
            (timings["all_branches_resumed_ns"] - timings["replay_started_ns"]) / 1_000_000_000.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            recompute["restore_trigger_to_first_resumed_token_seconds"],
            (timings["first_resumed_token_ns"] - timings["restore_trigger_ns"]) / 1_000_000_000.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            recompute["restore_trigger_to_all_branches_resumed_seconds"],
            (timings["all_branches_resumed_ns"] - timings["restore_trigger_ns"]) / 1_000_000_000.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            recompute["restore_trigger_to_all_branches_complete_seconds"],
            (timings["continuation_complete_ns"] - timings["restore_trigger_ns"]) / 1_000_000_000.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not isinstance(trigger_timeline, Mapping)
        or trigger_timeline.get("schema_version") != "sloforge.branchfabric.v11-trigger-timeline/v1"
        or trigger_timeline.get("passed") is not True
        or trigger_timeline.get("trigger_precedes_state_quiescence") is not True
        or not isinstance(trigger_events, Mapping)
        or set(trigger_events)
        != {
            "OVERLOAD_DETECTED",
            "TRIGGER_PREDICATE_SATISFIED",
            "RECLAIM_TRIGGER_EMITTED",
            "ROLLOUT_ADMISSION_STOP",
            "STATE_QUIESCENCE",
        }
        or trigger_events.get("RECLAIM_TRIGGER_EMITTED") != timings["reclaim_trigger_ns"]
        or trigger_events.get("ROLLOUT_ADMISSION_STOP") != timings["rollout_admission_stop_ns"]
        or trigger_events.get("STATE_QUIESCENCE") != timings["state_quiescence_ns"]
        or trigger_timeline.get("controller_reaction_latency_ns")
        != trigger_events["RECLAIM_TRIGGER_EMITTED"] - trigger_events["TRIGGER_PREDICATE_SATISFIED"]
        or trigger_timeline.get("rollout_stop_reaction_latency_ns")
        != trigger_events["ROLLOUT_ADMISSION_STOP"] - trigger_events["RECLAIM_TRIGGER_EMITTED"]
        or not (
            trigger_events["OVERLOAD_DETECTED"]
            <= trigger_events["TRIGGER_PREDICATE_SATISFIED"]
            <= trigger_events["RECLAIM_TRIGGER_EMITTED"]
        )
        or not isinstance(serving_trigger_events, Mapping)
        or serving_trigger_events.get("OVERLOAD_DETECTED") != trigger_events["OVERLOAD_DETECTED"]
        or serving_trigger_events.get("TRIGGER_PREDICATE_SATISFIED")
        != trigger_events["TRIGGER_PREDICATE_SATISFIED"]
        or serving_trigger_events.get("RECLAIM_TRIGGER_EMITTED")
        != trigger_events["RECLAIM_TRIGGER_EMITTED"]
    ):
        raise RuntimeError("kill/recompute GPU1 discard/replay result is invalid")
    _validate_kill_control_consistency(serving, rollout)
    return validated


def _validate_kill_control_consistency(
    serving: Mapping[str, Any], rollout: Mapping[str, Any]
) -> None:
    """Deep-recompute the kill-only control policy through the frozen validators."""

    import gpu_reclamation_integrated_controller_v11 as integrated
    from gpu_reclamation_kill_recompute_worker_v11 import _kill_control_interval_evidence

    frozen_evaluator = integrated._control_interval_evidence

    def kill_evaluator(result: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        return _kill_control_interval_evidence(
            result,
            _base_evaluator=frozen_evaluator,
            **kwargs,
        )

    integrated._control_interval_evidence = kill_evaluator
    try:
        integrated._validate_gpu0_scientific_result(serving)
        integrated._validate_cross_role_scientific_consistency(serving, rollout)
    finally:
        integrated._control_interval_evidence = frozen_evaluator


def run_kill_recompute_controller(
    *,
    config_path: Path,
    work_root: Path,
    worker_path: Path,
    model_snapshot: Path,
    repository_root: Path,
    absolute_deadline_ns: int,
) -> dict[str, Any]:
    """Run the exact inherited controller with scoped kill bindings."""

    import gpu_reclamation_integrated_controller_v11 as integrated

    originals = {
        "config": integrated.Experiment004V11IntegratedConfig,
        "reservation": integrated.V11_INTEGRATED_RESERVATION_WALL_SECONDS,
        "absolute": integrated.ABSOLUTE_WALL_SECONDS,
        "verify": integrated.verify_sealed_evidence,
        "command": integrated._transaction_command,
        "worker_results": integrated._validate_worker_results,
    }
    integrated.Experiment004V11IntegratedConfig = Experiment004V11KillRecomputeConfig
    integrated.V11_INTEGRATED_RESERVATION_WALL_SECONDS = V11_KILL_RECOMPUTE_RESERVATION_WALL_SECONDS
    integrated.ABSOLUTE_WALL_SECONDS = V11_KILL_RECOMPUTE_RESERVATION_WALL_SECONDS
    integrated.verify_sealed_evidence = verify_kill_recompute_evidence
    integrated._transaction_command = _kill_transaction_command
    integrated._validate_worker_results = _validate_kill_worker_results
    try:
        result = integrated.run_integrated_v11_controller(
            config_path=config_path,
            work_root=work_root,
            worker_path=worker_path,
            model_snapshot=model_snapshot,
            repository_root=repository_root,
            absolute_deadline_ns=absolute_deadline_ns,
        )
        forbidden_tokens = (
            "checkpoint",
            "source-capture-commit",
            "pre-export",
            "post-export",
            "source-release",
            "state-pass",
            "movement",
            "optimized-export",
        )
        paths = sorted(
            path.relative_to(work_root).as_posix()
            for path in work_root.rglob("*")
            if path.is_file()
        )
        forbidden_matches = [
            path for path in paths if any(token in path.lower() for token in forbidden_tokens)
        ]
        if forbidden_matches:
            raise RuntimeError("kill/recompute controller observed preservation output artifacts")
        evidence = {
            "schema_version": ("sloforge.branchfabric.kill-recompute-no-preservation-outputs/v1"),
            "scan_scope": "controller work root after both workers and before function return",
            "scanned_file_count": len(paths),
            "scanned_relative_paths_sha256": hashlib.sha256(_canonical_bytes(paths)).hexdigest(),
            "forbidden_path_tokens": list(forbidden_tokens),
            "forbidden_matches": [],
            "checkpoint_materialized": False,
            "optimized_export_executed": False,
            "passed": True,
        }
        proof_path = work_root / "kill-no-preservation-output-artifacts.json"
        if proof_path.exists() or proof_path.is_symlink():
            raise FileExistsError("kill/recompute no-preservation proof already exists")
        with proof_path.open("xb") as handle:
            handle.write(_canonical_bytes(evidence))
        result["no_preservation_output_artifacts"] = evidence
        with (work_root / "kill-controller-result.json").open("xb") as handle:
            handle.write(_canonical_bytes(result))
        return result
    finally:
        integrated._validate_worker_results = originals["worker_results"]
        integrated._transaction_command = originals["command"]
        integrated.verify_sealed_evidence = originals["verify"]
        integrated.ABSOLUTE_WALL_SECONDS = originals["absolute"]
        integrated.V11_INTEGRATED_RESERVATION_WALL_SECONDS = originals["reservation"]
        integrated.Experiment004V11IntegratedConfig = originals["config"]


__all__ = [
    "_kill_transaction_command",
    "_validate_kill_control_consistency",
    "_validate_kill_worker_results",
    "run_kill_recompute_controller",
    "verify_kill_recompute_evidence",
]
