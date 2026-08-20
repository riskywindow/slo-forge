"""One-GPU production worker for Experiment 004 v11 micro-validation.

This module is intentionally isolated from the frozen v10 worker.  It reuses
only stable construction/workload helpers and executes the v11 adapter under
the production vLLM engine-step binding.  Importing the module is CUDA-clean;
all GPU dependencies are imported inside :func:`run_micro_validation`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED_LOGICAL_STATE_BYTES = 1_056_964_608
EXPECTED_TOTAL_BLOCKS = 1_152
EXPECTED_SHARED_BLOCKS = 1_024
EXPECTED_PRIVATE_BLOCKS = 128
_CHUNK_OFFSET_PATTERN = re.compile(r"^chunk-([0-9]{4})-offset-([0-9]{12})$")


class _RssPeakSampler:
    """Bounded actual-process RSS sampler used to isolate host temp deltas."""

    def __init__(self, *, interval_seconds: float = 0.01) -> None:
        if not 0 < interval_seconds <= 0.1:
            raise ValueError("RSS sampling interval must be in (0, 0.1] seconds")
        import psutil  # type: ignore[import-not-found]

        self._process = psutil.Process(os.getpid())
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_bytes = int(self._process.memory_info().rss)
        self.peak_bytes = self.baseline_bytes

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RSS sampler cannot be started twice")

        def sample() -> None:
            while not self._stop.wait(self._interval_seconds):
                self.peak_bytes = max(self.peak_bytes, int(self._process.memory_info().rss))

        self._thread = threading.Thread(target=sample, name="v11-rss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self.peak_bytes = max(self.peak_bytes, int(self._process.memory_info().rss))
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("v11 RSS sampler did not stop within its bound")
        return max(0, self.peak_bytes - self.baseline_bytes)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _public_value(value: Any) -> Any:
    """Serialize evidence without leaking in-process proof seals/objects."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _public_value(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _public_value(model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _public_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_public_value(item) for item in value]
    return value


def validate_exact_micro_topology(
    *,
    page_order: Sequence[tuple[str, int, int, tuple[str, ...]]],
    branch_count: int,
    logical_state_bytes: int,
) -> dict[str, int]:
    """Fail closed on any workload/layout drift before the measured export."""

    shared = sum(1 for _page, _block, _valid, owners in page_order if len(owners) == 8)
    private = sum(1 for _page, _block, _valid, owners in page_order if len(owners) == 1)
    invalid = len(page_order) - shared - private
    block_ids = [int(item[1]) for item in page_order]
    page_ids = [str(item[0]) for item in page_order]
    if (
        branch_count != 8
        or len(page_order) != EXPECTED_TOTAL_BLOCKS
        or shared != EXPECTED_SHARED_BLOCKS
        or private != EXPECTED_PRIVATE_BLOCKS
        or invalid
        or len(block_ids) != len(set(block_ids))
        or len(page_ids) != len(set(page_ids))
        or any(int(item[2]) != 16 for item in page_order)
        or logical_state_bytes != EXPECTED_LOGICAL_STATE_BYTES
    ):
        raise RuntimeError(
            "v11 micro runtime did not produce the exact 16K/fanout-8/1,152-block state"
        )
    return {
        "branch_count": branch_count,
        "shared_blocks": shared,
        "private_blocks": private,
        "total_blocks": len(page_order),
        "logical_state_bytes": logical_state_bytes,
    }


def summarize_state_passes(records: Sequence[Any], *, logical_state_bytes: int) -> dict[str, Any]:
    """Derive measurement totals exclusively from canonical StatePassRecords."""

    if logical_state_bytes <= 0 or not records:
        raise ValueError("v11 pass summary requires records and positive logical bytes")
    from sloforge.continuum.adapters.vllm_reclamation_v11_accounting import (
        canonical_pass_totals,
        canonical_physical_bytes_by_tier,
    )

    canonical_records = tuple(records)
    totals = canonical_pass_totals(canonical_records)
    full = totals["full_physical_bytes"]
    external = totals["external_transfer_bytes"]
    avoidable = totals["avoidable_physical_bytes"]
    required = totals["required_physical_bytes"]
    diagnostic = totals["diagnostic_physical_bytes"]
    if required + avoidable + diagnostic != full:
        raise RuntimeError("v11 pass classification does not conserve physical work")

    # A pass family is independent of chunk/layer/page splitting.  Summing the
    # integer number of logical-state coverages in each nonempty family gives
    # a deterministic full-state pass count without treating 28 MiB chunks as
    # separate algorithmic passes.
    families: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in records:
        phase = str(item.pass_id).split(":", 1)[0]
        key = (
            phase,
            str(item.operation),
            str(item.source_representation),
            str(item.destination_representation),
            str(item.source_memory_tier),
            str(item.destination_memory_tier),
            str(item.required_avoidable),
        )
        family = families.setdefault(key, {"logical_bytes": 0, "physical_bytes": 0, "chunks": {}})
        family["logical_bytes"] += int(item.logical_bytes)
        family["physical_bytes"] += int(item.physical_touch_bytes)
        matched = _CHUNK_OFFSET_PATTERN.fullmatch(str(item.chunk_id))
        if matched is not None:
            chunk_key = (int(matched.group(1)), int(matched.group(2)))
            family["chunks"][chunk_key] = family["chunks"].get(chunk_key, 0) + int(
                item.logical_bytes
            )

    full_state_families: list[str] = []
    incomplete_state_families: list[str] = []
    for key, values in families.items():
        _phase, operation, _source, _destination, source_tier, destination_tier, _class = key
        source_is_only_metadata = (
            "METADATA" in source_tier or "CONTROL" in source_tier
        ) and not any(marker in destination_tier for marker in ("NATIVE", "TRANSPORT", "PINNED"))
        if (
            values["physical_bytes"] <= 0
            or source_is_only_metadata
            or operation.lower()
            in {
                "index_upload",
                "destination_index_upload",
                "cuda_event_synchronize",
                "cuda_stream_synchronize",
            }
        ):
            continue
        cursor = 0
        exact_coverage = True
        for (_chunk_index, offset), extent in sorted(
            values["chunks"].items(), key=lambda item: (item[0][1], item[0][0])
        ):
            if offset != cursor or extent <= 0:
                exact_coverage = False
                break
            cursor += extent
        if exact_coverage and cursor == logical_state_bytes:
            full_state_families.append("|".join(key))
        elif values["chunks"]:
            incomplete_state_families.append("|".join(key))
    full_state_pass_count = len(full_state_families)
    if full_state_pass_count <= 0:
        raise RuntimeError("v11 pass ledger exposes no deterministic full-state pass")
    if incomplete_state_families:
        raise RuntimeError(
            "v11 pass ledger contains incomplete, gapped, or overlapping state-pass families: "
            + ",".join(sorted(incomplete_state_families))
        )
    d2h_bytes = sum(
        int(item.external_transfer_bytes)
        for item in records
        if str(item.operation).lower() == "d2h"
    )
    h2d_bytes = sum(
        int(item.external_transfer_bytes)
        for item in records
        if str(item.operation).lower() == "h2d"
    )
    link_control_bytes = external - d2h_bytes - h2d_bytes
    if link_control_bytes < 0 or d2h_bytes + h2d_bytes + link_control_bytes != external:
        raise RuntimeError("v11 directional/link-control movement does not conserve external bytes")
    return {
        "logical_state_bytes": logical_state_bytes,
        "record_count": len(records),
        "full_physical_bytes": full,
        "full_physical_amplification": full / logical_state_bytes,
        "external_movement_bytes": external,
        "external_movement_amplification": external / logical_state_bytes,
        "d2h_bytes": d2h_bytes,
        "h2d_bytes": h2d_bytes,
        "link_control_bytes": link_control_bytes,
        "avoidable_physical_bytes": avoidable,
        "avoidable_amplification": avoidable / logical_state_bytes,
        "required_physical_bytes": required,
        "diagnostic_physical_bytes": diagnostic,
        "cuda_time_ns": totals["cuda_time_ns_timing_owners"],
        # Recorder synchronization includes explicit event/stream fences and
        # synchronous scalar validation waits at the actual execution sites.
        "synchronization_time_ns": totals["synchronization_wait_ns"],
        "full_state_pass_count": full_state_pass_count,
        "full_state_pass_count_method": (
            "canonical state-memory pass families with gap-free, non-overlapping [0,L) "
            "chunk-offset coverage; explicit index/fence/sync exclusion"
        ),
        "full_state_pass_families": sorted(full_state_families),
        "incomplete_or_overlapping_state_pass_families": sorted(incomplete_state_families),
        "movement_accounting_complete": True,
        "pass_family_count": len(families),
        "physical_bytes_by_tier": canonical_physical_bytes_by_tier(canonical_records),
    }


def validate_micro_result(result: dict[str, Any]) -> None:
    required_true = (
        "allocator_epoch_pass",
        "ownership_release_pass",
        "engine_step_binding_pass",
        "integrity_pass",
        "destination_mapping_commitment_pass",
        "fresh_destination_allocations_pass",
        "all_branches_resumed",
        "first_token_exact_8_of_8",
        "movement_accounting_pass",
    )
    correctness = result.get("correctness")
    if not isinstance(correctness, dict) or any(
        correctness.get(key) is not True for key in required_true
    ):
        raise RuntimeError("v11 micro-validation correctness failed closed")
    if result.get("topology", {}).get("logical_state_bytes") != EXPECTED_LOGICAL_STATE_BYTES:
        raise RuntimeError("v11 micro-validation logical state differs from the exact workload")
    continuation = result.get("continuation", {})
    if continuation.get("minimum_tokens_per_branch", 0) < 8:
        raise RuntimeError("v11 micro-validation did not resume every branch for eight tokens")


def _source_epoch_map(plan: Any) -> dict[int, int]:
    result: dict[int, int] = {}
    for binding in plan.capture_evidence.bindings:
        block_id = int(binding.source.block_index)
        epoch = int(binding.source.allocation_epoch)
        prior = result.setdefault(block_id, epoch)
        if prior != epoch:
            raise RuntimeError("one source block carries multiple allocation epochs")
    if len(result) != EXPECTED_TOTAL_BLOCKS:
        raise RuntimeError("source epoch evidence does not cover 1,152 physical blocks")
    return result


def require_production_export_gate_v11(gate: Any, *, scheduler: Any, manager: Any) -> None:
    from sloforge.continuum.adapters.vllm_reclamation_v11_sync import (
        require_production_gate_v11,
    )

    require_production_gate_v11(
        gate,
        scheduler,
        manager,
        operation="EXPORT_CAPTURE",
    )


def consume_exact_source_allocation_queue_v11(
    scheduler: Any,
    manager: Any,
    *,
    expected_block_ids: Sequence[int],
    export_gate: Any,
    source_allocation_lifetime_sha256: str,
    ownership_snapshot_sha256: str,
    capture_manifest_sha256: str,
) -> dict[str, Any]:
    """Consume only the source lifetime's exact allocator zero-queue evidence.

    vLLM's production KV manager reports newly allocated native blocks through
    ``take_new_block_ids``.  Source construction legitimately populates that
    queue before export.  The restore stager must begin with an empty queue so
    that every subsequently drained ID is attributable to a fresh destination
    allocation.  Bind the one source-side drain to the already authenticated
    1,152-block capture plan and fail closed on any missing, duplicate, or
    unrelated ID.
    """

    from sloforge.continuum.adapters.vllm_reclamation_v11 import (
        V11RuntimeTeardownRequired,
    )

    zeroing_flag = getattr(scheduler, "needs_kv_cache_zeroing", None)
    context: dict[str, Any] = {
        "schema_version": "sloforge.branchfabric.v11-source-allocation-queue/v1",
        "gate_id": getattr(export_gate, "gate_id", None),
        "gate_binding_id": getattr(export_gate, "binding_id", None),
        "gate_operation": getattr(export_gate, "operation", None),
        "needs_kv_cache_zeroing": zeroing_flag,
        "expected_count": len(expected_block_ids),
        "expected_block_ids_sha256": None,
        "source_allocation_lifetime_sha256": source_allocation_lifetime_sha256,
        "ownership_snapshot_sha256": ownership_snapshot_sha256,
        "capture_manifest_sha256": capture_manifest_sha256,
        "destructive_drain_performed": False,
    }

    def teardown(
        message: str, *, affected_block_ids: Sequence[int], extra: dict[str, Any] | None = None
    ) -> V11RuntimeTeardownRequired:
        return V11RuntimeTeardownRequired(
            message,
            affected_block_ids=affected_block_ids,
            teardown_evidence={**context, **(extra or {}), "passed": False},
        )

    require_production_export_gate_v11(
        export_gate,
        scheduler=scheduler,
        manager=manager,
    )
    if zeroing_flag is not False:
        raise teardown(
            "source allocation queue drain requires production attention-only zeroing mode",
            affected_block_ids=(),
        )
    commitments = (
        source_allocation_lifetime_sha256,
        ownership_snapshot_sha256,
        capture_manifest_sha256,
    )
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in commitments):
        raise teardown(
            "source allocation queue commitments are missing or malformed",
            affected_block_ids=(),
        )
    if any(
        isinstance(block_id, bool) or not isinstance(block_id, int)
        for block_id in expected_block_ids
    ):
        raise teardown(
            "source allocation queue expected IDs are not strict integers",
            affected_block_ids=(),
        )
    expected = tuple(expected_block_ids)
    context["expected_block_ids_sha256"] = hashlib.sha256(
        _canonical_bytes(tuple(sorted(expected)))
    ).hexdigest()
    if (
        len(expected) != EXPECTED_TOTAL_BLOCKS
        or len(set(expected)) != len(expected)
        or any(block_id < 0 for block_id in expected)
    ):
        raise teardown(
            "source allocation queue requires 1,152 unique block IDs",
            affected_block_ids=expected,
        )
    take_new_block_ids = getattr(manager, "take_new_block_ids", None)
    if not callable(take_new_block_ids):
        raise teardown(
            "production KV manager exposes no allocation zero-queue",
            affected_block_ids=(),
        )
    try:
        raw_observed = tuple(take_new_block_ids())
    except BaseException as error:
        raise teardown(
            "production source allocation zero-queue drain failed", affected_block_ids=()
        ) from error
    context["destructive_drain_performed"] = True
    context["observed_count"] = len(raw_observed)
    if any(
        isinstance(block_id, bool) or not isinstance(block_id, int) for block_id in raw_observed
    ):
        raise teardown(
            "source allocation zero-queue returned non-integer block IDs",
            affected_block_ids=(),
        )
    observed = raw_observed
    context["observed_block_ids_sha256"] = hashlib.sha256(
        _canonical_bytes(tuple(sorted(observed)))
    ).hexdigest()
    missing = tuple(sorted(set(expected) - set(observed)))
    unexpected = tuple(sorted(set(observed) - set(expected)))
    duplicates = tuple(
        sorted(block_id for block_id in set(observed) if observed.count(block_id) > 1)
    )
    context.update(
        {
            "missing_block_ids": missing,
            "unexpected_block_ids": unexpected,
            "duplicate_block_ids": duplicates,
        }
    )
    if (
        len(observed) != len(expected)
        or len(set(observed)) != len(observed)
        or set(observed) != set(expected)
    ):
        raise teardown(
            "source allocation zero-queue differs from the authenticated capture plan; "
            "runtime teardown required",
            affected_block_ids=observed,
        )
    require_production_export_gate_v11(
        export_gate,
        scheduler=scheduler,
        manager=manager,
    )
    ordered = tuple(sorted(observed))
    return {
        **context,
        "observed_count": len(ordered),
        "observed_block_ids_sha256": hashlib.sha256(_canonical_bytes(ordered)).hexdigest(),
        "exact_capture_plan_match": True,
        "consumed_under_export_capture": True,
        "passed": True,
    }


def _run_continuation(
    engine: Any,
    *,
    output_ids: dict[str, str],
    timeout_seconds: float,
) -> tuple[dict[str, int], dict[str, int], int, int, int]:
    deadline = time.monotonic_ns() + round(timeout_seconds * 1e9)
    counts = {external_id: 0 for external_id in output_ids.values()}
    first_tokens: dict[str, int] = {}
    first_ns: int | None = None
    all_first_ns: int | None = None
    while min(counts.values()) < 8:
        if time.monotonic_ns() >= deadline:
            raise TimeoutError("v11 restored continuation exceeded its bounded timeout")
        outputs = engine.step()
        observed = time.monotonic_ns()
        for output in outputs:
            request_id = str(getattr(output, "request_id", ""))
            if request_id not in counts:
                continue
            completions = getattr(output, "outputs", ())
            tokens = tuple(int(item) for item in getattr(completions[0], "token_ids", ()))
            counts[request_id] = len(tokens)
            if tokens and request_id not in first_tokens:
                first_tokens[request_id] = tokens[0]
                first_ns = observed if first_ns is None else first_ns
                if len(first_tokens) == len(counts):
                    all_first_ns = observed
    complete_ns = time.monotonic_ns()
    if first_ns is None or all_first_ns is None or len(first_tokens) != 8:
        raise RuntimeError("not every v11 restored branch emitted a first token")
    by_logical = {logical: first_tokens[external] for logical, external in output_ids.items()}
    return by_logical, counts, first_ns, all_first_ns, complete_ns


def _run_independent_oracle(
    engine: Any,
    *,
    branch_tables: Sequence[Any],
    source_sampling_by_branch: dict[str, dict[str, Any]],
    sampling_params: Any,
    timeout_seconds: float,
) -> dict[str, int]:
    controls: dict[str, str] = {}
    for table in branch_tables:
        logical = str(table.logical_branch_id)
        external = f"{logical}@v11-independent-recompute"
        engine.add_request(
            external,
            {"prompt_token_ids": list(table.token_ids)},
            sampling_params(
                max_tokens=1,
                seed=int(source_sampling_by_branch[logical]["effective_seed"]),
            ),
        )
        controls[external] = logical
    deadline = time.monotonic_ns() + round(timeout_seconds * 1e9)
    expected: dict[str, int] = {}
    while len(expected) < 8:
        if time.monotonic_ns() >= deadline:
            raise TimeoutError("v11 independent recompute oracle exceeded its timeout")
        for output in engine.step():
            logical = controls.get(str(getattr(output, "request_id", "")))
            if logical is None:
                continue
            completions = getattr(output, "outputs", ())
            tokens = tuple(int(item) for item in getattr(completions[0], "token_ids", ()))
            if tokens:
                expected[logical] = tokens[0]
    return expected


def run_micro_validation(
    config_payload: dict[str, Any],
    *,
    model_snapshot: Path,
    physical_gpu_uuid: str,
    work_root: Path,
) -> dict[str, Any]:
    """Execute the exact real-runtime v11 micro transaction on one visible A100."""

    from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
        Experiment004V11MicroConfig,
        validate_bound_artifact,
    )

    config = Experiment004V11MicroConfig.model_validate(config_payload)
    repository_root = Path(__file__).resolve().parents[2]
    validate_bound_artifact(
        repository_root,
        reference=config.offline_gate_manifest,
        expected_sha256=config.offline_gate_manifest_sha256,
    )
    validate_bound_artifact(
        repository_root,
        reference=config.budget_authorization,
        expected_sha256=config.budget_authorization_sha256,
    )
    os.environ.update(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRITON_CACHE_DIR": str(work_root / "triton"),
        }
    )
    (work_root / "triton").mkdir(parents=True, exist_ok=True)
    import torch  # type: ignore[import-not-found]

    if (
        importlib.metadata.version("vllm") != config.runtime_version
        or importlib.metadata.version("torch") != config.torch_version
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_capability(0) != (8, 0)
        or "A100" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("v11 micro worker runtime pins or one-A100 visibility are invalid")
    inputs = json.loads((model_snapshot / "BRANCHFABRIC_INPUTS.json").read_text())
    from gpu_reclamation_worker import (
        _add_restore_requests,
        _create_adapter,
        _geometry,
        _identity,
        _prepare_rollouts,
        _runtime_capture_inputs,
        _sampling_params,
    )

    from sloforge.continuum.adapters.vllm_reclamation import build_canonical_capture_plan
    from sloforge.continuum.adapters.vllm_reclamation_v11 import (
        V11PipelineConfig,
        Vllm0230EngineStepBinding,
        Vllm0230StreamingRestoreStager,
        capture_native_to_transport_v11,
        destination_target_identity_v11,
        scrub_native_pages_v11,
        verify_transport_streaming_v11,
        write_and_validate_native_subset_v11,
    )
    from sloforge.continuum.adapters.vllm_reclamation_v11_accounting import (
        V11StatePassRecorder,
        validate_authoritative_pass_records,
    )
    from sloforge.continuum.adapters.vllm_reclamation_v11_ownership import (
        capture_source_ownership_v11,
        release_source_and_prove_v11,
    )

    adapter: Any = None
    binding: Any = None
    started_ns = time.monotonic_ns()
    try:
        adapter = _create_adapter(config.model_dump(mode="json"), model_snapshot, physical_gpu_uuid)
        binding = Vllm0230EngineStepBinding(
            adapter._view,
            acquire_timeout_seconds=min(30.0, float(config.transaction_timeout_seconds)),
        )
        prepared = _prepare_rollouts(adapter, config.model_dump(mode="json"), inputs)
        branches = tuple(str(item) for item in prepared["branches"])
        tensors, geometry = _geometry(adapter)
        pipeline = V11PipelineConfig(
            seed=config.seed,
            expected_device_type="cuda",
            expected_cuda_device_index=0,
            expected_gpu_uuid=physical_gpu_uuid,
            maximum_chunk_bytes=config.maximum_chunk_bytes,
            buffer_count=config.buffer_count,
            pin_memory=True,
            asynchronous_copy=True,
        )
        source_sampling_by_branch = {
            branch: {
                "effective_seed": int(
                    adapter._effective_seed(config.seed, adapter._sessions[branch])
                )
            }
            for branch in branches
        }

        export_base_gpu = int(torch.cuda.memory_allocated(0))
        torch.cuda.reset_peak_memory_stats(0)
        export_rss_sampler = _RssPeakSampler()
        export_rss_sampler.start()
        export_started_ns = time.monotonic_ns()
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id=f"{config.attempt_id}:export"
        ) as export_gate:
            runtime_inputs = _runtime_capture_inputs(
                adapter,
                branches,
                parent_logical_branch_id=str(prepared["root_id"]),
            )
            plan = build_canonical_capture_plan(
                branches=runtime_inputs,
                block_size_tokens=geometry.block_size_tokens,
                logical_token_bytes=geometry.logical_token_bytes,
                physical_page_bytes=geometry.physical_page_bytes,
                gpu_uuid=physical_gpu_uuid,
                allocation_epoch_by_block=dict(adapter._observer.block_epochs),
            )
            topology = validate_exact_micro_topology(
                page_order=plan.page_order,
                branch_count=len(plan.branch_tables),
                logical_state_bytes=plan.logical_state_bytes,
            )
            source_epochs = _source_epoch_map(plan)
            ownership_before = capture_source_ownership_v11(
                adapter,
                root_session_id=str(prepared["root_id"]),
                branch_session_ids=branches,
                expected_allocation_epochs=source_epochs,
                expected_source_block_count=config.expected_total_blocks,
                admission_gate=export_gate,
                timeout_s=60.0,
            )
            captured = capture_native_to_transport_v11(
                tensors,
                geometry,
                page_order=plan.page_order,
                branch_tables=plan.branch_tables,
                identity=_identity(),
                config=pipeline,
                engine_step_gate=export_gate,
            )
            source_allocation_lifetime_sha256 = hashlib.sha256(
                _canonical_bytes(sorted(source_epochs.items()))
            ).hexdigest()
            ownership_snapshot_sha256 = hashlib.sha256(
                _canonical_bytes(_public_value(ownership_before))
            ).hexdigest()
            capture_manifest_sha256 = hashlib.sha256(
                captured.state.manifest.canonical_bytes()
            ).hexdigest()
            source_allocation_queue = consume_exact_source_allocation_queue_v11(
                adapter._view.scheduler,
                adapter._view.manager,
                expected_block_ids=tuple(source_epochs),
                export_gate=export_gate,
                source_allocation_lifetime_sha256=source_allocation_lifetime_sha256,
                ownership_snapshot_sha256=ownership_snapshot_sha256,
                capture_manifest_sha256=capture_manifest_sha256,
            )
            ownership_after = release_source_and_prove_v11(
                adapter,
                ownership_before,
                admission_gate=export_gate,
                timeout_s=60.0,
            )
            ownership_after.require_passed()
        export_ended_ns = time.monotonic_ns()
        export_rss_delta = export_rss_sampler.stop()
        export_peak_gpu = int(torch.cuda.max_memory_allocated(0))
        export_temporary_gpu = max(0, export_peak_gpu - export_base_gpu)

        restore_base_gpu = int(torch.cuda.memory_allocated(0))
        torch.cuda.reset_peak_memory_stats(0)
        restore_rss_sampler = _RssPeakSampler()
        restore_rss_sampler.start()
        restore_started_ns = time.monotonic_ns()
        restore_stats: list[Any] = []
        validation_evidence: list[Any] = []
        branch_group = captured.state_passes[0].branch_group
        restore_recorder = V11StatePassRecorder(
            device=f"cuda:0@{physical_gpu_uuid}", branch_group=branch_group
        )
        with binding.critical_section(
            "IMPORT_ADMISSION", gate_id=f"{config.attempt_id}:import"
        ) as import_gate:
            verified = verify_transport_streaming_v11(
                captured.state,
                maximum_chunk_bytes=config.maximum_chunk_bytes,
                pass_recorder=restore_recorder,
            )
            runtime_ids, output_ids = _add_restore_requests(
                adapter._view.llm_engine,
                captured.state.manifest.branches,
                source_sampling_by_branch=source_sampling_by_branch,
            )
            destination_target = destination_target_identity_v11(tensors, geometry, pipeline)

            def write_and_validate(mapping: dict[str, int], epoch_proof: Any) -> Any:
                evidence, stats = write_and_validate_native_subset_v11(
                    captured.state,
                    tensors,
                    geometry,
                    destination_block_indices=mapping,
                    expected_identity=_identity(),
                    verified_transport=verified,
                    config=pipeline,
                    allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
                    engine_step_gate=import_gate,
                    pass_recorder=restore_recorder,
                    caller_guarantees_outer_scrub=True,
                )
                validation_evidence.append(evidence)
                restore_stats.append(stats)
                return evidence

            def scrub(mapping: dict[str, int], epoch_proof: Any) -> Any:
                return scrub_native_pages_v11(
                    tensors,
                    geometry,
                    state=captured.state,
                    verified_transport=verified,
                    config=pipeline,
                    allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
                    destination_block_indices=mapping,
                    engine_step_gate=import_gate,
                    pass_recorder=restore_recorder,
                )

            imported = Vllm0230StreamingRestoreStager(
                adapter._view.scheduler, adapter._view.manager
            ).import_group_v11(
                captured.state,
                verified_transport=verified,
                runtime_request_ids=runtime_ids,
                expected_identity=_identity(),
                destination_target_sha256=destination_target,
                admission_gate=import_gate,
                write_and_validate_pages=write_and_validate,
                scrub_pages_on_failure=scrub,
            )
        restore_passes = restore_recorder.resolve()
        validate_authoritative_pass_records(
            restore_passes,
            expected_device=f"cuda:0@{physical_gpu_uuid}",
            expected_branch_group=branch_group,
        )
        restore_ended_ns = time.monotonic_ns()
        restore_rss_delta = restore_rss_sampler.stop()
        restore_peak_gpu = int(torch.cuda.max_memory_allocated(0))
        restore_temporary_gpu = max(0, restore_peak_gpu - restore_base_gpu)

        source_lifetimes = {(block, epoch) for block, epoch in source_epochs.items()}
        destination_lifetimes = {
            (int(block), int(epoch))
            for evidence in validation_evidence
            for block, epoch in evidence.allocation_epochs
        }
        destination_blocks = set(imported.logical_page_destinations.values())
        if len(destination_blocks) != EXPECTED_TOTAL_BLOCKS or not destination_lifetimes:
            raise RuntimeError("fresh destination allocation evidence is incomplete")
        if source_lifetimes & destination_lifetimes:
            raise RuntimeError("v11 restore reused a stale source allocation lifetime")

        continuation_started_ns = time.monotonic_ns()
        observed_first, continuation_counts, first_ns, all_first_ns, complete_ns = (
            _run_continuation(
                adapter._view.llm_engine,
                output_ids=output_ids,
                timeout_seconds=120.0,
            )
        )
        if not adapter._view.manager.reset_prefix_cache():
            raise RuntimeError("restored requests did not drain before independent oracle")
        oracle_started_ns = time.monotonic_ns()
        expected_first = _run_independent_oracle(
            adapter._view.llm_engine,
            branch_tables=captured.state.manifest.branches,
            source_sampling_by_branch=source_sampling_by_branch,
            sampling_params=_sampling_params,
            timeout_seconds=180.0,
        )
        oracle_ended_ns = time.monotonic_ns()
        exact = observed_first == expected_first and len(observed_first) == 8

        all_passes = tuple(captured.state_passes) + tuple(restore_passes)
        movement = summarize_state_passes(
            all_passes, logical_state_bytes=captured.state.manifest.logical_state_bytes
        )
        export_summary = summarize_state_passes(
            captured.state_passes,
            logical_state_bytes=captured.state.manifest.logical_state_bytes,
        )
        restore_summary = summarize_state_passes(
            restore_passes,
            logical_state_bytes=captured.state.manifest.logical_state_bytes,
        )
        peak_pinned_temporary = max(
            [captured.stats.peak_host_temporary_bytes]
            + [item.peak_host_temporary_bytes for item in restore_stats]
        )
        export_pageable_delta = max(
            0,
            export_rss_delta
            - captured.stats.checkpoint_resident_host_bytes
            - captured.stats.peak_host_temporary_bytes,
        )
        restore_pageable_delta = max(
            0,
            restore_rss_delta - max(item.peak_host_temporary_bytes for item in restore_stats),
        )
        result = {
            "schema_version": "sloforge.branchfabric.experiment-004-v11-micro-result/v1",
            "status": "succeeded",
            "attempt_id": config.attempt_id,
            "seed": config.seed,
            "physical_gpu_uuid": physical_gpu_uuid,
            "runtime": {
                "model": config.model,
                "model_revision": config.model_revision,
                "vllm": importlib.metadata.version("vllm"),
                "torch": importlib.metadata.version("torch"),
                "torch_cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
            },
            "topology": topology,
            "correctness": {
                "allocator_epoch_pass": all(item.allocator_issued for item in validation_evidence),
                "ownership_release_pass": ownership_after.passed,
                "engine_step_binding_pass": binding.evidence().passed,
                "integrity_pass": all(item.passed for item in validation_evidence),
                "destination_mapping_commitment_pass": all(
                    bool(item.destination_mapping_sha256) for item in validation_evidence
                ),
                "fresh_destination_allocations_pass": not bool(
                    source_lifetimes & destination_lifetimes
                ),
                "all_branches_resumed": min(continuation_counts.values()) >= 8,
                "first_token_exact_8_of_8": exact,
                "movement_accounting_pass": bool(all_passes)
                and movement["movement_accounting_complete"] is True,
            },
            "continuation": {
                "observed_first_tokens": observed_first,
                "independent_recompute_first_tokens": expected_first,
                "exact_matches": sum(
                    observed_first.get(key) == expected_first.get(key) for key in expected_first
                ),
                "minimum_tokens_per_branch": min(continuation_counts.values()),
                "continuation_start_ns": continuation_started_ns,
                "first_resumed_token_ns": first_ns,
                "all_branches_resumed_ns": all_first_ns,
                "continuation_complete_ns": complete_ns,
                "oracle_start_ns": oracle_started_ns,
                "oracle_end_ns": oracle_ended_ns,
            },
            "timings": {
                "export_wall_time_ns": export_ended_ns - export_started_ns,
                "export_cuda_time_ns": export_summary["cuda_time_ns"],
                "restore_wall_time_ns": restore_ended_ns - restore_started_ns,
                "restore_cuda_time_ns": restore_summary["cuda_time_ns"],
                "integrity_time_ns": sum(
                    item.wall_end_ns - item.wall_start_ns
                    for item in all_passes
                    if "sha256" in item.operation
                ),
                "synchronization_time_ns": movement["synchronization_time_ns"],
            },
            "temporary_memory": {
                "peak_pinned_host_temporary_bytes": peak_pinned_temporary,
                "peak_pageable_host_temporary_bytes": max(
                    export_pageable_delta, restore_pageable_delta
                ),
                "host_measurement_method": (
                    "10ms process-RSS peak deltas with checkpoint-resident and explicit "
                    "pinned extents subtracted for pageable temporary attribution"
                ),
                "checkpoint_resident_host_bytes": captured.stats.checkpoint_resident_host_bytes,
                "export_process_rss_baseline_bytes": export_rss_sampler.baseline_bytes,
                "export_process_rss_peak_bytes": export_rss_sampler.peak_bytes,
                "export_process_rss_delta_bytes": export_rss_delta,
                "export_pageable_temporary_delta_bytes": export_pageable_delta,
                "restore_process_rss_baseline_bytes": restore_rss_sampler.baseline_bytes,
                "restore_process_rss_peak_bytes": restore_rss_sampler.peak_bytes,
                "restore_process_rss_delta_bytes": restore_rss_delta,
                "restore_pageable_temporary_delta_bytes": restore_pageable_delta,
                "peak_gpu_temporary_bytes": max(
                    export_temporary_gpu,
                    restore_temporary_gpu,
                    captured.stats.peak_transform_temporary_bytes,
                    *[item.peak_transform_temporary_bytes for item in restore_stats],
                ),
                "export_torch_allocated_baseline_bytes": export_base_gpu,
                "export_peak_torch_allocated_bytes": export_peak_gpu,
                "export_peak_torch_temporary_delta_bytes": export_temporary_gpu,
                "restore_torch_allocated_baseline_bytes": restore_base_gpu,
                "restore_peak_torch_allocated_bytes": restore_peak_gpu,
                "restore_peak_torch_temporary_delta_bytes": restore_temporary_gpu,
            },
            "movement": movement,
            "export_movement": export_summary,
            "restore_movement": restore_summary,
            "source_allocations": {
                "count": len(source_lifetimes),
                "allocation_lifetime_sha256": source_allocation_lifetime_sha256,
                "pre_release_ownership_snapshot_sha256": ownership_snapshot_sha256,
                "allocation_zero_queue": source_allocation_queue,
            },
            "destination_allocations": {
                "count": len(destination_lifetimes),
                "physical_block_id_overlap_count": len(set(source_epochs) & destination_blocks),
                "allocation_lifetime_overlap_count": len(source_lifetimes & destination_lifetimes),
                "allocation_lifetime_sha256": hashlib.sha256(
                    _canonical_bytes(sorted(destination_lifetimes))
                ).hexdigest(),
            },
            "engine_step_binding": _public_value(binding.evidence()),
            "post_free_ownership": _public_value(ownership_after),
            "integrity": _public_value(captured.integrity),
            "state_manifest_sha256": capture_manifest_sha256,
            "state_passes": [item.as_dict() for item in all_passes],
            "started_ns": started_ns,
            "ended_ns": time.monotonic_ns(),
        }
        validate_micro_result(result)
        return result
    finally:
        for sampler_name in ("export_rss_sampler", "restore_rss_sampler"):
            sampler = locals().get(sampler_name)
            if isinstance(sampler, _RssPeakSampler):
                sampler.stop()
        if binding is not None:
            binding.close()
        if adapter is not None:
            adapter.cleanup_runtime(timeout_s=float(config.cleanup_timeout_seconds))
        try:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        except (NameError, RuntimeError):
            pass


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--physical-gpu-uuid", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(args.config.resolve(strict=True).read_text())
        result = run_micro_validation(
            payload,
            model_snapshot=args.model_snapshot.resolve(strict=True),
            physical_gpu_uuid=str(args.physical_gpu_uuid),
            work_root=work_root,
        )
    except BaseException as error:
        failure = {
            "schema_version": "sloforge.branchfabric.experiment-004-v11-micro-failure/v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        affected_block_ids = getattr(error, "affected_block_ids", None)
        if affected_block_ids is not None:
            failure["affected_block_ids"] = [int(item) for item in affected_block_ids]
        teardown_evidence = getattr(error, "teardown_evidence", None)
        if teardown_evidence is not None:
            failure["teardown_evidence"] = _public_value(teardown_evidence)
        _write_new(work_root / "failure.json", failure)
        _write_new(work_root / "result.json", failure)
        return 1
    result["completed_at_utc"] = datetime.now(UTC).isoformat()
    _write_new(work_root / "result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
