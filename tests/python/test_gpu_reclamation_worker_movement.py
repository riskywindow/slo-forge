from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sloforge.continuum.adapters.real_runtime import (
    PhysicalKvBlockReleaseEvidence,
    PhysicalKvReleaseEvidence,
)
from sloforge.continuum.adapters.vllm_reclamation import (
    NativeAllocationRef,
    NativeCaptureEvidence,
    NativePageBinding,
)
from sloforge.helix.characterization.gpu_reclamation_accounting import (
    EdgeKind,
    MemoryDomain,
    StatePassOperation,
    TransferDirection,
)

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "experiments" / "branchfabric" / "gpu_reclamation_worker.py"
SPEC = importlib.util.spec_from_file_location("gpu_reclamation_worker_movement", WORKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_capture_reads_scheduler_and_blocks_by_runtime_request_identity() -> None:
    requested_objects: list[str] = []
    requested_tables: list[str] = []

    class _View:
        def request_object(self, request_id: str) -> object | None:
            requested_objects.append(request_id)
            return SimpleNamespace(
                num_computed_tokens=3,
                all_token_ids=[11, 12, 13, 14],
            )

        def request_blocks(self, request_id: str) -> tuple[tuple[object, ...], ...]:
            requested_tables.append(request_id)
            return ((SimpleNamespace(block_id=21), SimpleNamespace(block_id=22)),)

    adapter = SimpleNamespace(
        _view=_View(),
        _sessions={"branch.0": SimpleNamespace(runtime_request_id="runtime-request-91")},
    )
    inputs = MODULE._runtime_capture_inputs(
        adapter,
        ("branch.0",),
        parent_logical_branch_id="logical-root",
    )

    assert requested_objects == ["runtime-request-91"]
    assert requested_tables == ["runtime-request-91"]
    assert inputs[0].logical_branch_id == "branch.0"
    assert inputs[0].parent_logical_branch_id == "logical-root"
    assert inputs[0].token_ids == (11, 12, 13, 14)
    assert inputs[0].computed_tokens == 3
    assert inputs[0].source_block_indices == (21, 22)


def test_restore_tracks_randomized_scheduler_ids_separately_from_output_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_sampling_params",
        lambda *, max_tokens, seed: {"max_tokens": max_tokens, "seed": seed},
    )

    class _Engine:
        def add_request(self, external_id: str, prompt: object, params: object) -> str:
            assert prompt == {"prompt_token_ids": [7, 8, 9]}
            assert params == {"max_tokens": 8, "seed": 101}
            return f"{external_id}-R4ND0M1D"

    tables = (SimpleNamespace(logical_branch_id="branch.0", token_ids=(7, 8, 9)),)
    internal, external = MODULE._add_restore_requests(
        _Engine(),
        tables,
        source_sampling_by_branch={"branch.0": {"effective_seed": 101}},
    )

    assert internal == {"branch.0": "branch.0@restore-1-R4ND0M1D"}
    assert external == {"branch.0": "branch.0@restore-1"}


def _ledger() -> object:
    return MODULE._WorkerMovementLedger(
        page_order=(
            ("page.shared", 1, 4, ("branch.0", "branch.1")),
            ("page.private.0", 2, 2, ("branch.0",)),
            ("page.private.1", 3, 3, ("branch.1",)),
        ),
        logical_token_bytes=4,
        physical_page_bytes=16,
        branch_group="group-004",
        device="GPU-fixture",
    )


def test_worker_ledger_tracks_shared_once_and_copy_link_once() -> None:
    ledger = _ledger()
    ledger.record(
        label="source-read",
        operation=StatePassOperation.READ,
        source_memory=MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
        destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        start_ns=0,
        end_ns=10,
        read_basis="physical",
        write_basis="physical",
        temporary_basis="physical",
        temporary_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
    )
    ledger.record(
        label="d2h",
        operation=StatePassOperation.D2H,
        source_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
        destination_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
        start_ns=10,
        end_ns=20,
        read_basis="logical",
        write_basis="logical",
        transfer_direction=TransferDirection.D2H,
        transfer_basis="logical",
    )
    ledger.record(
        label="h2d",
        operation=StatePassOperation.H2D,
        source_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
        destination_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
        start_ns=20,
        end_ns=30,
        read_basis="logical",
        write_basis="logical",
        transfer_direction=TransferDirection.H2D,
        transfer_basis="logical",
    )
    ledger.record(
        label="destination-write",
        operation=StatePassOperation.WRITE,
        source_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        destination_memory=MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
        start_ns=30,
        end_ns=40,
        read_basis="physical",
        write_basis="physical",
        required_unavoidable=True,
    )
    ledger.record(
        label="destination-validation-read",
        operation=StatePassOperation.READ,
        source_memory=MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
        destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        start_ns=40,
        end_ns=50,
        read_basis="physical",
        write_basis="physical",
    )
    report = ledger.report()

    assert report.accounting.logical_state_bytes == 36
    assert {item.segment_id for item in report.logical_segments} == {
        "kv.shared-root",
        "kv.private:branch.0",
        "kv.private:branch.1",
    }
    assert report.accounting.physical_bytes_read == 96
    assert report.accounting.physical_bytes_written == 48
    assert report.accounting.d2h_bytes == 36
    assert report.accounting.h2d_bytes == 36
    transfer_edges = [item for item in report.edges if item.edge_kind is EdgeKind.LINK_TRANSFER]
    assert len(transfer_edges) == 6
    assert sum(item.bytes for item in transfer_edges) == 72
    assert len({item.edge_id for item in transfer_edges}) == len(transfer_edges)


def test_worker_ledger_rejects_unknown_and_noncontiguous_page_subsets() -> None:
    ledger = MODULE._WorkerMovementLedger(
        page_order=tuple(
            (f"shared.{index}", index, 4, ("branch.0", "branch.1")) for index in range(3)
        ),
        logical_token_bytes=4,
        physical_page_bytes=16,
        branch_group="group-004",
        device="GPU-fixture",
    )
    common = {
        "label": "adversarial",
        "operation": StatePassOperation.READ,
        "source_memory": MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
        "destination_memory": MemoryDomain.GPU_TRANSFORM_BUFFER,
        "start_ns": 0,
        "end_ns": 10,
        "read_basis": "physical",
        "write_basis": "physical",
    }
    with pytest.raises(ValueError, match="empty or unknown"):
        ledger.record(**common, logical_page_ids={"missing"})
    with pytest.raises(ValueError, match="non-contiguous"):
        ledger.record(**common, logical_page_ids={"shared.0", "shared.2"})


def test_causal_stage_projection_is_gap_free_and_preserves_measured_intervals() -> None:
    stages = MODULE._causalize_stages(
        (
            MODULE._stage_row("first", 110, 120),
            MODULE._stage_row("second", 125, 140),
        ),
        timeline_start_ns=100,
        timeline_end_ns=140,
    )
    assert [(item["start_ns"], item["end_ns"]) for item in stages] == [
        (100, 120),
        (120, 140),
    ]
    assert [item["measured_duration_ns"] for item in stages] == [10, 15]
    assert [item["orchestration_attributed_ns"] for item in stages] == [10, 5]
    assert sum(item["duration_ns"] for item in stages) == 40

    with pytest.raises(ValueError, match="overlap"):
        MODULE._causalize_stages(
            (
                MODULE._stage_row("first", 100, 120),
                MODULE._stage_row("overlap", 119, 130),
            ),
            timeline_start_ns=100,
            timeline_end_ns=130,
        )


def _capture_evidence() -> NativeCaptureEvidence:
    return NativeCaptureEvidence(
        bindings=tuple(
            NativePageBinding(
                logical_page_id=f"page.{index}",
                source=NativeAllocationRef(
                    gpu_uuid="GPU-fixture", block_index=index + 10, allocation_epoch=index + 2
                ),
            )
            for index in range(3)
        )
    )


def _release_evidence() -> PhysicalKvReleaseEvidence:
    capture = _capture_evidence()
    blocks = tuple(
        PhysicalKvBlockReleaseEvidence(
            runtime_block_id=MODULE._source_runtime_block_id(binding.source.block_index),
            block_index=binding.source.block_index,
            allocation_epoch=binding.source.allocation_epoch,
            native_refcount=0,
            block_hash_present=False,
            allocator_available=True,
            is_null=False,
        )
        for binding in capture.bindings
    )
    return PhysicalKvReleaseEvidence(
        runtime="vllm",
        runtime_version="0.23.0",
        device="cuda:0",
        observed_at_monotonic_ns=100,
        requested_block_ids=tuple(block.runtime_block_id for block in blocks),
        blocks=blocks,
        pool_free_block_count=100,
        pool_usable_block_count=100,
    )


def test_exact_source_release_requires_hash_clear_refcount_zero_and_full_pool() -> None:
    summary = MODULE._validate_exact_source_release(_capture_evidence(), _release_evidence())
    assert summary == {
        "exact_source_block_count": 3,
        "all_native_refcounts_zero": True,
        "all_allocator_available": True,
        "all_hashes_cleared": True,
        "allocation_epochs_match_capture": True,
        "pool_free_block_count": 100,
        "pool_usable_block_count": 100,
        "full_free_pool_recovered": True,
    }

    release = _release_evidence()
    corrupt = release.model_copy(
        update={
            "blocks": (
                release.blocks[0].model_copy(update={"block_hash_present": True}),
                *release.blocks[1:],
            )
        }
    )
    with pytest.raises(RuntimeError, match="fully released"):
        MODULE._validate_exact_source_release(_capture_evidence(), corrupt)

    wrong_epoch = release.model_copy(
        update={
            "blocks": (
                release.blocks[0].model_copy(
                    update={"allocation_epoch": release.blocks[0].allocation_epoch + 1}
                ),
                *release.blocks[1:],
            )
        }
    )
    with pytest.raises(RuntimeError, match="allocation-epoch"):
        MODULE._validate_exact_source_release(_capture_evidence(), wrong_epoch)

    incomplete_pool = release.model_copy(update={"pool_free_block_count": 99})
    with pytest.raises(RuntimeError, match="complete usable KV block pool"):
        MODULE._validate_exact_source_release(_capture_evidence(), incomplete_pool)
