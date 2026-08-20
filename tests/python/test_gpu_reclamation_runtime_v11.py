from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from sloforge.continuum.adapters.vllm_reclamation_v11_accounting import V11StatePassRecord

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments/branchfabric"


def _worker_module():
    import sys

    sys.path.insert(0, str(EXPERIMENTS))
    try:
        import gpu_reclamation_worker_v11

        return gpu_reclamation_worker_v11
    finally:
        sys.path.remove(str(EXPERIMENTS))


def _record(
    pass_id: str,
    operation: str,
    *,
    logical_bytes: int = 100,
    read: int = 100,
    write: int = 100,
    external: int = 0,
    source_tier: str = "GPU_HBM_NATIVE_POOL",
    destination_tier: str = "HOST_PINNED",
    chunk_id: str = "chunk-0000-offset-000000000000",
    source_representation: str = "SOURCE",
    destination_representation: str = "DESTINATION",
    physical_bytes_by_tier: tuple[tuple[str, int], ...] = (),
) -> V11StatePassRecord:
    return V11StatePassRecord(
        pass_id=pass_id,
        operation=operation,
        source_representation=source_representation,
        destination_representation=destination_representation,
        source_memory_tier=source_tier,
        destination_memory_tier=destination_tier,
        logical_bytes=logical_bytes,
        physical_bytes_read=read,
        physical_bytes_written=write,
        temporary_bytes=0,
        device="cpu",
        cuda_stream=None,
        start_cuda_event=None,
        end_cuda_event=None,
        wall_start_ns=1,
        wall_end_ns=2,
        synchronization_dependency=(),
        chunk_id=chunk_id,
        branch_group="branch-group-test",
        required_avoidable="REQUIRED",
        external_transfer_bytes=external,
        physical_bytes_by_tier=physical_bytes_by_tier,
        cuda_elapsed_ns=None,
    )


def test_exact_micro_topology_accepts_only_16k_fanout8_layout() -> None:
    worker = _worker_module()
    pages = [
        (f"shared.{index}", index, 16, tuple(f"branch.{item}" for item in range(8)))
        for index in range(1024)
    ] + [(f"private.{index}", 1024 + index, 16, (f"branch.{index // 16}",)) for index in range(128)]
    topology = worker.validate_exact_micro_topology(
        page_order=pages,
        branch_count=8,
        logical_state_bytes=worker.EXPECTED_LOGICAL_STATE_BYTES,
    )
    assert topology == {
        "branch_count": 8,
        "shared_blocks": 1024,
        "private_blocks": 128,
        "total_blocks": 1152,
        "logical_state_bytes": 1_056_964_608,
    }
    with pytest.raises(RuntimeError, match="exact 16K"):
        worker.validate_exact_micro_topology(
            page_order=pages[:-1],
            branch_count=8,
            logical_state_bytes=worker.EXPECTED_LOGICAL_STATE_BYTES,
        )
    partial = list(pages)
    page_id, block_id, _valid, owners = partial[-1]
    partial[-1] = (page_id, block_id, 15, owners)
    with pytest.raises(RuntimeError, match="exact 16K"):
        worker.validate_exact_micro_topology(
            page_order=partial,
            branch_count=8,
            logical_state_bytes=worker.EXPECTED_LOGICAL_STATE_BYTES,
        )


def test_source_allocation_queue_is_consumed_only_on_exact_capture_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker_module()
    expected = tuple(range(worker.EXPECTED_TOTAL_BLOCKS))
    commitments = {
        "source_allocation_lifetime_sha256": "1" * 64,
        "ownership_snapshot_sha256": "2" * 64,
        "capture_manifest_sha256": "3" * 64,
    }
    gate_checks: list[object] = []

    def require_gate(gate: object, *, scheduler: object, manager: object) -> None:
        del scheduler, manager
        gate_checks.append(gate)

    monkeypatch.setattr(worker, "require_production_export_gate_v11", require_gate)

    class Scheduler:
        needs_kv_cache_zeroing = False

    class Manager:
        def __init__(self, queued: tuple[object, ...]) -> None:
            self.queued = list(queued)
            self.take_count = 0

        def take_new_block_ids(self) -> list[object]:
            self.take_count += 1
            result = self.queued
            self.queued = []
            return result

    manager = Manager(tuple(reversed(expected)))
    evidence = worker.consume_exact_source_allocation_queue_v11(
        Scheduler(),
        manager,
        expected_block_ids=expected,
        export_gate="sealed-export-gate",
        **commitments,
    )
    assert evidence["observed_count"] == 1_152
    assert (
        evidence["observed_block_ids_sha256"]
        == "5919b50143dc9e0be494a1001577588e039a428906d9ab9ea02a3137abcec8de"
    )
    assert evidence["needs_kv_cache_zeroing"] is False
    assert evidence["source_allocation_lifetime_sha256"] == "1" * 64
    assert evidence["ownership_snapshot_sha256"] == "2" * 64
    assert evidence["capture_manifest_sha256"] == "3" * 64
    assert evidence["exact_capture_plan_match"] is True
    assert evidence["consumed_under_export_capture"] is True
    assert evidence["passed"] is True
    assert manager.queued == []
    assert manager.take_count == 1
    assert gate_checks == ["sealed-export-gate", "sealed-export-gate"]

    for bad_queue in (
        (),
        expected[:-1],
        (*expected[:-1], expected[-2]),
        (*expected[:-1], 2_000),
        (*expected, 2_000),
    ):
        with pytest.raises(RuntimeError, match="authenticated capture plan"):
            worker.consume_exact_source_allocation_queue_v11(
                Scheduler(),
                Manager(tuple(bad_queue)),
                expected_block_ids=expected,
                export_gate=object(),
                **commitments,
            )

    with pytest.raises(RuntimeError, match="1,152 unique"):
        worker.consume_exact_source_allocation_queue_v11(
            Scheduler(),
            Manager(expected),
            expected_block_ids=expected[:-1],
            export_gate=object(),
            **commitments,
        )
    with pytest.raises(RuntimeError, match="exposes no allocation"):
        worker.consume_exact_source_allocation_queue_v11(
            Scheduler(),
            object(),
            expected_block_ids=expected,
            export_gate=object(),
            **commitments,
        )

    for malformed in ((True, *expected[1:]), ("0", *expected[1:]), (-1, *expected[1:])):
        with pytest.raises(RuntimeError, match=r"strict integers|1,152 unique"):
            worker.consume_exact_source_allocation_queue_v11(
                Scheduler(),
                Manager(expected),
                expected_block_ids=malformed,
                export_gate=object(),
                **commitments,
            )
    with pytest.raises(RuntimeError, match="non-integer"):
        worker.consume_exact_source_allocation_queue_v11(
            Scheduler(),
            Manager(("0", *expected[1:])),
            expected_block_ids=expected,
            export_gate=object(),
            **commitments,
        )

    class RaisingManager(Manager):
        def take_new_block_ids(self) -> list[object]:
            raise RuntimeError("injected allocator observation failure")

    with pytest.raises(RuntimeError, match="zero-queue drain failed"):
        worker.consume_exact_source_allocation_queue_v11(
            Scheduler(),
            RaisingManager(expected),
            expected_block_ids=expected,
            export_gate=object(),
            **commitments,
        )
    zeroing_scheduler = Scheduler()
    zeroing_scheduler.needs_kv_cache_zeroing = True
    with pytest.raises(RuntimeError, match="attention-only"):
        worker.consume_exact_source_allocation_queue_v11(
            zeroing_scheduler,
            Manager(expected),
            expected_block_ids=expected,
            export_gate=object(),
            **commitments,
        )


def test_source_queue_drain_order_is_inside_export_after_capture_before_release() -> None:
    worker_path = EXPERIMENTS / "gpu_reclamation_worker_v11.py"
    source_text = worker_path.read_text()
    tree = ast.parse(source_text)
    run = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_micro_validation"
    )
    source = ast.get_source_segment(source_text, run)
    assert source is not None
    gate = source.index('binding.critical_section(\n            "EXPORT_CAPTURE"')
    topology = source.index("validate_exact_micro_topology", gate)
    epochs = source.index("_source_epoch_map", topology)
    ownership = source.index("capture_source_ownership_v11", epochs)
    capture = source.index("capture_native_to_transport_v11", ownership)
    drain = source.index("consume_exact_source_allocation_queue_v11", capture)
    release = source.index("release_source_and_prove_v11", drain)
    gate_end = source.index("export_ended_ns", release)
    assert gate < topology < epochs < ownership < capture < drain < release < gate_end
    assert ".step(" not in source[drain:release]


def test_typed_source_queue_failure_is_persisted_without_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker = _worker_module()
    from sloforge.continuum.adapters.vllm_reclamation_v11 import (
        V11RuntimeTeardownRequired,
    )

    config = tmp_path / "config.json"
    config.write_text("{}\n")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    work_root = tmp_path / "work"

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise V11RuntimeTeardownRequired(
            "typed destructive source queue mismatch",
            affected_block_ids=(11, 7, 11),
            teardown_evidence={
                "expected_count": 1_152,
                "observed_count": 1_151,
                "missing_block_ids": [19],
                "passed": False,
            },
        )

    monkeypatch.setattr(worker, "run_micro_validation", fail)
    status = worker.main(
        [
            "--config",
            str(config),
            "--model-snapshot",
            str(snapshot),
            "--physical-gpu-uuid",
            "GPU-00000000-0000-0000-0000-000000000000",
            "--work-root",
            str(work_root),
        ]
    )
    assert status == 1
    failure = json.loads((work_root / "failure.json").read_text())
    result = json.loads((work_root / "result.json").read_text())
    assert failure == result
    assert failure["status"] == "failed"
    assert failure["error_type"] == "V11RuntimeTeardownRequired"
    assert failure["affected_block_ids"] == [7, 11]
    assert failure["teardown_evidence"] == {
        "expected_count": 1_152,
        "observed_count": 1_151,
        "missing_block_ids": [19],
        "passed": False,
    }
    assert "succeeded" not in failure.values()


def test_summary_uses_canonical_tiers_directions_and_gap_free_pass_families() -> None:
    worker = _worker_module()
    records = (
        _record("capture:chunk:d2h", "d2h", external=100),
        _record(
            "restore:chunk:h2d",
            "h2d",
            external=100,
            source_tier="HOST_PINNED",
            destination_tier="GPU_HBM_TEMPORARY",
        ),
        _record(
            "restore:chunk:index",
            "index_upload",
            read=8,
            write=8,
            external=8,
            source_tier="HOST_PAGEABLE_METADATA",
            destination_tier="GPU_HBM_TEMPORARY",
        ),
    )
    summary = worker.summarize_state_passes(records, logical_state_bytes=100)
    assert summary["d2h_bytes"] == 100
    assert summary["h2d_bytes"] == 100
    assert summary["link_control_bytes"] == 8
    assert (
        summary["d2h_bytes"] + summary["h2d_bytes"] + summary["link_control_bytes"]
        == summary["external_movement_bytes"]
    )
    assert summary["full_state_pass_count"] == 2
    assert len(summary["full_state_pass_families"]) == 2
    assert sum(summary["physical_bytes_by_tier"].values()) == summary["full_physical_bytes"]


def test_pass_count_includes_hash_and_raw_validation_but_rejects_gaps_and_overlaps() -> None:
    worker = _worker_module()
    valid_records = (
        _record(
            "capture:sha:0",
            "sha256_whole_and_pages",
            logical_bytes=40,
            read=80,
            write=0,
            source_tier="HOST_PINNED",
            destination_tier="HOST_PAGEABLE_METADATA",
            chunk_id="chunk-0000-offset-000000000000",
            source_representation="STATE",
            destination_representation="DIGEST",
        ),
        _record(
            "capture:sha:1",
            "sha256_whole_and_pages",
            logical_bytes=60,
            read=120,
            write=0,
            source_tier="HOST_PINNED",
            destination_tier="HOST_PAGEABLE_METADATA",
            chunk_id="chunk-0001-offset-000000000040",
            source_representation="STATE",
            destination_representation="DIGEST",
        ),
        _record(
            "restore:raw:0",
            "raw_bit_equal",
            logical_bytes=50,
            read=100,
            write=0,
            source_tier="GPU_HBM_NATIVE_POOL",
            destination_tier="HOST_CONTROL",
            chunk_id="chunk-0000-offset-000000000000",
            source_representation="NATIVE_STATE",
            destination_representation="RAW_VALIDATION",
        ),
        _record(
            "restore:raw:1",
            "raw_bit_equal",
            logical_bytes=50,
            read=100,
            write=0,
            source_tier="GPU_HBM_NATIVE_POOL",
            destination_tier="HOST_CONTROL",
            chunk_id="chunk-0001-offset-000000000050",
            source_representation="NATIVE_STATE",
            destination_representation="RAW_VALIDATION",
        ),
        _record(
            "restore:index:0",
            "index_upload",
            logical_bytes=100,
            read=8,
            write=8,
            external=8,
            source_tier="HOST_PAGEABLE_METADATA",
            destination_tier="GPU_HBM_TEMPORARY",
        ),
    )
    summary = worker.summarize_state_passes(valid_records, logical_state_bytes=100)
    assert summary["full_state_pass_count"] == 2
    assert any("sha256_whole_and_pages" in item for item in summary["full_state_pass_families"])
    assert any("raw_bit_equal" in item for item in summary["full_state_pass_families"])
    assert all("index_upload" not in item for item in summary["full_state_pass_families"])
    assert summary["incomplete_or_overlapping_state_pass_families"] == []

    valid_anchor = _record("capture:anchor", "anchor_read")
    gap = (
        valid_anchor,
        _record(
            "capture:gap:0",
            "gap_read",
            logical_bytes=40,
            read=40,
            write=0,
            chunk_id="chunk-0000-offset-000000000000",
        ),
        _record(
            "capture:gap:1",
            "gap_read",
            logical_bytes=50,
            read=50,
            write=0,
            chunk_id="chunk-0001-offset-000000000050",
        ),
    )
    with pytest.raises(RuntimeError, match="incomplete, gapped, or overlapping"):
        worker.summarize_state_passes(gap, logical_state_bytes=100)

    overlap = (
        valid_anchor,
        _record(
            "restore:overlap:0",
            "overlap_read",
            logical_bytes=60,
            read=60,
            write=0,
            chunk_id="chunk-0000-offset-000000000000",
        ),
        _record(
            "restore:overlap:1",
            "overlap_read",
            logical_bytes=50,
            read=50,
            write=0,
            chunk_id="chunk-0001-offset-000000000050",
        ),
    )
    with pytest.raises(RuntimeError, match="incomplete, gapped, or overlapping"):
        worker.summarize_state_passes(overlap, logical_state_bytes=100)


def test_summary_honors_explicit_canonical_tier_split_not_display_tiers() -> None:
    worker = _worker_module()
    record = _record(
        "capture:custom-tier",
        "state_read",
        read=40,
        write=30,
        external=30,
        source_tier="DISPLAY_SOURCE",
        destination_tier="DISPLAY_DESTINATION",
        physical_bytes_by_tier=(("CANONICAL_GPU", 40), ("CANONICAL_HOST", 30), ("LINK", 30)),
    )
    summary = worker.summarize_state_passes((record,), logical_state_bytes=100)
    assert summary["physical_bytes_by_tier"] == {
        "CANONICAL_GPU": 40,
        "CANONICAL_HOST": 30,
        "LINK": 30,
    }


def _calls_in(node: ast.AST) -> set[str]:
    result: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        function = child.func
        if isinstance(function, ast.Name):
            result.add(function.id)
        elif isinstance(function, ast.Attribute):
            result.add(function.attr)
    return result


def test_runtime_callsites_hold_one_production_gate_for_each_atomic_phase() -> None:
    source = (EXPERIMENTS / "gpu_reclamation_worker_v11.py").read_text()
    tree = ast.parse(source)
    sections: dict[str, ast.With] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        call = node.items[0].context_expr
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "critical_section"
            and call.args
            and isinstance(call.args[0], ast.Constant)
        ):
            sections[str(call.args[0].value)] = node
    assert set(sections) == {"EXPORT_CAPTURE", "IMPORT_ADMISSION"}
    assert {
        "_runtime_capture_inputs",
        "capture_source_ownership_v11",
        "capture_native_to_transport_v11",
        "release_source_and_prove_v11",
    }.issubset(_calls_in(sections["EXPORT_CAPTURE"]))
    assert {
        "verify_transport_streaming_v11",
        "_add_restore_requests",
        "import_group_v11",
        "write_and_validate_native_subset_v11",
        "scrub_native_pages_v11",
    }.issubset(_calls_in(sections["IMPORT_ADMISSION"]))


def test_restore_callbacks_share_one_authoritative_recorder() -> None:
    source = (EXPERIMENTS / "gpu_reclamation_worker_v11.py").read_text()
    tree = ast.parse(source)
    recorder_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.id
            if isinstance(function, ast.Name)
            else (function.attr if isinstance(function, ast.Attribute) else "")
        )
        if name not in {
            "verify_transport_streaming_v11",
            "write_and_validate_native_subset_v11",
            "scrub_native_pages_v11",
        }:
            continue
        values = {
            keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None
        }
        recorder = values.get("pass_recorder")
        assert isinstance(recorder, ast.Name)
        recorder_calls.append(recorder.id)
        if name == "write_and_validate_native_subset_v11":
            outer_scrub = values.get("caller_guarantees_outer_scrub")
            assert isinstance(outer_scrub, ast.Constant)
            assert outer_scrub.value is True
    assert recorder_calls == ["restore_recorder", "restore_recorder", "restore_recorder"]


def test_modal_definition_is_one_a100_bounded_and_preflight_gated() -> None:
    source = (EXPERIMENTS / "modal_gpu_reclamation_v11.py").read_text()
    tree = ast.parse(source)
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"GPU_REQUEST", "GPU_COUNT", "GPU_FUNCTION_TIMEOUT_SECONDS"}
    }
    assert constants == {
        "GPU_REQUEST": "A100-80GB:1",
        "GPU_COUNT": 1,
        "GPU_FUNCTION_TIMEOUT_SECONDS": 525,
    }
    assert "if _CLOUD_GRAPH_ENABLED:" in source
    assert "SLOFORGE_EXP004_RESERVATION_ID" in source
    assert "modal.Secret.from_dict" in source
    assert "secrets=[_launch_token_secret]" in source
    assert source.count("validate_bound_artifact(") >= 2
    assert "create_if_missing=False" in source


def test_controller_rejects_nonexclusive_or_wrong_gpu_inventory() -> None:
    import sys

    sys.path.insert(0, str(EXPERIMENTS))
    try:
        from gpu_reclamation_controller_v11 import parse_single_a100_inventory
    finally:
        sys.path.remove(str(EXPERIMENTS))
    row = parse_single_a100_inventory("0, GPU-abc, NVIDIA A100-SXM4-80GB, 580.95, 81920, 0, 0\n")
    assert row["uuid"] == "GPU-abc"
    with pytest.raises(RuntimeError, match="exactly one"):
        parse_single_a100_inventory("")
    with pytest.raises(RuntimeError, match="A100-80GB"):
        parse_single_a100_inventory("0, GPU-abc, NVIDIA H100 80GB HBM3, 580.95, 81920, 0, 0\n")
