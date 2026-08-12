from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation import (
    REQUIRED_PRESERVATION_PHASES,
    BranchGroupSemanticEvidence,
    BranchSemanticRecord,
    PilotValidityEvidence,
    ReclamationMode,
    RuntimeAllocationIdentity,
    RuntimeIncarnation,
)
from sloforge.helix.characterization.gpu_reclamation_accounting import (
    LogicalStateSegment,
    MemoryDomain,
    StatePassOperation,
    StatePassRecord,
    StateSegmentKind,
    TransferDirection,
    build_state_movement_report,
)
from sloforge.helix.characterization.gpu_reclamation_analysis import (
    CriticalPath,
    CriticalPathKind,
    Experiment004Outcome,
    MeasuredInterval,
)
from sloforge.helix.characterization.gpu_reclamation_methodology import (
    ArtifactSampleRef,
    Experiment004GpuHourLedger,
)
from sloforge.helix.characterization.gpu_reclamation_report import (
    Experiment004ReportEvidence,
    HbmPlotPoint,
    ReportEmissionMode,
    ReportTrialSummary,
    RuntimeEnvironmentSummary,
    ServingPlotPoint,
    SoftwareOptimizationEvidence,
    build_plot_payloads,
    decision_document_path,
    derive_outcome_evidence,
    derive_software_optimization_evidence,
    publish_experiment_004_report,
)
from sloforge.helix.characterization.gpu_reclamation_serving import ServingSLO
from sloforge.helix.characterization.matrix import EvidenceClass

GPU_UUIDS = ("GPU-first-1", "GPU-second-2")


def _artifact(label: str) -> ArtifactSampleRef:
    return ArtifactSampleRef(
        artifact_reference=f"raw/{label}.json",
        artifact_sha256="a" * 64,
        sample_selector="$",
    )


def _semantics() -> BranchGroupSemanticEvidence:
    source = []
    restored = []
    source_incarnations = []
    restored_incarnations = []
    expected: dict[str, int] = {}
    for index in range(8):
        logical_id = f"branch.{index}"
        common = {
            "logical_branch_id": logical_id,
            "parent_logical_branch_id": "root",
            "policy_epoch": "policy-1",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "a" * 40,
            "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
            "tokenizer_revision": "a" * 40,
            "token_count": 16_641,
            "computed_tokens": 16_640,
            "token_history_sha256": f"{index + 1:064x}",
            "sampling_params_sha256": "f" * 64,
        }
        source_id = f"{logical_id}@source"
        restored_id = f"{logical_id}@restore"
        source.append(BranchSemanticRecord(runtime_request_id=source_id, **common))
        restored.append(BranchSemanticRecord(runtime_request_id=restored_id, **common))
        source_incarnations.append(
            RuntimeIncarnation(
                logical_branch_id=logical_id,
                runtime_request_id=source_id,
                allocations=(
                    RuntimeAllocationIdentity(
                        gpu_uuid=GPU_UUIDS[1], block_index=index, allocation_epoch=1
                    ),
                ),
            )
        )
        restored_incarnations.append(
            RuntimeIncarnation(
                logical_branch_id=logical_id,
                runtime_request_id=restored_id,
                allocations=(
                    RuntimeAllocationIdentity(
                        gpu_uuid=GPU_UUIDS[1], block_index=index, allocation_epoch=2
                    ),
                ),
            )
        )
        expected[logical_id] = 100 + index
    return BranchGroupSemanticEvidence(
        source=tuple(source),
        restored=tuple(restored),
        source_incarnations=tuple(source_incarnations),
        restored_incarnations=tuple(restored_incarnations),
        expected_first_token_ids=expected,
        observed_first_token_ids=expected,
        continuation_token_counts={key: 8 for key in expected},
    )


def _pilot() -> PilotValidityEvidence:
    return PilotValidityEvidence(
        gpu_uuids=GPU_UUIDS,
        gpu_models=("NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB PCIe"),
        both_models_warm_before_trigger=True,
        gpu0_baseline_valid=True,
        branch_count=8,
        prefix_tokens=16_384,
        suffix_tokens=256,
        shared_bytes=80,
        private_bytes=20,
        logical_bytes=100,
        source_assigned_bytes_before=100,
        source_assigned_bytes_after=0,
        source_pool_reserved_bytes_before=1_000,
        source_pool_reserved_bytes_after=1_000,
        source_blocks_allocator_available=True,
        source_hashes_cleared=True,
        transport_integrity_valid=True,
        source_transport_destination_layouts_distinct=True,
        movement_domains_complete=True,
        required_phase_events=REQUIRED_PRESERVATION_PHASES,
        critical_timelines_valid=True,
        gpu1_served_real_request=True,
        serving_metrics_recorded=True,
        restored_branch_count=8,
        all_restored_allocations_fresh=True,
        branch_semantics=_semantics(),
        resumed_continuation_valid=True,
        required_trace_events_dropped=0,
        cleanup_passed=True,
        nvml_hbm_release_claimed=False,
    )


def _movement():
    segment = LogicalStateSegment(
        segment_id="state",
        branch_group_id="group",
        kind=StateSegmentKind.SHARED_KV,
        logical_bytes=100,
    )
    passes = (
        StatePassRecord(
            record_id="d2h",
            state_segment="state",
            branch_group="group",
            operation=StatePassOperation.D2H,
            source_memory=MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
            destination_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
            bytes_read=100,
            bytes_written=100,
            transfer_direction=TransferDirection.D2H,
            transfer_bytes=100,
            logical_bytes=100,
            start_ns=0,
            end_ns=10,
            device="cuda:1",
            temporary_allocation_bytes=100,
            temporary_allocation_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
            temporary_allocation_id="transport",
            required_unavoidable=True,
            read_event_id="d2h-read",
            write_event_id="d2h-write",
            transfer_event_id="d2h-link",
        ),
        StatePassRecord(
            record_id="h2d",
            state_segment="state",
            branch_group="group",
            operation=StatePassOperation.H2D,
            source_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
            destination_memory=MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
            bytes_read=100,
            bytes_written=100,
            transfer_direction=TransferDirection.H2D,
            transfer_bytes=100,
            logical_bytes=100,
            start_ns=10,
            end_ns=20,
            device="cuda:1",
            required_unavoidable=True,
            read_event_id="h2d-read",
            write_event_id="h2d-write",
            transfer_event_id="h2d-link",
        ),
    )
    return build_state_movement_report(logical_segments=(segment,), passes=passes)


def _path(kind: CriticalPathKind, start: int, mode: ReclamationMode) -> CriticalPath:
    prefix = "recompute" if mode is ReclamationMode.KILL_AND_RECOMPUTE else "preserve"
    return CriticalPath(
        kind=kind,
        intervals=(
            MeasuredInterval(name=f"{prefix}-prepare", start_ns=start, end_ns=start + 10),
            MeasuredInterval(name=f"{prefix}-movement", start_ns=start + 10, end_ns=start + 30),
            MeasuredInterval(name=f"{prefix}-ready", start_ns=start + 30, end_ns=start + 40),
        ),
    )


def _trial(mode: ReclamationMode, seed: int) -> ReportTrialSummary:
    preserve = mode is not ReclamationMode.KILL_AND_RECOMPUTE
    return ReportTrialSummary(
        trial_id=f"{mode.value.lower()}-{seed}",
        seed=seed,
        mode=mode,
        evidence_class=EvidenceClass.SYNTHETIC,
        provenance=_artifact(f"{mode.value}-{seed}"),
        gpu_uuids=GPU_UUIDS,
        semantic_validation_passed=True,
        reclamation_path=_path(CriticalPathKind.RECLAMATION, 0, mode),
        restore_path=_path(CriticalPathKind.RESTORE, 100, mode),
        full_transaction_path=_path(CriticalPathKind.FULL_TRANSACTION, 200, mode),
        slo_restoration_path=CriticalPath(
            kind=CriticalPathKind.SLO_RESTORATION,
            intervals=(
                MeasuredInterval(name="slo-degraded", start_ns=0, end_ns=30),
                MeasuredInterval(name="slo-stable", start_ns=30, end_ns=60),
            ),
        ),
        movement=_movement() if preserve else None,
        logical_state_bytes=100,
        kv_logical_state_bytes=100,
        logical_state_accounting_scope=(
            "gpu-resident KV payload; fixture required metadata is separate"
        ),
        host_resident_transport_metadata_bytes=10,
        helix_environment_state_bytes=0,
        helix_environment_state_scope="bounded model-only fixture",
        shared_bytes=80,
        private_bytes=20,
        physical_state_bytes=128,
        physical_block_count=1,
        temporary_memory_bytes=100 if preserve else 0,
        time_to_useful_reclaimed_capacity_ns=40,
        time_to_serving_slo_restoration_ns=60,
        serving_interference_fraction=0.05,
        lost_gpu_work_ns=1_000 if not preserve else 0,
        serving_series=(
            ServingPlotPoint(
                timestamp_ns=0,
                gpu0_ttft_ns=10,
                gpu0_throughput_tokens_per_second=100.0,
                queue_depth=0,
                phase="control",
            ),
            ServingPlotPoint(
                timestamp_ns=100,
                gpu0_ttft_ns=20,
                gpu0_throughput_tokens_per_second=90.0,
                queue_depth=1,
                phase="spike",
            ),
        ),
        hbm_series=(
            HbmPlotPoint(
                timestamp_ns=0,
                gpu0_used_bytes=10,
                gpu1_used_bytes=200,
                phase="control",
            ),
            HbmPlotPoint(
                timestamp_ns=100,
                gpu0_used_bytes=10,
                gpu1_used_bytes=100,
                phase="reclaimed",
            ),
        ),
    )


def _evidence() -> Experiment004ReportEvidence:
    return Experiment004ReportEvidence(
        report_id="fixture-report",
        evidence_class=EvidenceClass.SYNTHETIC,
        environment=RuntimeEnvironmentSummary(
            gpu_models=("NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB PCIe"),
            gpu_uuids=GPU_UUIDS,
            topology_summary="fixture topology",
            cuda_version="fixture",
            driver_version="fixture",
            p2p_capability="fixture only",
        ),
        pilot=_pilot(),
        pilot_provenance=_artifact("pilot"),
        trials=tuple(_trial(mode, 41) for mode in ReclamationMode),
        serving_slo=ServingSLO(
            maximum_p95_ttft_ns=2_000_000_000,
            maximum_p95_inter_token_latency_ns=500_000_000,
        ),
        gpu_hours=Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0),
        software_optimization=SoftwareOptimizationEvidence(
            description="fixture preallocated pinned buffers",
            bottleneck_stage="fixture-transform",
            naive_trial_ids=("preserve_naive-41",),
            optimized_trial_ids=("preserve_optimized-41",),
            measured_reclamation_improvement_fraction=0.1,
            profile_hardware_backed=False,
            semantics_unchanged=True,
            trace_overhead_gate_passed=False,
        ),
    )


def _relative_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_fixture_preview_emits_nine_watermarked_deterministic_plots(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    publication = publish_experiment_004_report(
        _evidence(), first, mode=ReportEmissionMode.FIXTURE_PREVIEW
    )
    reordered = _evidence().model_copy(update={"trials": tuple(reversed(_evidence().trials))})
    publish_experiment_004_report(reordered, second, mode=ReportEmissionMode.FIXTURE_PREVIEW)

    assert not publication.publishable_hardware_report
    assert publication.outcome is None
    assert publication.decision_document is None
    assert publication.characterization_document is None
    assert publication.experiment_005_plan is None
    assert publication.artifact_manifest is None
    assert len(publication.plot_json) == len(publication.plot_svg) == 9
    assert _relative_bytes(first) == _relative_bytes(second)
    assert not (first / "reports/branchfabric-gpu-validation-experiment-004.json").exists()
    report = json.loads(Path(publication.report_json).read_text())
    assert report["fixture_preview"] is True
    assert report["outcome"] is None
    assert all(b"FIXTURE PREVIEW" in Path(path).read_bytes() for path in publication.plot_svg)
    serving_svg = Path(publication.plot_svg[4]).read_text()
    hbm_svg = Path(publication.plot_svg[5]).read_text()
    assert "throughput" in serving_svg and "TTFT" in serving_svg and "spike" in serving_svg
    assert "GPU0 HBM" in hbm_svg and "GPU1 HBM" in hbm_svg
    sankey = json.loads(Path(publication.plot_json[-1]).read_text())
    assert sankey["data"]["edges"] == [edge.model_dump(mode="json") for edge in _movement().edges]


def test_fixture_cannot_emit_final_report_or_decision_document(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot emit final"):
        publish_experiment_004_report(
            _evidence(), tmp_path, mode=ReportEmissionMode.FINAL_HARDWARE_REPORT
        )

    assert not any(tmp_path.rglob("*"))


def test_incomplete_pilot_or_baseline_set_is_rejected() -> None:
    raw = _evidence().model_dump(mode="python")
    replacement = dict(raw["trials"][1])
    replacement["trial_id"] = "naive-second"
    replacement["seed"] = 73
    raw["trials"] = (raw["trials"][0], raw["trials"][1], replacement)
    with pytest.raises(ValidationError, match="requires kill, naive-preserve"):
        Experiment004ReportEvidence.model_validate(raw, strict=True)

    raw = _evidence().model_dump(mode="python")
    raw.pop("pilot")
    with pytest.raises(ValidationError, match="pilot"):
        Experiment004ReportEvidence.model_validate(raw, strict=True)


def test_preview_refuses_overwrite_and_plot_ids_are_exact(tmp_path: Path) -> None:
    evidence = _evidence()
    expected = {f"{index:02d}" for index in range(1, 10)}
    assert {plot.plot_id[:2] for plot in build_plot_payloads(evidence)} == expected
    publish_experiment_004_report(evidence, tmp_path, mode=ReportEmissionMode.FIXTURE_PREVIEW)
    with pytest.raises(FileExistsError, match="already exist"):
        publish_experiment_004_report(evidence, tmp_path, mode=ReportEmissionMode.FIXTURE_PREVIEW)


def test_decision_filename_contract_does_not_alias_the_outcome() -> None:
    assert decision_document_path(Experiment004Outcome.GPU_SOFTWARE_TARGET).endswith(
        "GPU_SOFTWARE_TARGET.md"
    )
    assert decision_document_path(Experiment004Outcome.HOST_PIPELINE_HARDWARE_INTEREST).endswith(
        "STATE_PIPELINE_HARDWARE_INTEREST.md"
    )
    assert decision_document_path(Experiment004Outcome.FABRIC_HARDWARE_INTEREST).endswith(
        "STATE_PIPELINE_HARDWARE_INTEREST.md"
    )
    assert decision_document_path(Experiment004Outcome.MOVEMENT_CLOSED).endswith(
        "MOVEMENT_CLOSED.md"
    )
    assert decision_document_path(Experiment004Outcome.PRESERVATION_NOT_ECONOMIC).endswith(
        "MOVEMENT_CLOSED.md"
    )


def _optimization_movement(start: int):
    segment = LogicalStateSegment(
        segment_id="state",
        branch_group_id="group",
        kind=StateSegmentKind.SHARED_KV,
        logical_bytes=100,
    )
    specs = (
        (
            "source-read",
            StatePassOperation.READ,
            MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
            MemoryDomain.GPU_TRANSFORM_BUFFER,
            0,
            10,
            TransferDirection.NONE,
            True,
        ),
        (
            "unpage",
            StatePassOperation.UNPAGE,
            MemoryDomain.GPU_TRANSFORM_BUFFER,
            MemoryDomain.GPU_TRANSPORT_BUFFER,
            10,
            20,
            TransferDirection.NONE,
            False,
        ),
        (
            "checksum",
            StatePassOperation.CHECKSUM,
            MemoryDomain.GPU_TRANSPORT_BUFFER,
            MemoryDomain.PAGEABLE_HOST_BUFFER,
            20,
            30,
            TransferDirection.NONE,
            False,
        ),
        (
            "d2h",
            StatePassOperation.D2H,
            MemoryDomain.GPU_TRANSPORT_BUFFER,
            MemoryDomain.PINNED_HOST_TRANSPORT,
            30,
            40,
            TransferDirection.D2H,
            True,
        ),
        (
            "h2d",
            StatePassOperation.H2D,
            MemoryDomain.PINNED_HOST_TRANSPORT,
            MemoryDomain.GPU_TRANSPORT_BUFFER,
            100,
            110,
            TransferDirection.H2D,
            True,
        ),
        (
            "unpack",
            StatePassOperation.UNPACK,
            MemoryDomain.GPU_TRANSPORT_BUFFER,
            MemoryDomain.GPU_TRANSFORM_BUFFER,
            110,
            120,
            TransferDirection.NONE,
            False,
        ),
        (
            "repage",
            StatePassOperation.REPAGE,
            MemoryDomain.GPU_TRANSFORM_BUFFER,
            MemoryDomain.GPU_TRANSFORM_BUFFER,
            120,
            130,
            TransferDirection.NONE,
            False,
        ),
        (
            "write",
            StatePassOperation.WRITE,
            MemoryDomain.GPU_TRANSFORM_BUFFER,
            MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
            130,
            140,
            TransferDirection.NONE,
            True,
        ),
    )
    passes = tuple(
        StatePassRecord(
            record_id=name,
            state_segment="state",
            branch_group="group",
            operation=operation,
            source_memory=source,
            destination_memory=destination,
            bytes_read=100,
            bytes_written=100,
            transfer_direction=direction,
            transfer_bytes=100 if direction is not TransferDirection.NONE else 0,
            checksum_bytes=100 if operation is StatePassOperation.CHECKSUM else 0,
            logical_bytes=100,
            start_ns=start + begin,
            end_ns=start + end,
            device="cuda:1",
            temporary_allocation_bytes=(
                100
                if destination
                in {
                    MemoryDomain.GPU_TRANSFORM_BUFFER,
                    MemoryDomain.GPU_TRANSPORT_BUFFER,
                    MemoryDomain.PAGEABLE_HOST_BUFFER,
                }
                else 0
            ),
            temporary_allocation_memory=(
                destination
                if destination
                in {
                    MemoryDomain.GPU_TRANSFORM_BUFFER,
                    MemoryDomain.GPU_TRANSPORT_BUFFER,
                    MemoryDomain.PAGEABLE_HOST_BUFFER,
                }
                else None
            ),
            temporary_allocation_id=(
                f"alloc-{name}"
                if destination
                in {
                    MemoryDomain.GPU_TRANSFORM_BUFFER,
                    MemoryDomain.GPU_TRANSPORT_BUFFER,
                    MemoryDomain.PAGEABLE_HOST_BUFFER,
                }
                else None
            ),
            required_unavoidable=required,
        )
        for name, operation, source, destination, begin, end, direction, required in specs
    )
    return build_state_movement_report(logical_segments=(segment,), passes=passes)


def _derived_fixture() -> Experiment004ReportEvidence:
    trials = []
    for seed in (41, 73, 113):
        for mode in ReclamationMode:
            base = _trial(mode, seed)
            preserve = mode is not ReclamationMode.KILL_AND_RECOMPUTE
            movement = _optimization_movement(0) if preserve else None
            reclamation_end = (
                200
                if mode is ReclamationMode.PRESERVE_NAIVE
                else 150
                if mode is ReclamationMode.PRESERVE_OPTIMIZED
                else 100
            )
            temporary = (
                movement.accounting.total_temporary_allocation_bytes if movement is not None else 0
            )
            trials.append(
                base.model_copy(
                    update={
                        "evidence_class": EvidenceClass.HARDWARE_BACKED_REAL,
                        "movement": movement,
                        "temporary_memory_bytes": temporary,
                        "reclamation_path": CriticalPath(
                            kind=CriticalPathKind.RECLAMATION,
                            intervals=(
                                MeasuredInterval(name="capture", start_ns=0, end_ns=50),
                                MeasuredInterval(
                                    name="movement", start_ns=50, end_ns=reclamation_end
                                ),
                            ),
                        ),
                        "restore_path": CriticalPath(
                            kind=CriticalPathKind.RESTORE,
                            intervals=(MeasuredInterval(name="restore", start_ns=100, end_ns=150),),
                        ),
                        "full_transaction_path": CriticalPath(
                            kind=CriticalPathKind.FULL_TRANSACTION,
                            intervals=(
                                MeasuredInterval(name="transaction", start_ns=0, end_ns=200),
                            ),
                        ),
                        "slo_restoration_path": CriticalPath(
                            kind=CriticalPathKind.SLO_RESTORATION,
                            intervals=(MeasuredInterval(name="slo", start_ns=0, end_ns=200),),
                        ),
                        "time_to_useful_reclaimed_capacity_ns": reclamation_end,
                        "time_to_serving_slo_restoration_ns": 200,
                    }
                )
            )
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        TraceMetricControl,
        evaluate_trace_controls,
    )

    trace = evaluate_trace_controls(
        control_id="trace-derived",
        seed=41,
        workload_digest="b" * 64,
        gpu_uuids=GPU_UUIDS,
        controls=(
            TraceMetricControl(
                metric="reclamation",
                unit="ns",
                disabled_value=100.0,
                minimal_value=100.0,
                full_value=120.0,
                materiality_threshold_fraction=0.05,
                disabled_provenance=_artifact("disabled"),
                minimal_provenance=_artifact("minimal"),
                full_provenance=_artifact("full"),
            ),
        ),
    )
    base = _evidence()
    provisional = base.model_copy(
        update={
            "evidence_class": EvidenceClass.HARDWARE_BACKED_REAL,
            "trials": tuple(trials),
            "trace_gate": trace,
            "software_optimization": base.software_optimization.model_copy(
                update={
                    "profile_hardware_backed": True,
                    "trace_overhead_gate_passed": True,
                }
            ),
        }
    )
    optimization = derive_software_optimization_evidence(provisional)
    return provisional.model_copy(update={"software_optimization": optimization})


def test_raw_derivation_uses_exact_modes_counts_bytes_and_amdahl_fractions() -> None:
    evidence = _derived_fixture()
    optimization = derive_software_optimization_evidence(evidence)
    outcome = derive_outcome_evidence(evidence)

    assert len(optimization.naive_trial_ids) == len(optimization.optimized_trial_ids) == 3
    assert optimization.measured_reclamation_improvement_fraction == pytest.approx(0.25)
    assert outcome.kill_trials == outcome.naive_trials == outcome.optimized_trials == 3
    assert {gate.chain.occurrence_count for gate in outcome.chain_gates} == {3}
    assert {gate.chain.logical_bytes for gate in outcome.chain_gates} == {300}
    assert all(
        gate.chain.state_passes > 0 and gate.chain.physical_bytes > 0
        for gate in outcome.chain_gates
    )
    assert all(
        gate.ideal_free_end_to_end_speedup
        == pytest.approx(1 / (1 - gate.fraction_of_full_transaction))
        for gate in outcome.chain_gates
    )


def test_caller_authored_optimization_and_outcome_values_are_detectable() -> None:
    evidence = _derived_fixture()
    derived = derive_outcome_evidence(evidence)
    forged_optimization = evidence.software_optimization.model_copy(
        update={"measured_reclamation_improvement_fraction": 0.99}
    )
    forged_outcome = derived.model_copy(update={"optimized_movement_fraction": 0.99})

    assert derive_software_optimization_evidence(evidence) != forged_optimization
    assert derive_outcome_evidence(evidence) != forged_outcome


def test_summary_projection_has_no_self_referential_provenance_fields() -> None:
    trial = _derived_fixture().trials[0]
    projection = trial.model_dump(mode="json", exclude={"provenance", "raw_provenance"})
    payload = json.dumps({"report_trial_summary": projection}, sort_keys=True).encode()

    assert "provenance" not in projection
    assert hashlib.sha256(payload).hexdigest()
