from __future__ import annotations

import hashlib

import pytest

from sloforge.continuum.planner import (
    AccessPattern,
    ContinuumMigrationExtension,
    ExactnessRequirement,
    MeasuredRate,
    MigrationPlanningInput,
    MigrationStrategy,
    fabric_transfer_rates,
    plan_migration,
    plan_with_fabric_and_warmpath,
    with_continuum_extension,
)
from sloforge.fabric.profiling import benchmark_synthetic_fabric
from sloforge.fabric.topology import build_discovery_fixture
from sloforge.ir import Extensions
from sloforge.warmpath import WarmPathPlan


def _rate(name: str, value: float) -> MeasuredRate:
    return MeasuredRate(
        name=name,
        bytes_per_second=value,
        sample_count=7,
        coefficient_of_variation=0.05,
        artifact_uri=f"artifacts/{name}.json",
        artifact_sha256=hashlib.sha256(name.encode()).hexdigest(),
        synthetic=True,
    )


def _request(**changes: object) -> MigrationPlanningInput:
    payload: dict[str, object] = {
        "seed": 41,
        "source_runtime": "reference-a",
        "destination_runtime": "reference-b",
        "state_size_bytes": 128 * 1024 * 1024,
        "dirty_rate_bytes_per_second": 2 * 1024 * 1024,
        "generation_tokens_per_second": 40.0,
        "source_load_fraction": 0.5,
        "destination_ready_ms": 10.0,
        "transfer_rates": (_rate("tcp", 128 * 1024 * 1024),),
        "conversion_rates": (
            _rate("canonical", 64 * 1024 * 1024),
            _rate("direct", 256 * 1024 * 1024),
        ),
        "memory_limit_bytes": 256 * 1024 * 1024,
        "maximum_interruption_ms": 100.0,
        "exactness": ExactnessRequirement.EXACT_SEMANTIC,
        "compatibility_allows_recomputation": True,
        "rollback_required": True,
        "failure_probability": 0.0,
        "migration_budget_usd": 1.0,
        "access_patterns": (
            AccessPattern(
                segment_id="kv",
                size_bytes=120 * 1024 * 1024,
                state_type="attention_kv",
                required_before_resume=True,
                streamable_before_use=False,
                recomputable=True,
                dense_full_attention=True,
            ),
            AccessPattern(
                segment_id="history",
                size_bytes=8 * 1024 * 1024,
                state_type="token_history",
                required_before_resume=True,
                streamable_before_use=True,
                recomputable=False,
            ),
        ),
    }
    payload.update(changes)
    return MigrationPlanningInput.model_validate(payload)


def test_low_dirty_rate_selects_measured_direct_precopy_and_reports_baselines() -> None:
    planned = plan_migration(_request())
    assert planned.strategy is MigrationStrategy.PRE_COPY
    assert planned.selected_conversion == "direct"
    assert planned.expected_interruption_ms < 100.0
    assert {candidate.strategy for candidate in planned.candidates} == set(MigrationStrategy)
    lazy = next(
        candidate
        for candidate in planned.candidates
        if candidate.strategy is MigrationStrategy.CONSTRAINED_LAZY
    )
    assert lazy.legal is False
    assert "dense_full_attention" in str(lazy.rejection_reason)


def test_high_dirty_rate_rejects_precopy_and_uses_hybrid_when_slo_allows() -> None:
    request = _request(
        dirty_rate_bytes_per_second=256 * 1024 * 1024,
        maximum_interruption_ms=2_000.0,
    )
    planned = plan_migration(request)
    pre_copy = next(
        candidate
        for candidate in planned.candidates
        if candidate.strategy is MigrationStrategy.PRE_COPY
    )
    assert pre_copy.legal is False
    assert "dirty rate" in str(pre_copy.rejection_reason)
    assert planned.strategy in {
        MigrationStrategy.STOP_AND_COPY,
        MigrationStrategy.HYBRID_PRE_COPY,
        MigrationStrategy.RECOMPUTATION_ASSISTED,
    }


def test_planner_requires_artifact_provenance_and_complete_segment_coverage() -> None:
    with pytest.raises(ValueError, match="cover"):
        _request(
            access_patterns=(
                AccessPattern(
                    segment_id="history",
                    size_bytes=1,
                    state_type="token_history",
                    required_before_resume=True,
                    streamable_before_use=True,
                    recomputable=False,
                ),
            )
        )
    with pytest.raises(ValueError, match="measured"):
        _request(transfer_rates=())


def test_fabric_and_warmpath_evidence_drive_joint_planning(tmp_path) -> None:
    profile = benchmark_synthetic_fabric(
        build_discovery_fixture("two_node_infiniband"),
        seed=53,
        suite="quick",
        sample_count=3,
        output_dir=tmp_path / "fabric",
    )
    rates = fabric_transfer_rates(profile, profile_uri="artifacts/fabric/profile")
    assert rates
    assert all(rate.synthetic for rate in rates)
    assert all(
        rate.artifact_sha256 in {item.artifact_hash for item in profile.results} for rate in rates
    )

    warm = WarmPathPlan.model_construct(
        plan_id="warm-continuum-destination",
        predicted_p95_ready_time_ms=27.0,
    )
    integrated = plan_with_fabric_and_warmpath(
        _request(maximum_interruption_ms=2_000.0),
        fabric_profile=profile,
        fabric_profile_uri="artifacts/fabric/profile",
        warmpath_plan=warm,
    )
    assert integrated.destination_ready_ms == 27.0
    assert integrated.fabric_profile_hash == profile.profile_hash
    assert integrated.plan.destination_warmup_required is True


def test_continuum_migration_references_extend_fabric_without_schema_fork() -> None:
    digest = hashlib.sha256(b"continuum-plan").hexdigest()
    migration = ContinuumMigrationExtension(
        capsule_uri="artifacts/continuum/capsules/session-1/capsule.json",
        compatibility_report_uri="artifacts/continuum/compatibility/session-1.json",
        conversion_plan_uri="artifacts/continuum/conversions/session-1.json",
        migration_plan_uri="artifacts/continuum/migrations/session-1.json",
        migration_plan_sha256=digest,
        source_topology_fingerprint=hashlib.sha256(b"source").hexdigest(),
        destination_topology_fingerprint=hashlib.sha256(b"destination").hexdigest(),
        segment_deadlines_ms={"attention-kv": 100.0, "recurrent": 50.0},
        required_before_use=("attention-kv", "recurrent"),
        conversion_placement_options=("source_cpu", "destination_cpu"),
    )
    extended = with_continuum_extension(Extensions(root={}), migration)
    assert extended.root["sloforge.ai/continuum-migration-v1"] == migration.model_dump(mode="json")
    with pytest.raises(ValueError, match="already carries"):
        with_continuum_extension(extended, migration)
