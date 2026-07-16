"""Evidence-preserving bridges from Fabric and WarmPath into Continuum planning."""

from __future__ import annotations

import math
import statistics
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from sloforge.fabric.profiling import BenchmarkStatus, MeasurementMode, Primitive
from sloforge.fabric.profiling.models import FabricProfile
from sloforge.ir import Extensions
from sloforge.warmpath import WarmPathPlan

from .core import MeasuredRate, MigrationPlanningInput, PlannedMigration, plan_migration

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ContinuumMigrationExtension(BaseModel):
    """A versioned reference carried by a Fabric physical plan extension."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["sloforge.continuum.fabric-migration/v1"] = (
        "sloforge.continuum.fabric-migration/v1"
    )
    capsule_uri: str
    compatibility_report_uri: str
    conversion_plan_uri: str
    migration_plan_uri: str
    migration_plan_sha256: Sha256
    source_topology_fingerprint: Sha256
    destination_topology_fingerprint: Sha256
    segment_deadlines_ms: dict[str, Annotated[float, Field(ge=0)]]
    required_before_use: tuple[str, ...]
    conversion_placement_options: tuple[
        Literal["source_cpu", "destination_cpu", "source_gpu", "destination_gpu"], ...
    ]


class IntegratedPlanningResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    plan: PlannedMigration
    fabric_profile_hash: Sha256
    warmpath_plan_id: str
    destination_ready_ms: Annotated[float, Field(ge=0)]


def _coefficient_of_variation(values: tuple[float, ...]) -> float:
    mean = statistics.fmean(values)
    if len(values) < 2 or mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def fabric_transfer_rates(
    profile: FabricProfile,
    *,
    profile_uri: str,
) -> tuple[MeasuredRate, ...]:
    """Extract usable byte-transfer rates while retaining raw Fabric provenance."""

    supported = {
        Primitive.KV_TRANSFER,
        Primitive.SEND_RECV,
        Primitive.HOST_MEMCPY,
        Primitive.GPU_P2P,
    }
    rates: list[MeasuredRate] = []
    for result in profile.results:
        if result.status is not BenchmarkStatus.SUCCESS or result.case.primitive not in supported:
            continue
        values = tuple(
            sample.throughput_bytes_per_second
            if sample.throughput_bytes_per_second is not None
            else result.case.message_bytes / max(sample.duration_microseconds / 1_000_000, 1e-12)
            for sample in result.raw_samples
        )
        positive = tuple(value for value in values if value > 0 and math.isfinite(value))
        if not positive:
            continue
        rates.append(
            MeasuredRate(
                name=f"fabric:{result.case.case_id}:{result.case.invocation.adapter}",
                bytes_per_second=statistics.median(positive),
                sample_count=len(positive),
                coefficient_of_variation=_coefficient_of_variation(positive),
                artifact_uri=(
                    f"{profile_uri.rstrip('/')}/{result.raw_artifact}"
                    if result.raw_artifact is not None
                    else profile_uri
                ),
                artifact_sha256=result.artifact_hash,
                synthetic=(
                    result.mode is not MeasurementMode.MEASURED
                    or any(sample.synthetic for sample in result.raw_samples)
                ),
            )
        )
    if not rates:
        raise ValueError("Fabric profile contains no successful byte-transfer measurements")
    return tuple(rates)


def plan_with_fabric_and_warmpath(
    request: MigrationPlanningInput,
    *,
    fabric_profile: FabricProfile,
    fabric_profile_uri: str,
    warmpath_plan: WarmPathPlan,
) -> IntegratedPlanningResult:
    """Jointly account for measured movement and destination readiness."""

    integrated = request.model_copy(
        update={
            "transfer_rates": fabric_transfer_rates(
                fabric_profile,
                profile_uri=fabric_profile_uri,
            ),
            "destination_ready_ms": warmpath_plan.predicted_p95_ready_time_ms,
        }
    )
    plan = plan_migration(MigrationPlanningInput.model_validate(integrated.model_dump()))
    return IntegratedPlanningResult(
        plan=plan,
        fabric_profile_hash=fabric_profile.profile_hash,
        warmpath_plan_id=warmpath_plan.plan_id,
        destination_ready_ms=warmpath_plan.predicted_p95_ready_time_ms,
    )


def with_continuum_extension(
    extensions: Extensions,
    migration: ContinuumMigrationExtension,
) -> Extensions:
    """Attach a JSON-only migration reference without changing Fabric's major schema."""

    updated = dict(extensions.root)
    key = "sloforge.ai/continuum-migration-v1"
    if key in updated:
        raise ValueError("physical plan already carries a Continuum migration extension")
    updated[key] = migration.model_dump(mode="json")
    return Extensions(root=updated)
