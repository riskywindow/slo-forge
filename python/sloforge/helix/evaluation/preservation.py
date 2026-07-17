"""Executed multi-seed state-preservation evaluation for capacity reclamation.

The campaign intentionally uses the deterministic Continuum reference runtime and
the Helix environment/rollout reference implementations. Reference-accounting ticks
combine observed counts with declared weights; they are neither an instrumented
operation trace nor wall-clock latency measurements.
The source revision is a labeled deterministic fixture, not workspace Git provenance.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable
from contextlib import suppress
from enum import StrEnum
from hashlib import sha256
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.continuum.adapters import (
    ModelContract,
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
)
from sloforge.continuum.operations import (
    AuthorizationError,
    CapsuleRestoreError,
    CheckpointArtifact,
    OperationError,
    checkpoint_artifact_document,
    load_checkpoint_artifact,
    pause_and_checkpoint,
    resume_checkpoint,
)
from sloforge.continuum.storage import MemoryContentStore
from sloforge.continuum.transaction import (
    CutoverPhase,
    DurableCoordinator,
    GatewayCommitLedger,
    JournalEntry,
    SessionLease,
    TokenEvent,
)
from sloforge.helix.effects import EffectClass
from sloforge.helix.environments import (
    BranchNotFoundError,
    EnvironmentBackend,
    EnvironmentBranch,
    EnvironmentStateCapsule,
)
from sloforge.helix.environments.models import EntryKind, canonical_json, content_digest
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.rollouts import (
    ActionMutation,
    CandidateAction,
    ReferenceRolloutWorker,
    ReferenceTrajectory,
)

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")]
_STRATEGY_COUNT = 4
_STAMP = "2026-08-09T00:00:00Z"
_FIXTURE_SOURCE_REVISION = "b856a7191298ca1f477fdc8ebb3bb2093b5e8a304ad0796599d269d74f4ffdd8"
MAX_PRESERVATION_ARTIFACT_BYTES = 32 * 1024 * 1024


class PreservationError(RuntimeError):
    """Base error for an invalid or incomplete preservation experiment."""


class PreservationCompatibilityError(PreservationError):
    """A model checkpoint could not be reused under the destination contract."""


class PreservationStrategy(StrEnum):
    TERMINATE_RESTART = "terminate_restart"
    ENVIRONMENT_ONLY = "environment_only_checkpoint"
    MODEL_ONLY = "model_only_checkpoint"
    JOINT = "joint_continuum_environment"

    @property
    def preserves_model(self) -> bool:
        return self in {self.MODEL_ONLY, self.JOINT}

    @property
    def preserves_environment(self) -> bool:
        return self in {self.ENVIRONMENT_ONLY, self.JOINT}


_STRATEGIES = tuple(PreservationStrategy)


class _PreservationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class PreservationScenario(_PreservationModel):
    schema_version: Literal["sloforge.helix.preservation-scenario/v1"] = (
        "sloforge.helix.preservation-scenario/v1"
    )
    scenario_id: Identifier
    source_revision_kind: Literal["fixture"]
    source_revision: Digest
    tenant_id: Identifier
    input_token_ids: Annotated[tuple[int, ...], Field(min_length=1, max_length=64)]
    prefix_token_count: Annotated[int, Field(ge=1, le=64)]
    maximum_seeds: Annotated[int, Field(ge=2, le=16)]
    virtual_tick_ms: Annotated[int, Field(ge=1, le=10_000)]
    tick_basis: Literal["deterministic_reference_accounting_ticks"]
    tick_accounting_note: Literal[
        "fixed environment and dispatch ticks are declared accounting weights, not an instrumented operation trace"
    ]
    cost_basis: Literal["scenario_microunit_rates_applied_to_accounting_ticks_bytes_and_lost_work"]
    tick_cost_microunits: Annotated[int, Field(ge=0, le=1_000_000)]
    state_kib_cost_microunits: Annotated[int, Field(ge=0, le=1_000_000)]
    lost_token_cost_microunits: Annotated[int, Field(ge=0, le=1_000_000)]
    lost_tool_work_cost_microunits: Annotated[int, Field(ge=0, le=1_000_000)]
    prefix_tool_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    initial_tool_content: Annotated[str, StringConstraints(min_length=1, max_length=65_536)]
    preserved_tool_content: Annotated[str, StringConstraints(min_length=1, max_length=65_536)]
    serving_tool_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    initial_serving_content: Annotated[str, StringConstraints(min_length=1, max_length=65_536)]
    restored_serving_content: Annotated[str, StringConstraints(min_length=1, max_length=65_536)]
    strategies: Annotated[tuple[PreservationStrategy, ...], Field(min_length=4, max_length=4)]

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if self.strategies != _STRATEGIES:
            raise ValueError("preservation scenario must declare the canonical four-strategy order")
        if any(token < 0 or token > 255 for token in self.input_token_ids):
            raise ValueError("reference input token identifiers must fit the byte vocabulary")
        for value in (self.prefix_tool_path, self.serving_tool_path):
            parsed = PurePosixPath(value)
            if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != value:
                raise ValueError("preservation tool paths must be normalized and relative")
        if self.prefix_tool_path == self.serving_tool_path:
            raise ValueError("prefix and serving tool work require distinct paths")
        if self.source_revision != _FIXTURE_SOURCE_REVISION:
            raise ValueError("scenario must declare the deterministic fixture source revision")
        return self


class StateReuseEvidence(_PreservationModel):
    model_disposition: Literal["discarded_and_recomputed", "checkpoint_restored"]
    environment_disposition: Literal["discarded", "capsule_restored"]
    directly_reused_components: tuple[str, ...]
    recomputed_components: tuple[str, ...]
    discarded_components: tuple[str, ...]
    unsupported_components: tuple[str, ...] = ()
    verification_obligations: tuple[str, ...]

    @model_validator(mode="after")
    def disjoint_dispositions(self) -> Self:
        groups = (
            set(self.directly_reused_components),
            set(self.recomputed_components),
            set(self.discarded_components),
            set(self.unsupported_components),
        )
        for index, left in enumerate(groups):
            if any(left & right for right in groups[index + 1 :]):
                raise ValueError("state-reuse component dispositions overlap")
        if self.unsupported_components:
            raise ValueError("successful preservation evidence cannot contain unsupported state")
        return self


class ModelRestoreEvidence(_PreservationModel):
    restore_kind: Literal["restart_recompute", "continuum_checkpoint_resume"]
    source_runtime: str
    destination_runtime: str
    expected_model_hash: Digest
    destination_model_hash: Digest
    source_continuation_hash: Digest
    restored_continuation_hash: Digest
    checkpoint_capsule_id: Digest | None = None
    checkpoint_manifest_id: Digest | None = None
    checkpoint_watermark: int
    restored_watermark: int
    replayed_token_count: Annotated[int, Field(ge=0, le=64)]
    checkpoint_state_bytes: Annotated[int, Field(ge=0)]
    checkpoint_document: dict[str, object] | None = None
    resume_transaction_id: Digest | None = None
    resume_journal: Annotated[tuple[JournalEntry, ...], Field(max_length=64)] = ()
    continuation_validation_passed: bool
    expected_probe_token_id: Annotated[int, Field(ge=0, le=255)]
    observed_probe_token_id: Annotated[int, Field(ge=0, le=255)]

    @model_validator(mode="after")
    def validate_restore(self) -> Self:
        resumed = self.restore_kind == "continuum_checkpoint_resume"
        required = (
            self.checkpoint_capsule_id,
            self.checkpoint_manifest_id,
            self.checkpoint_document,
            self.resume_transaction_id,
        )
        if resumed != all(item is not None for item in required):
            raise ValueError("Continuum resume evidence requires complete checkpoint bindings")
        if resumed != bool(self.resume_journal):
            raise ValueError("only a Continuum resume may carry a transaction journal")
        if resumed != (self.replayed_token_count == 0):
            raise ValueError("restart must replay tokens while checkpoint resume must not")
        if resumed:
            assert self.checkpoint_document is not None
            checkpoint = _parse_checkpoint_document(self.checkpoint_document)
            if self.checkpoint_capsule_id != checkpoint.capsule.identity.capsule_id:
                raise ValueError("checkpoint capsule claim disagrees with checkpoint document")
            if self.checkpoint_manifest_id != checkpoint.store_manifest.manifest_id:
                raise ValueError("checkpoint manifest claim disagrees with checkpoint document")
            if self.checkpoint_watermark != checkpoint.capsule.transaction.commit_watermark:
                raise ValueError("checkpoint watermark claim disagrees with checkpoint document")
            if self.restored_watermark != self.checkpoint_watermark:
                raise ValueError("preserved-state resume changed the committed watermark")
            assert self.resume_transaction_id is not None
            if any(
                entry.transaction_id != self.resume_transaction_id for entry in self.resume_journal
            ):
                raise ValueError("resume journal contains a different transaction identifier")
            if tuple(entry.sequence for entry in self.resume_journal) != tuple(
                range(len(self.resume_journal))
            ):
                raise ValueError("resume journal sequence is not contiguous from zero")
            if (
                self.resume_journal[0].from_phase is not CutoverPhase.PROPOSED
                or self.resume_journal[0].to_phase is not CutoverPhase.PROPOSED
            ):
                raise ValueError("resume journal does not begin at the proposed phase")
            if any(
                left.to_phase is not right.from_phase
                for left, right in pairwise(self.resume_journal)
            ):
                raise ValueError("resume journal phase flow is not contiguous")
            if self.resume_journal[-1].to_phase is not CutoverPhase.COMPLETED:
                raise ValueError("resume journal does not terminate in COMPLETED")
            unsuccessful_terminal_phases = {
                CutoverPhase.REJECTED,
                CutoverPhase.ROLLED_BACK,
                CutoverPhase.FAILED_BEFORE_COMMIT,
                CutoverPhase.FAILED_AFTER_COMMIT,
                CutoverPhase.OPERATOR_REQUIRED,
            }
            if any(
                entry.to_phase in unsuccessful_terminal_phases
                or (
                    entry.to_phase is CutoverPhase.COMPLETED
                    and entry is not self.resume_journal[-1]
                )
                for entry in self.resume_journal
            ):
                raise ValueError("resume journal contains an invalid terminal flow")
        if self.expected_model_hash != self.destination_model_hash:
            raise ValueError("successful model restore evidence changed model identity")
        hashes_match = self.source_continuation_hash == self.restored_continuation_hash
        if self.continuation_validation_passed != hashes_match:
            raise ValueError("continuation validation claim disagrees with continuation hashes")
        if not self.continuation_validation_passed:
            raise ValueError("successful model restore requires continuation validation")
        if self.expected_probe_token_id != self.observed_probe_token_id:
            raise ValueError(
                "restored model continuation token differs from uninterrupted execution"
            )
        return self


class EnvironmentRestoreEvidence(_PreservationModel):
    restore_kind: Literal["base_restart", "environment_capsule_restore"]
    base_capsule_id: Digest
    selected_capsule_id: Digest
    restored_capsule_id: Digest
    final_serving_capsule_id: Digest
    selected_capsule_manifest: dict[str, object]
    expected_prefix_content_hash: Digest
    restored_prefix_content_hash: Digest
    prefix_tool_work_preserved: bool
    checkpoint_state_bytes: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_environment_restore(self) -> Self:
        preserved = self.restore_kind == "environment_capsule_restore"
        hashes_match = self.expected_prefix_content_hash == self.restored_prefix_content_hash
        if self.prefix_tool_work_preserved != hashes_match:
            raise ValueError("tool-work preservation claim disagrees with restored content hashes")
        if preserved != self.prefix_tool_work_preserved:
            raise ValueError("environment restore kind contradicts observed tool work")
        if preserved != (self.checkpoint_state_bytes > 0):
            raise ValueError("only preserved environment state may contribute checkpoint bytes")
        manifest = dict(self.selected_capsule_manifest)
        manifest_capsule_id = manifest.pop("capsule_id", None)
        if manifest_capsule_id != self.selected_capsule_id:
            raise ValueError("selected environment manifest names a different capsule")
        if content_digest(canonical_json(manifest)) != self.selected_capsule_id:
            raise ValueError("selected environment manifest content identity is invalid")
        return self


class RolloutRestoreEvidence(_PreservationModel):
    prefix_trajectory_id: Digest
    prefix_token_count: Annotated[int, Field(ge=1, le=256)]
    prefix_action_count: Annotated[int, Field(ge=1, le=128)]
    prefix_tool_work_units: Annotated[int, Field(ge=1, le=128)]
    prefix_trajectory: dict[str, object]
    post_restore_trajectory_id: Digest
    post_restore_event_count: Annotated[int, Field(ge=4, le=1024)]
    post_restore_action_count: Annotated[int, Field(ge=1, le=128)]
    post_restore_trajectory: dict[str, object]

    @model_validator(mode="after")
    def bind_trajectories(self) -> Self:
        if self.prefix_trajectory.get("trajectory_id") != self.prefix_trajectory_id:
            raise ValueError("prefix trajectory document is not bound to its identifier")
        if self.post_restore_trajectory.get("trajectory_id") != self.post_restore_trajectory_id:
            raise ValueError("post-restore trajectory document is not bound to its identifier")
        return self


class PreservationMetricInputs(_PreservationModel):
    virtual_tick_ms: Annotated[int, Field(ge=1, le=10_000)]
    environment_restore_validation_ticks: Literal[2]
    serving_probe_dispatch_ticks: Literal[1]
    tick_cost_microunits: Annotated[int, Field(ge=0, le=1_000_000)]
    state_kib_cost_microunits: Annotated[int, Field(ge=0, le=1_000_000)]
    lost_token_cost_microunits: Annotated[int, Field(ge=0, le=1_000_000)]
    lost_tool_work_cost_microunits: Annotated[int, Field(ge=0, le=1_000_000)]


class PreservationMetricSnapshot(_PreservationModel):
    lost_tokens: Annotated[int, Field(ge=0, le=64)]
    lost_tool_work_units: Annotated[int, Field(ge=0, le=128)]
    resume_ticks: Annotated[int, Field(ge=1)]
    resume_time_ms: Annotated[int, Field(ge=1)]
    serving_restoration_ticks: Annotated[int, Field(ge=1)]
    model_state_bytes: Annotated[int, Field(ge=0)]
    environment_state_bytes: Annotated[int, Field(ge=0)]
    total_state_bytes: Annotated[int, Field(ge=0)]
    cost_microunits: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_totals(self) -> Self:
        if self.total_state_bytes != self.model_state_bytes + self.environment_state_bytes:
            raise ValueError("preservation total state bytes disagree with component bytes")
        if self.serving_restoration_ticks <= self.resume_ticks:
            raise ValueError("serving restoration must follow state resume")
        return self


class PreservationRawEvidence(_PreservationModel):
    schema_version: Literal["sloforge.helix.preservation-evidence/v1"] = (
        "sloforge.helix.preservation-evidence/v1"
    )
    evidence_id: Digest
    scenario_digest: Digest
    source_revision_kind: Literal["fixture"]
    source_revision: Digest
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    strategy: PreservationStrategy
    state_reuse: StateReuseEvidence
    model_restore: ModelRestoreEvidence
    environment_restore: EnvironmentRestoreEvidence
    rollout_restore: RolloutRestoreEvidence
    metric_inputs: PreservationMetricInputs
    metrics: PreservationMetricSnapshot
    tick_basis: Literal["deterministic_reference_accounting_ticks"]
    tick_accounting_note: Literal[
        "fixed environment and dispatch ticks are declared accounting weights, not an instrumented operation trace"
    ]
    cost_basis: Literal["scenario_microunit_rates_applied_to_accounting_ticks_bytes_and_lost_work"]

    @model_validator(mode="after")
    def verify_identity(self) -> Self:
        if self.source_revision != _FIXTURE_SOURCE_REVISION:
            raise ValueError("raw evidence changed the deterministic fixture source revision")
        checkpoint_document = self.model_restore.checkpoint_document
        if checkpoint_document is not None:
            capsule = checkpoint_document.get("capsule")
            identity = capsule.get("identity") if isinstance(capsule, dict) else None
            checkpoint_revision = identity.get("git_commit") if isinstance(identity, dict) else None
            if checkpoint_revision != self.source_revision:
                raise ValueError("Continuum checkpoint revision differs from its fixture label")
        expected_metrics = _derive_metric_snapshot(
            strategy=self.strategy,
            model=self.model_restore,
            environment=self.environment_restore,
            rollout=self.rollout_restore,
            inputs=self.metric_inputs,
        )
        if self.metrics != expected_metrics:
            raise ValueError("raw preservation metrics disagree with restore evidence")
        if _digest(self.model_dump(mode="json", exclude={"evidence_id"})) != self.evidence_id:
            raise ValueError("preservation evidence identifier is invalid")
        return self


class StrategyObservation(_PreservationModel):
    observation_id: Digest
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    strategy: PreservationStrategy
    lost_tokens: Annotated[int, Field(ge=0, le=64)]
    lost_tool_work_units: Annotated[int, Field(ge=0, le=128)]
    resume_ticks: Annotated[int, Field(ge=1)]
    resume_time_ms: Annotated[int, Field(ge=1)]
    serving_restoration_ticks: Annotated[int, Field(ge=1)]
    model_state_bytes: Annotated[int, Field(ge=0)]
    environment_state_bytes: Annotated[int, Field(ge=0)]
    total_state_bytes: Annotated[int, Field(ge=0)]
    cost_microunits: Annotated[int, Field(ge=0)]
    evidence_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    evidence_sha256: Digest
    evidence_id: Digest

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.total_state_bytes != self.model_state_bytes + self.environment_state_bytes:
            raise ValueError("preservation total state bytes disagree with component bytes")
        if self.serving_restoration_ticks <= self.resume_ticks:
            raise ValueError("serving restoration must follow state resume")
        identity = self.model_dump(mode="json", exclude={"observation_id"})
        if _digest(identity) != self.observation_id:
            raise ValueError("preservation observation identifier is invalid")
        return self


def _derive_metric_snapshot(
    *,
    strategy: PreservationStrategy,
    model: ModelRestoreEvidence,
    environment: EnvironmentRestoreEvidence,
    rollout: RolloutRestoreEvidence,
    inputs: PreservationMetricInputs,
) -> PreservationMetricSnapshot:
    model_resumed = model.restore_kind == "continuum_checkpoint_resume"
    environment_resumed = environment.restore_kind == "environment_capsule_restore"
    if model_resumed != strategy.preserves_model:
        raise ValueError("model restore evidence contradicts the preservation strategy")
    if environment_resumed != strategy.preserves_environment:
        raise ValueError("environment restore evidence contradicts the preservation strategy")
    model_ticks = len(model.resume_journal) if model_resumed else 1 + model.replayed_token_count
    resume_ticks = model_ticks + inputs.environment_restore_validation_ticks
    serving_ticks = (
        resume_ticks + inputs.serving_probe_dispatch_ticks + rollout.post_restore_event_count
    )
    lost_tokens = 0 if model_resumed else model.replayed_token_count
    lost_tool_work = 0 if environment_resumed else rollout.prefix_tool_work_units
    model_bytes = model.checkpoint_state_bytes
    environment_bytes = environment.checkpoint_state_bytes
    total_bytes = model_bytes + environment_bytes
    cost = (
        serving_ticks * inputs.tick_cost_microunits
        + ((total_bytes + 1023) // 1024) * inputs.state_kib_cost_microunits
        + lost_tokens * inputs.lost_token_cost_microunits
        + lost_tool_work * inputs.lost_tool_work_cost_microunits
    )
    return PreservationMetricSnapshot(
        lost_tokens=lost_tokens,
        lost_tool_work_units=lost_tool_work,
        resume_ticks=resume_ticks,
        resume_time_ms=resume_ticks * inputs.virtual_tick_ms,
        serving_restoration_ticks=serving_ticks,
        model_state_bytes=model_bytes,
        environment_state_bytes=environment_bytes,
        total_state_bytes=total_bytes,
        cost_microunits=cost,
    )


class StrategyAggregate(_PreservationModel):
    strategy: PreservationStrategy
    sample_count: Annotated[int, Field(ge=2, le=16)]
    mean_lost_tokens: Annotated[float, Field(ge=0.0)]
    mean_lost_tool_work_units: Annotated[float, Field(ge=0.0)]
    mean_resume_ticks: Annotated[float, Field(ge=0.0)]
    mean_serving_restoration_ticks: Annotated[float, Field(ge=0.0)]
    mean_state_bytes: Annotated[float, Field(ge=0.0)]
    mean_cost_microunits: Annotated[float, Field(ge=0.0)]

    @model_validator(mode="after")
    def finite_means(self) -> Self:
        values = (
            self.mean_lost_tokens,
            self.mean_lost_tool_work_units,
            self.mean_resume_ticks,
            self.mean_serving_restoration_ticks,
            self.mean_state_bytes,
            self.mean_cost_microunits,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("preservation aggregate means must be finite")
        return self


class PreservationCampaign(_PreservationModel):
    schema_version: Literal["sloforge.helix.preservation-campaign/v1"] = (
        "sloforge.helix.preservation-campaign/v1"
    )
    campaign_id: Digest
    scenario_id: Identifier
    scenario_digest: Digest
    scenario_path: Literal["scenario.json"]
    scenario_sha256: Digest
    source_revision_kind: Literal["fixture"]
    source_revision: Digest
    seeds: Annotated[tuple[int, ...], Field(min_length=2, max_length=16)]
    observations: Annotated[tuple[StrategyObservation, ...], Field(min_length=8, max_length=64)]
    aggregates: Annotated[tuple[StrategyAggregate, ...], Field(min_length=4, max_length=4)]
    tick_basis: Literal["deterministic_reference_accounting_ticks"]
    tick_accounting_note: Literal[
        "fixed environment and dispatch ticks are declared accounting weights, not an instrumented operation trace"
    ]
    cost_basis: Literal["scenario_microunit_rates_applied_to_accounting_ticks_bytes_and_lost_work"]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_campaign(self) -> Self:
        if self.source_revision != _FIXTURE_SOURCE_REVISION:
            raise ValueError("campaign changed the deterministic fixture source revision")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("preservation campaign seeds must be unique")
        expected = tuple((seed, strategy) for seed in self.seeds for strategy in _STRATEGIES)
        observed = tuple((item.seed, item.strategy) for item in self.observations)
        if observed != expected:
            raise ValueError("preservation campaign lacks the canonical seed/strategy matrix")
        if tuple(item.strategy for item in self.aggregates) != _STRATEGIES:
            raise ValueError("preservation aggregates must use canonical strategy order")
        if self.aggregates != _aggregates(self.observations):
            raise ValueError("preservation aggregates disagree with raw observations")
        if not self.limitations:
            raise ValueError("preservation campaign must declare measurement limitations")
        if _digest(self.model_dump(mode="json", exclude={"campaign_id"})) != self.campaign_id:
            raise ValueError("preservation campaign identifier is invalid")
        return self


DestinationFactory = Callable[[ModelContract], ReferenceHeadMajorAdapter]


def load_preservation_scenario(path: Path) -> PreservationScenario:
    return PreservationScenario.model_validate_json(path.read_bytes(), strict=True)


def _scenario_metric_inputs(scenario: PreservationScenario) -> PreservationMetricInputs:
    return PreservationMetricInputs(
        virtual_tick_ms=scenario.virtual_tick_ms,
        environment_restore_validation_ticks=2,
        serving_probe_dispatch_ticks=1,
        tick_cost_microunits=scenario.tick_cost_microunits,
        state_kib_cost_microunits=scenario.state_kib_cost_microunits,
        lost_token_cost_microunits=scenario.lost_token_cost_microunits,
        lost_tool_work_cost_microunits=scenario.lost_tool_work_cost_microunits,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _bounded_campaign_artifact(root: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != relative:
        raise PreservationError("preservation artifact path is not normalized")
    cursor = root
    for part in parsed.parts:
        cursor /= part
        if cursor.is_symlink():
            raise PreservationError("preservation artifact path contains a symbolic link")
    resolved = root.joinpath(*parsed.parts).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise PreservationError("preservation artifact escapes the campaign output")
    if resolved.stat().st_size > MAX_PRESERVATION_ARTIFACT_BYTES:
        raise PreservationError("preservation artifact exceeds the byte limit")
    return resolved


def _parse_checkpoint_document(document: dict[str, object]) -> CheckpointArtifact:
    """Parse and verify a complete embedded descriptor with Continuum's bounded loader."""

    with tempfile.TemporaryDirectory(prefix="sloforge-preservation-checkpoint-") as temporary:
        descriptor = Path(temporary) / "checkpoint.json"
        descriptor.write_bytes(_canonical_bytes(document))
        try:
            return load_checkpoint_artifact(descriptor)
        except (OSError, OperationError, ValueError) as error:
            raise ValueError(
                f"checkpoint document failed complete Continuum verification: {error}"
            ) from error


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _write_model(path: Path, value: BaseModel) -> str:
    payload = _canonical_bytes(value.model_dump(mode="json")) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _lease(runtime: ReferenceTokenMajorAdapter, session_id: str) -> SessionLease:
    metadata = runtime.inspect_session(session_id)
    return SessionLease(
        session_id=session_id,
        owner_runtime=runtime.identity.runtime_name,
        owner_epoch=metadata.owner_epoch,
        fencing_token=metadata.owner_epoch,
        expiration_ms=120_000,
        coordinator_version=1,
        last_committed_state_version=metadata.state_version,
        last_committed_token_index=metadata.committed_output_index,
    )


def _gateway_event(event: Any) -> TokenEvent:
    return TokenEvent(
        session_id=event.session_id,
        owner_epoch=event.owner_epoch,
        token_index=event.token_index,
        token_id=event.token_id,
        state_commit_version=event.state_commit_version,
        transaction_id=event.transaction_id,
    )


def _commit_runtime_event(
    runtime: ReferenceTokenMajorAdapter | ReferenceHeadMajorAdapter,
    event: Any,
    *,
    gateway: GatewayCommitLedger | None = None,
) -> None:
    if gateway is not None:
        gateway.accept(_gateway_event(event))
    runtime.acknowledge_gateway(
        event.session_id,
        token_index=event.token_index,
        owner_epoch=event.owner_epoch,
    )


def _model_state_bytes(artifact: CheckpointArtifact | None) -> int:
    return (
        sum(reference.size_bytes for reference in artifact.chunk_references)
        if artifact is not None
        else 0
    )


def _environment_state_bytes(capsule: EnvironmentStateCapsule | None) -> int:
    if capsule is None:
        return 0
    unique_content = {
        entry.content_hash: entry.size_bytes
        for entry in capsule.files
        if entry.kind is EntryKind.FILE and entry.content_hash is not None
    }
    return len(canonical_json(capsule.to_dict())) + sum(unique_content.values())


def _prefix_candidates(scenario: PreservationScenario) -> tuple[CandidateAction, ...]:
    return (
        CandidateAction(
            action="record-prefix-work",
            tool_id="reference-workspace-edit",
            effect_class=EffectClass.IDEMPOTENT_WRITE,
            mutations=(
                ActionMutation(
                    path=scenario.prefix_tool_path,
                    content=scenario.preserved_tool_content,
                    expected_before_hash=content_digest(scenario.initial_tool_content.encode()),
                ),
            ),
        ),
        CandidateAction(
            action="skip-prefix-work",
            tool_id="reference-noop",
            effect_class=EffectClass.PURE,
        ),
    )


def _serving_candidates(scenario: PreservationScenario) -> tuple[CandidateAction, ...]:
    return (
        CandidateAction(
            action="serve-after-restore",
            tool_id="reference-serving-tool",
            effect_class=EffectClass.IDEMPOTENT_WRITE,
            mutations=(
                ActionMutation(
                    path=scenario.serving_tool_path,
                    content=scenario.restored_serving_content,
                    expected_before_hash=content_digest(scenario.initial_serving_content.encode()),
                ),
            ),
        ),
        CandidateAction(
            action="skip-serving-probe",
            tool_id="reference-noop",
            effect_class=EffectClass.PURE,
        ),
    )


def _run_rollout(
    *,
    worker: ReferenceRolloutWorker,
    branch: EnvironmentBranch,
    initial_capsule_id: str,
    policy: DeterministicPolicy,
    candidates: tuple[CandidateAction, ...],
    seed: int,
    phase: str,
    source_model_capsule_id: str,
    reuse_hash: str,
) -> ReferenceTrajectory:
    return worker.run(
        branch=branch,
        initial_environment_capsule_id=initial_capsule_id,
        branch_group_id=f"preservation-{seed}-{phase}",
        branch_point_id=f"preservation-{phase}",
        branch_point_hash=_digest({"seed": seed, "phase": phase}),
        source_model_capsule_id=source_model_capsule_id,
        state_reuse_report_hash=reuse_hash,
        policy=policy,
        observation=f"execute bounded {phase} work for preservation seed {seed}",
        candidates=candidates,
        seed=seed,
        forced_action=policy.actions[0],
    )


def _safe_cleanup(branch: EnvironmentBranch | None) -> None:
    if branch is None:
        return
    with suppress(BranchNotFoundError):
        branch.cleanup()


def _model_restore(
    *,
    scenario: PreservationScenario,
    strategy: PreservationStrategy,
    seed: int,
    session_id: str,
    source: ReferenceTokenMajorAdapter,
    store: MemoryContentStore,
    expected_probe: int,
    source_continuation_hash: str,
    destination_factory: DestinationFactory,
) -> tuple[ModelRestoreEvidence, int, CheckpointArtifact | None]:
    checkpoint: CheckpointArtifact | None = None
    source_runtime = source.identity.runtime_name
    source_model = source.config.model
    if strategy.preserves_model:
        logical_time = seed % 10_000
        paused = pause_and_checkpoint(
            source,
            session_id,
            store=store,
            lease=_lease(source, session_id),
            published_at_ms=logical_time + 1,
            capture_timestamp=_STAMP,
            git_commit=scenario.source_revision,
            continuum_version="0.1.0",
        )
        checkpoint = paused.checkpoint
        source.cancel_session(session_id)
        destination = destination_factory(source_model)
        try:
            with (
                DurableCoordinator(":memory:") as coordinator,
                GatewayCommitLedger(":memory:", max_token_records=128) as gateway,
            ):
                coordinator.create_lease(
                    session_id=session_id,
                    owner_runtime=source_runtime,
                    expiration_ms=120_000,
                    initial_token_index=scenario.prefix_token_count - 1,
                )
                gateway.register(
                    session_id=session_id,
                    owner_epoch=1,
                    next_token_index=scenario.prefix_token_count,
                )
                resumed = resume_checkpoint(
                    checkpoint,
                    store=store,
                    destination=destination,
                    source_release_confirmed=True,
                    expected_tenant_id=scenario.tenant_id,
                    expected_model=destination.config.model,
                    coordinator=coordinator,
                    gateway=gateway,
                    seed=seed,
                    now_ms=logical_time + 10,
                    timeout_ms=60_000,
                )
                restored_before_probe = destination.capture_consistent(session_id)
                probe = destination.generate_token(session_id)
                _commit_runtime_event(destination, probe, gateway=gateway)
                journal = coordinator.journal(resumed.transaction.transaction_id)
        except (AuthorizationError, CapsuleRestoreError) as error:
            raise PreservationCompatibilityError(
                f"{strategy.value} rejected incompatible model-state reuse: {error}"
            ) from error
        ticks = len(journal)
        return (
            ModelRestoreEvidence(
                restore_kind="continuum_checkpoint_resume",
                source_runtime=source_runtime,
                destination_runtime=destination.identity.runtime_name,
                expected_model_hash=source_model.model_hash,
                destination_model_hash=destination.config.model.model_hash,
                source_continuation_hash=source_continuation_hash,
                restored_continuation_hash=restored_before_probe.logical.continuation_hash,
                checkpoint_capsule_id=checkpoint.capsule.identity.capsule_id,
                checkpoint_manifest_id=checkpoint.store_manifest.manifest_id,
                checkpoint_watermark=checkpoint.capsule.transaction.commit_watermark,
                restored_watermark=(
                    restored_before_probe.logical.client_delivery.last_gateway_committed_token_index
                ),
                replayed_token_count=0,
                checkpoint_state_bytes=_model_state_bytes(checkpoint),
                checkpoint_document=checkpoint_artifact_document(checkpoint),
                resume_transaction_id=resumed.transaction.transaction_id,
                resume_journal=journal,
                continuation_validation_passed=(
                    resumed.validation.structurally_valid
                    and resumed.validation.continuation_valid
                    and restored_before_probe.logical.continuation_hash == source_continuation_hash
                ),
                expected_probe_token_id=expected_probe,
                observed_probe_token_id=probe.token_id,
            ),
            ticks,
            checkpoint,
        )

    source.cancel_session(session_id)
    restarted = ReferenceTokenMajorAdapter(model=source_model)
    restarted.create_session(
        session_id=session_id,
        request_id=f"preservation-restart-{seed}",
        tenant_id=scenario.tenant_id,
        input_token_ids=scenario.input_token_ids,
        seed=seed,
    )
    with GatewayCommitLedger(":memory:", max_token_records=128) as gateway:
        gateway.register(session_id=session_id, owner_epoch=1)
        replayed = restarted.stream_tokens(session_id, count=scenario.prefix_token_count)
        for event in replayed:
            _commit_runtime_event(restarted, event, gateway=gateway)
        restored_before_probe = restarted.capture_consistent(session_id)
        probe = restarted.generate_token(session_id)
        _commit_runtime_event(restarted, probe, gateway=gateway)
    if restored_before_probe.logical.continuation_hash != source_continuation_hash:
        raise PreservationCompatibilityError(
            "deterministic restart failed to reconstruct model state"
        )
    return (
        ModelRestoreEvidence(
            restore_kind="restart_recompute",
            source_runtime=source_runtime,
            destination_runtime=restarted.identity.runtime_name,
            expected_model_hash=source_model.model_hash,
            destination_model_hash=restarted.config.model.model_hash,
            source_continuation_hash=source_continuation_hash,
            restored_continuation_hash=restored_before_probe.logical.continuation_hash,
            checkpoint_watermark=scenario.prefix_token_count - 1,
            restored_watermark=(
                restored_before_probe.logical.client_delivery.last_gateway_committed_token_index
            ),
            replayed_token_count=len(replayed),
            checkpoint_state_bytes=0,
            continuation_validation_passed=True,
            expected_probe_token_id=expected_probe,
            observed_probe_token_id=probe.token_id,
        ),
        1 + len(replayed),
        None,
    )


def _reuse_evidence(
    strategy: PreservationStrategy, artifact: CheckpointArtifact | None
) -> StateReuseEvidence:
    model_components = (
        tuple(
            descriptor.semantic_id
            for descriptor in artifact.capsule.logical_state.component_descriptors()
        )
        if artifact is not None
        else ()
    )
    directly_reused = (
        (*model_components, "environment/filesystem", "environment/tool-work")
        if strategy is PreservationStrategy.JOINT
        else model_components
        if strategy is PreservationStrategy.MODEL_ONLY
        else ("environment/filesystem", "environment/tool-work")
        if strategy is PreservationStrategy.ENVIRONMENT_ONLY
        else ()
    )
    recomputed = () if strategy.preserves_model else ("model/token-history",)
    discarded = (
        ()
        if strategy.preserves_environment
        else (
            "environment/filesystem",
            "environment/tool-work",
        )
    )
    obligations = ["restored continuation token equals uninterrupted reference token"]
    if strategy.preserves_model:
        obligations.append("Continuum transaction reaches COMPLETED under exact model identity")
    if strategy.preserves_environment:
        obligations.append("restored prefix tool bytes equal the checkpointed bytes")
    return StateReuseEvidence(
        model_disposition=(
            "checkpoint_restored" if strategy.preserves_model else "discarded_and_recomputed"
        ),
        environment_disposition=(
            "capsule_restored" if strategy.preserves_environment else "discarded"
        ),
        directly_reused_components=tuple(directly_reused),
        recomputed_components=recomputed,
        discarded_components=discarded,
        verification_obligations=tuple(obligations),
    )


def _execute_strategy(
    *,
    root: Path,
    raw_root: Path,
    scenario: PreservationScenario,
    scenario_digest: str,
    strategy: PreservationStrategy,
    seed: int,
    destination_factory: DestinationFactory,
) -> StrategyObservation:
    session_id = f"preservation-{seed}-{strategy.value}"
    workspace = root / "source"
    workspace.mkdir(parents=True)
    prefix_path = workspace.joinpath(*PurePosixPath(scenario.prefix_tool_path).parts)
    serving_path = workspace.joinpath(*PurePosixPath(scenario.serving_tool_path).parts)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    serving_path.parent.mkdir(parents=True, exist_ok=True)
    prefix_path.write_text(scenario.initial_tool_content)
    serving_path.write_text(scenario.initial_serving_content)

    backend = EnvironmentBackend(root / "environment-store", tenant_id=scenario.tenant_id)
    base = backend.capture(
        workspace,
        seed=seed,
        allowed_tools=("reference-serving-tool", "reference-workspace-edit"),
    )
    prefix_branch: EnvironmentBranch | None = backend.fork(
        base, branch_id=f"prefix-{seed}-{strategy.value}", seed=seed
    )
    restored_branch: EnvironmentBranch | None = None
    source = ReferenceTokenMajorAdapter()
    source.create_session(
        session_id=session_id,
        request_id=f"preservation-{strategy.value}-{seed}",
        tenant_id=scenario.tenant_id,
        input_token_ids=scenario.input_token_ids,
        seed=seed,
    )
    for event in source.stream_tokens(session_id, count=scenario.prefix_token_count):
        _commit_runtime_event(source, event)
    source_capture = source.capture_consistent(session_id)
    source_continuation_hash = source_capture.logical.continuation_hash
    expected_probe = source.dry_run_next_token(session_id)
    policy = DeterministicPolicy(
        policy_epoch_id=f"preservation-policy-{seed}",
        actions=("record-prefix-work", "skip-prefix-work"),
        logits=(0.0, 0.0),
    )
    worker = ReferenceRolloutWorker(tenant_id=scenario.tenant_id, max_events=64)
    live_model_id = _digest(
        {"runtime": source.identity.runtime_name, "continuation": source_continuation_hash}
    )
    prefix_reuse_hash = _digest({"strategy": strategy.value, "phase": "prefix", "seed": seed})
    try:
        assert prefix_branch is not None
        prefix_trajectory = _run_rollout(
            worker=worker,
            branch=prefix_branch,
            initial_capsule_id=base.capsule_id,
            policy=policy,
            candidates=_prefix_candidates(scenario),
            seed=seed,
            phase="prefix",
            source_model_capsule_id=live_model_id,
            reuse_hash=prefix_reuse_hash,
        )
        checkpointed_environment = (
            prefix_branch.checkpoint() if strategy.preserves_environment else None
        )
        store = MemoryContentStore()
        model_restore, executed_model_ticks, checkpoint = _model_restore(
            scenario=scenario,
            strategy=strategy,
            seed=seed,
            session_id=session_id,
            source=source,
            store=store,
            expected_probe=expected_probe,
            source_continuation_hash=source_continuation_hash,
            destination_factory=destination_factory,
        )

        _safe_cleanup(prefix_branch)
        prefix_branch = None
        selected = checkpointed_environment or base
        restored_branch = backend.fork(
            selected,
            branch_id=f"restored-{seed}-{strategy.value}",
            seed=seed,
        )
        observed_prefix = restored_branch.read_text(scenario.prefix_tool_path)
        expected_prefix_hash = content_digest(scenario.preserved_tool_content.encode())
        observed_prefix_hash = content_digest(observed_prefix.encode())
        tool_preserved = observed_prefix_hash == expected_prefix_hash
        if tool_preserved != strategy.preserves_environment:
            raise PreservationError("restored environment contradicted the preservation strategy")
        restored_validation = restored_branch.checkpoint()
        reuse = _reuse_evidence(strategy, checkpoint)
        reuse_hash = _digest(reuse.model_dump(mode="json"))
        serving_policy = DeterministicPolicy(
            policy_epoch_id=f"preservation-serving-policy-{seed}",
            actions=("serve-after-restore", "skip-serving-probe"),
            logits=(0.0, 0.0),
        )
        post_trajectory = _run_rollout(
            worker=worker,
            branch=restored_branch,
            initial_capsule_id=selected.capsule_id,
            policy=serving_policy,
            candidates=_serving_candidates(scenario),
            seed=seed,
            phase="post-restore",
            source_model_capsule_id=(
                checkpoint.capsule.identity.capsule_id if checkpoint is not None else live_model_id
            ),
            reuse_hash=reuse_hash,
        )
        final_environment = backend.load_capsule(post_trajectory.final_environment_capsule_id)
        environment_bytes = _environment_state_bytes(checkpointed_environment)
        environment_restore = EnvironmentRestoreEvidence(
            restore_kind=(
                "environment_capsule_restore" if strategy.preserves_environment else "base_restart"
            ),
            base_capsule_id=base.capsule_id,
            selected_capsule_id=selected.capsule_id,
            restored_capsule_id=restored_validation.capsule_id,
            final_serving_capsule_id=final_environment.capsule_id,
            selected_capsule_manifest=selected.to_dict(),
            expected_prefix_content_hash=expected_prefix_hash,
            restored_prefix_content_hash=observed_prefix_hash,
            prefix_tool_work_preserved=tool_preserved,
            checkpoint_state_bytes=environment_bytes,
        )
        rollout_restore = RolloutRestoreEvidence(
            prefix_trajectory_id=prefix_trajectory.trajectory_id,
            prefix_token_count=len(prefix_trajectory.tokens),
            prefix_action_count=len(prefix_trajectory.actions),
            prefix_tool_work_units=sum(
                len(action.mutations) for action in _prefix_candidates(scenario)
            ),
            prefix_trajectory=prefix_trajectory.model_dump(mode="json"),
            post_restore_trajectory_id=post_trajectory.trajectory_id,
            post_restore_event_count=len(post_trajectory.events),
            post_restore_action_count=len(post_trajectory.actions),
            post_restore_trajectory=post_trajectory.model_dump(mode="json"),
        )
        metric_inputs = _scenario_metric_inputs(scenario)
        metrics = _derive_metric_snapshot(
            strategy=strategy,
            model=model_restore,
            environment=environment_restore,
            rollout=rollout_restore,
            inputs=metric_inputs,
        )
        if metrics.resume_ticks != (
            executed_model_ticks + metric_inputs.environment_restore_validation_ticks
        ):
            raise PreservationError("executed model ticks disagree with raw restore evidence")
        raw_body: dict[str, object] = {
            "schema_version": "sloforge.helix.preservation-evidence/v1",
            "scenario_digest": scenario_digest,
            "source_revision_kind": scenario.source_revision_kind,
            "source_revision": scenario.source_revision,
            "seed": seed,
            "strategy": strategy,
            "state_reuse": reuse,
            "model_restore": model_restore,
            "environment_restore": environment_restore,
            "rollout_restore": rollout_restore,
            "metric_inputs": metric_inputs,
            "metrics": metrics,
            "tick_basis": "deterministic_reference_accounting_ticks",
            "tick_accounting_note": scenario.tick_accounting_note,
            "cost_basis": (
                "scenario_microunit_rates_applied_to_accounting_ticks_bytes_and_lost_work"
            ),
        }
        raw = PreservationRawEvidence.model_validate(
            {
                "evidence_id": _digest(
                    {
                        key: (
                            value.model_dump(mode="json") if isinstance(value, BaseModel) else value
                        )
                        for key, value in raw_body.items()
                    }
                ),
                **raw_body,
            },
            strict=True,
        )
        evidence_path = Path("raw") / f"seed-{seed}" / f"{strategy.value}.json"
        evidence_sha256 = _write_model(raw_root / evidence_path, raw)
        observation_body = {
            "seed": seed,
            "strategy": strategy,
            **metrics.model_dump(mode="json"),
            "evidence_path": evidence_path.as_posix(),
            "evidence_sha256": evidence_sha256,
            "evidence_id": raw.evidence_id,
        }
        return StrategyObservation.model_validate(
            {
                "observation_id": _digest(
                    {
                        key: value.value if isinstance(value, PreservationStrategy) else value
                        for key, value in observation_body.items()
                    }
                ),
                **observation_body,
            },
            strict=True,
        )
    finally:
        _safe_cleanup(restored_branch)
        _safe_cleanup(prefix_branch)


def _mean(values: tuple[int, ...]) -> float:
    return sum(values) / len(values)


def _aggregates(observations: tuple[StrategyObservation, ...]) -> tuple[StrategyAggregate, ...]:
    results: list[StrategyAggregate] = []
    for strategy in _STRATEGIES:
        selected = tuple(item for item in observations if item.strategy is strategy)
        if not selected:
            continue
        results.append(
            StrategyAggregate(
                strategy=strategy,
                sample_count=len(selected),
                mean_lost_tokens=_mean(tuple(item.lost_tokens for item in selected)),
                mean_lost_tool_work_units=_mean(
                    tuple(item.lost_tool_work_units for item in selected)
                ),
                mean_resume_ticks=_mean(tuple(item.resume_ticks for item in selected)),
                mean_serving_restoration_ticks=_mean(
                    tuple(item.serving_restoration_ticks for item in selected)
                ),
                mean_state_bytes=_mean(tuple(item.total_state_bytes for item in selected)),
                mean_cost_microunits=_mean(tuple(item.cost_microunits for item in selected)),
            )
        )
    return tuple(results)


def _default_destination(model: ModelContract) -> ReferenceHeadMajorAdapter:
    return ReferenceHeadMajorAdapter(model=model)


def run_preservation_campaign(
    output: Path,
    *,
    scenario_path: Path,
    seeds: tuple[int, ...],
    destination_factory: DestinationFactory = _default_destination,
) -> PreservationCampaign:
    """Execute and atomically publish the bounded four-strategy preservation matrix."""

    scenario = load_preservation_scenario(scenario_path)
    if not 2 <= len(seeds) <= scenario.maximum_seeds or len(set(seeds)) != len(seeds):
        raise ValueError("preservation campaign requires two to maximum_seeds unique seeds")
    if any(seed < 0 or seed > 2**63 - 1 for seed in seeds):
        raise ValueError("preservation seeds must fit signed 64-bit integers")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("preservation output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    scenario_digest = _digest(scenario.model_dump(mode="json"))
    with tempfile.TemporaryDirectory(prefix=".preservation-stage-", dir=output) as stage_name:
        stage = Path(stage_name)
        observations: list[StrategyObservation] = []
        with tempfile.TemporaryDirectory(prefix="work-", dir=stage) as work_name:
            work = Path(work_name)
            for seed in seeds:
                for strategy in _STRATEGIES:
                    observations.append(
                        _execute_strategy(
                            root=work / f"seed-{seed}" / strategy.value,
                            raw_root=stage,
                            scenario=scenario,
                            scenario_digest=scenario_digest,
                            strategy=strategy,
                            seed=seed,
                            destination_factory=destination_factory,
                        )
                    )
        typed_observations = tuple(observations)
        sealed_scenario_path = Path("scenario.json")
        sealed_scenario_sha256 = _write_model(stage / sealed_scenario_path, scenario)
        body = {
            "schema_version": "sloforge.helix.preservation-campaign/v1",
            "scenario_id": scenario.scenario_id,
            "scenario_digest": scenario_digest,
            "scenario_path": sealed_scenario_path.as_posix(),
            "scenario_sha256": sealed_scenario_sha256,
            "source_revision_kind": scenario.source_revision_kind,
            "source_revision": scenario.source_revision,
            "seeds": seeds,
            "observations": typed_observations,
            "aggregates": _aggregates(typed_observations),
            "tick_basis": "deterministic_reference_accounting_ticks",
            "tick_accounting_note": scenario.tick_accounting_note,
            "cost_basis": (
                "scenario_microunit_rates_applied_to_accounting_ticks_bytes_and_lost_work"
            ),
            "limitations": (
                "reference-accounting ticks are deterministic bookkeeping, not wall time",
                "fixed environment and dispatch ticks are declared accounting weights, not an instrumented operation trace",
                "costs apply scenario rates to accounting ticks and are not cloud bills",
                "the experiment covers local reference runtimes and filesystem-backed environments",
                "source_revision is a deterministic fixture identifier, not workspace Git HEAD",
            ),
        }
        campaign = PreservationCampaign.model_validate(
            {
                "campaign_id": _digest(
                    {
                        key: (
                            [item.model_dump(mode="json") for item in value]
                            if isinstance(value, tuple)
                            and value
                            and isinstance(value[0], BaseModel)
                            else value
                        )
                        for key, value in body.items()
                    }
                ),
                **body,
            },
            strict=True,
        )
        _write_model(stage / "campaign.json", campaign)
        for observation in campaign.observations:
            evidence = stage / observation.evidence_path
            if sha256(evidence.read_bytes()).hexdigest() != observation.evidence_sha256:
                raise PreservationError("raw preservation evidence changed before publication")
            loaded = PreservationRawEvidence.model_validate_json(evidence.read_bytes(), strict=True)
            if loaded.evidence_id != observation.evidence_id:
                raise PreservationError("observation points at different raw preservation evidence")
        raw = stage / "raw"
        raw.replace(output / "raw")
        (stage / sealed_scenario_path).replace(output / sealed_scenario_path)
        (stage / "campaign.json").replace(output / "campaign.json")
        return campaign


def validate_preservation_campaign(
    output: Path,
    *,
    scenario_path: Path | None = None,
) -> PreservationCampaign:
    """Reopen a campaign and validate its scenario and raw restore evidence."""

    root = output.resolve(strict=True)
    campaign_path = _bounded_campaign_artifact(root, "campaign.json")
    campaign = PreservationCampaign.model_validate_json(campaign_path.read_bytes(), strict=True)
    sealed_scenario_path = _bounded_campaign_artifact(root, campaign.scenario_path)
    sealed_scenario_payload = sealed_scenario_path.read_bytes()
    if sha256(sealed_scenario_payload).hexdigest() != campaign.scenario_sha256:
        raise PreservationError("sealed preservation scenario digest mismatch")
    scenario = PreservationScenario.model_validate_json(sealed_scenario_payload, strict=True)
    if (
        campaign.scenario_id != scenario.scenario_id
        or campaign.scenario_digest != _digest(scenario.model_dump(mode="json"))
        or campaign.source_revision_kind != scenario.source_revision_kind
        or campaign.source_revision != scenario.source_revision
        or campaign.tick_basis != scenario.tick_basis
        or campaign.tick_accounting_note != scenario.tick_accounting_note
        or campaign.cost_basis != scenario.cost_basis
    ):
        raise PreservationError("published campaign scenario digest does not match")
    if scenario_path is not None and load_preservation_scenario(scenario_path) != scenario:
        raise PreservationError("external scenario differs from the sealed campaign scenario")

    seen: set[str] = set()
    for observation in campaign.observations:
        if observation.evidence_path in seen:
            raise PreservationError("preservation campaign reused a raw evidence path")
        seen.add(observation.evidence_path)
        parsed = PurePosixPath(observation.evidence_path)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or str(parsed) != observation.evidence_path
            or not parsed.parts
            or parsed.parts[0] != "raw"
        ):
            raise PreservationError("preservation evidence path is not a normalized raw path")
        evidence_path = _bounded_campaign_artifact(root, observation.evidence_path)
        payload = evidence_path.read_bytes()
        if sha256(payload).hexdigest() != observation.evidence_sha256:
            raise PreservationError("raw preservation evidence digest mismatch")
        evidence = PreservationRawEvidence.model_validate_json(payload, strict=True)
        if evidence.evidence_id != observation.evidence_id:
            raise PreservationError("observation points at different raw preservation evidence")
        if (
            evidence.scenario_digest != campaign.scenario_digest
            or evidence.source_revision_kind != campaign.source_revision_kind
            or evidence.source_revision != campaign.source_revision
            or evidence.seed != observation.seed
            or evidence.strategy is not observation.strategy
            or evidence.tick_basis != campaign.tick_basis
            or evidence.tick_accounting_note != campaign.tick_accounting_note
            or evidence.cost_basis != campaign.cost_basis
        ):
            raise PreservationError("raw evidence does not belong to its campaign observation")
        if evidence.metric_inputs != _scenario_metric_inputs(scenario):
            raise PreservationError("raw metric inputs do not match the preservation scenario")
        observation_metrics = PreservationMetricSnapshot(
            lost_tokens=observation.lost_tokens,
            lost_tool_work_units=observation.lost_tool_work_units,
            resume_ticks=observation.resume_ticks,
            resume_time_ms=observation.resume_time_ms,
            serving_restoration_ticks=observation.serving_restoration_ticks,
            model_state_bytes=observation.model_state_bytes,
            environment_state_bytes=observation.environment_state_bytes,
            total_state_bytes=observation.total_state_bytes,
            cost_microunits=observation.cost_microunits,
        )
        if evidence.metrics != observation_metrics:
            raise PreservationError("campaign metrics disagree with recomputed raw evidence")
        ReferenceTrajectory.model_validate_json(
            _canonical_bytes(evidence.rollout_restore.prefix_trajectory), strict=True
        )
        ReferenceTrajectory.model_validate_json(
            _canonical_bytes(evidence.rollout_restore.post_restore_trajectory), strict=True
        )
    expected_files = {"campaign.json", campaign.scenario_path, *seen}
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PreservationError("preservation campaign contains a symbolic link")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise PreservationError("preservation campaign artifact inventory is incomplete")
    return campaign


__all__ = [
    "MAX_PRESERVATION_ARTIFACT_BYTES",
    "EnvironmentRestoreEvidence",
    "ModelRestoreEvidence",
    "PreservationCampaign",
    "PreservationCompatibilityError",
    "PreservationError",
    "PreservationMetricInputs",
    "PreservationMetricSnapshot",
    "PreservationRawEvidence",
    "PreservationScenario",
    "PreservationStrategy",
    "RolloutRestoreEvidence",
    "StateReuseEvidence",
    "StrategyAggregate",
    "StrategyObservation",
    "load_preservation_scenario",
    "run_preservation_campaign",
    "validate_preservation_campaign",
]
