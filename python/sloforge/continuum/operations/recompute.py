"""Explicit, proof-gated token-history recomputation for the reference runtime."""

from __future__ import annotations

import json
from hashlib import sha256

from sloforge.continuum.adapters import CapturedState, SessionLifecycle, SnapshotHandle
from sloforge.continuum.ir import RecomputationPermission, canonical_hash
from sloforge.continuum.reference.codec import encode_state
from sloforge.continuum.reference.models import HybridDecoderState
from sloforge.continuum.reference.runtime import DeterministicHybridRuntimeAdapter
from sloforge.continuum.storage import ContentStore

from .checkpoint import verify_checkpoint_artifact
from .models import (
    AuthorizationError,
    CheckpointArtifact,
    RecomputeEvidence,
    RecomputeProofError,
    RecomputeResult,
)

_RECOMPUTED_COMPONENTS = ("state/attention-kv", "state/recurrent")


def _read_token_history(
    artifact: CheckpointArtifact, store: ContentStore, tenant_id: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    manifests = {manifest.segment_id: manifest for manifest in artifact.capsule.segment_manifests}
    manifest = manifests.get("logical:history")
    if manifest is None or len(manifest.chunks) != 1:
        raise RecomputeProofError("capsule lacks one authenticated token-history chunk")
    chunk = manifest.chunks[0]
    references = {reference.digest: reference for reference in artifact.chunk_references}
    reference = references.get(chunk.content_hash.value)
    if reference is None or reference.tenant_id != tenant_id:
        raise AuthorizationError("token-history chunk is outside the authorized tenant")
    payload = store.read(tenant_id, reference)
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecomputeProofError("token-history chunk is not canonical JSON") from error
    if json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode() != payload:
        raise RecomputeProofError("token-history chunk is not canonically encoded")
    logical = artifact.capsule.logical_state.token_history
    expected = {
        "committed_output_token_ids": list(logical.committed_output_token_ids),
        "input_token_ids": list(logical.input_token_ids),
        "uncommitted_speculative_token_ids": list(logical.uncommitted_speculative_tokens),
    }
    if decoded != expected:
        raise RecomputeProofError("token-history bytes disagree with logical state")
    if logical.uncommitted_speculative_tokens:
        raise RecomputeProofError("uncommitted speculative tokens cannot be replayed as committed")
    return tuple(logical.input_token_ids), tuple(logical.committed_output_token_ids)


def _prove_dependencies(artifact: CheckpointArtifact) -> tuple[str, ...]:
    capsule = artifact.capsule
    if RecomputationPermission.FROM_TOKEN_HISTORY not in (
        capsule.compatibility.recomputation_permissions
    ):
        raise RecomputeProofError("capsule forbids token-history recomputation")
    components = {
        component.semantic_id: component
        for component in capsule.logical_state.component_descriptors()
    }
    edges = {
        (edge.upstream_component_id, edge.downstream_component_id): edge
        for edge in capsule.logical_state.dependency_graph.edges
    }
    proof: list[str] = []
    for component_id in _RECOMPUTED_COMPONENTS:
        component = components.get(component_id)
        edge = edges.get(("state/token-history", component_id))
        if component is None or component.recomputation_permission is not (
            RecomputationPermission.FROM_TOKEN_HISTORY
        ):
            raise RecomputeProofError(f"{component_id} lacks component-level replay permission")
        if edge is None or not edge.invalidated_by_weight_change:
            raise RecomputeProofError(
                f"{component_id} lacks an explicit state-producing dependency edge"
            )
        proof.append(
            f"{edge.upstream_component_id}->{edge.downstream_component_id}:"
            f"{edge.dependency_semantics}"
        )
    return tuple(proof)


def _teacher_force(
    adapter: DeterministicHybridRuntimeAdapter,
    *,
    session_id: str,
    request_id: str,
    tenant_id: str,
    owner_epoch: int,
    seed: int,
    input_tokens: tuple[int, ...],
    output_tokens: tuple[int, ...],
    client_acknowledged_index: int,
) -> HybridDecoderState:
    config = adapter.config
    state = HybridDecoderState.create(
        config,
        session_id=session_id,
        request_id=request_id,
        tenant_id=tenant_id,
        seed=seed,
        owner_epoch=owner_epoch,
        input_token_ids=input_tokens,
    )
    for token_index, token_id in enumerate(output_tokens):
        if token_id >= config.model.vocabulary_size:
            raise AuthorizationError("recorded token is outside destination vocabulary")
        position = state.token_count
        state.state_version += 1
        state._append_model_state(config, token_id=token_id, position=position)
        state.token_dirty_epochs.append(state.state_version)
        state.output_token_ids.append(token_id)
        state.sampler_counter += 1
        state.guided_state = (state.guided_state + (token_id // 4) % 3 + 1) % 4
        state.gateway_committed_index = token_index
        state.state_version += 1
    if not -1 <= client_acknowledged_index <= state.gateway_committed_index:
        raise RecomputeProofError("client cursor cannot be reconstructed from token history")
    state.client_acknowledged_index = client_acknowledged_index
    state.lifecycle = SessionLifecycle.PAUSED
    return state


def _bounded_continuation(
    state: HybridDecoderState,
    adapter: DeterministicHybridRuntimeAdapter,
    horizon: int,
) -> tuple[int, ...]:
    candidate = state.clone()
    candidate.lifecycle = SessionLifecycle.ACTIVE
    return tuple(
        candidate.generate(adapter.config, transaction_id=None).token_id for _ in range(horizon)
    )


def recompute_from_token_history(
    artifact: CheckpointArtifact,
    *,
    store: ContentStore,
    destination: DeterministicHybridRuntimeAdapter,
    expected_tenant_id: str,
    seed: int,
    continuation_horizon: int = 4,
) -> RecomputeResult:
    """Rebuild model-derived state by explicit teacher forcing, never by hidden fallback."""

    if not 1 <= continuation_horizon <= 64:
        raise ValueError("continuation horizon must be in 1..64")
    if destination.identity.adapter_version not in {
        "continuum-adapter-a/1.0.0",
        "continuum-adapter-b/1.0.0",
    }:
        raise RecomputeProofError("teacher-forcing hook is not version-scoped to this adapter")
    verify_checkpoint_artifact(artifact)
    capsule = artifact.capsule
    if capsule.identity.tenant_id != expected_tenant_id:
        raise AuthorizationError("caller is not authorized for the capsule tenant")
    if capsule.identity.tokenizer_hash.value != destination.config.model.tokenizer_hash:
        raise AuthorizationError("tokenizer mismatch makes token-history replay unsafe")
    if capsule.logical_state.guided_decoding is None:
        raise RecomputeProofError("guided state is required for bounded continuation")
    if (
        capsule.logical_state.guided_decoding.automaton_identity.value
        != destination.config.automaton_hash
    ):
        raise RecomputeProofError("guided automaton changed across recomputation")
    if capsule.logical_state.sampler.rng_algorithm != "continuum-counter-v1":
        raise RecomputeProofError("sampler RNG is not the replayable reference algorithm")
    dependencies = _prove_dependencies(artifact)
    input_tokens, output_tokens = _read_token_history(artifact, store, expected_tenant_id)
    replay_seed = capsule.logical_state.sampler.seed
    state = _teacher_force(
        destination,
        session_id=capsule.identity.session_id,
        request_id=capsule.logical_state.execution.request_id,
        tenant_id=expected_tenant_id,
        owner_epoch=capsule.identity.owner_epoch,
        seed=replay_seed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        client_acknowledged_index=(
            capsule.logical_state.client_delivery.last_client_acknowledged_token_index
            if capsule.logical_state.client_delivery.last_client_acknowledged_token_index
            is not None
            else -1
        ),
    )
    if state.sampler_counter != capsule.logical_state.sampler.rng_counter:
        raise RecomputeProofError("sampler counter is not derivable from committed token history")
    if str(state.guided_state) != (capsule.logical_state.guided_decoding.current_automaton_state):
        raise RecomputeProofError("guided state is not derivable from committed token history")
    first = _bounded_continuation(state, destination, continuation_horizon)
    independently_replayed = _teacher_force(
        destination,
        session_id=capsule.identity.session_id,
        request_id=capsule.logical_state.execution.request_id,
        tenant_id=expected_tenant_id,
        owner_epoch=capsule.identity.owner_epoch,
        seed=replay_seed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        client_acknowledged_index=state.client_acknowledged_index,
    )
    second = _bounded_continuation(independently_replayed, destination, continuation_horizon)
    if first != second:
        raise RecomputeProofError("independent bounded continuation replay diverged")
    encoded = encode_state(state, destination.config)
    snapshot_id = sha256(
        f"continuum-recompute/v1\0{capsule.identity.capsule_id}\0"
        f"{destination.identity.build_hash}\0{seed}".encode()
    ).hexdigest()
    captured = CapturedState(
        handle=SnapshotHandle(
            snapshot_id=snapshot_id,
            session_id=encoded.logical.session_id,
            owner_epoch=encoded.logical.owner_epoch,
            state_version=encoded.logical.state_version,
            dirty_epoch=encoded.logical.dirty_epoch,
            segment_count=len(encoded.segments),
        ),
        runtime=destination.identity,
        layout=destination.config.layout,
        logical=encoded.logical,
        segments=encoded.segments,
        page_table=encoded.page_table,
    )
    captured.verify()
    evidence_payload = {
        "schema": "sloforge.continuum.recomputation-evidence/v1",
        "source_capsule_id": capsule.identity.capsule_id,
        "destination_model_hash": destination.config.model.model_hash,
        "recomputed_components": _RECOMPUTED_COMPONENTS,
        "dependency_edges": dependencies,
        "token_count": len(input_tokens) + len(output_tokens),
        "continuation_horizon": continuation_horizon,
        "first_run_tokens": first,
        "independent_run_tokens": second,
        "seed": seed,
    }
    evidence = RecomputeEvidence(
        source_capsule_id=capsule.identity.capsule_id,
        destination_model_hash=destination.config.model.model_hash,
        recomputed_components=_RECOMPUTED_COMPONENTS,
        dependency_edges=dependencies,
        token_count=len(input_tokens) + len(output_tokens),
        continuation_horizon=continuation_horizon,
        seed=seed,
        first_run_tokens=first,
        independent_run_tokens=second,
        verified=True,
        evidence_digest=canonical_hash(evidence_payload),
    )
    return RecomputeResult(captured=captured, evidence=evidence)
