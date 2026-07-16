"""Deterministic HybridDecoder state used by the Continuum CPU reference runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from sloforge.continuum.adapters.sdk import (
    ClientDeliverySnapshot,
    ClientTerminalStatus,
    GuidedDecodingSnapshot,
    LogicalStateManifest,
    ModelContract,
    RuntimeLayout,
    SamplerSnapshot,
    SessionLifecycle,
    TokenEvent,
)

_MODULUS = 2_147_483_647
_MASK_64 = (1 << 64) - 1


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _default_model_contract() -> ModelContract:
    return ModelContract(
        model_id="continuum/hybrid-decoder-v1",
        model_hash=_digest("continuum-hybrid-decoder-v1/weights"),
        tokenizer_hash=_digest("continuum-byte-tokenizer-v1"),
        adapter_hash=_digest("continuum-no-adapter"),
        state_producer_hash=_digest("continuum-hybrid-state-producer-v1"),
        recurrent_update_hash=_digest("continuum-recurrent-equation-v1"),
        positional_encoding_hash=_digest("continuum-absolute-position-v1"),
        vocabulary_size=256,
    )


@dataclass(frozen=True, slots=True)
class HybridDecoderConfig:
    layout: RuntimeLayout
    layers: int = 2
    kv_heads: int = 4
    head_dimension: int = 4
    recurrent_width: int = 6
    max_context_tokens: int = 4096
    model: ModelContract = field(default_factory=_default_model_contract)
    automaton_id: str = "continuum/mod4-guidance-v1"
    max_dirty_events: int = 128

    def __post_init__(self) -> None:
        for name, value in (
            ("layers", self.layers),
            ("kv_heads", self.kv_heads),
            ("head_dimension", self.head_dimension),
            ("recurrent_width", self.recurrent_width),
            ("max_context_tokens", self.max_context_tokens),
            ("max_dirty_events", self.max_dirty_events),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.kv_heads % self.layout.tensor_parallel_degree:
            raise ValueError("kv_heads must divide evenly across tensor parallel ranks")
        if self.max_context_tokens < self.layout.page_size_tokens:
            raise ValueError("max_context_tokens must fit at least one page")

    @property
    def automaton_hash(self) -> str:
        return _digest(self.automaton_id)


@dataclass(slots=True)
class HybridDecoderState:
    session_id: str
    request_id: str
    tenant_id: str
    seed: int
    owner_epoch: int
    input_token_ids: list[int]
    output_token_ids: list[int]
    attention_keys: list[list[list[list[int]]]]
    attention_values: list[list[list[list[int]]]]
    token_dirty_epochs: list[int]
    recurrent_state: list[list[int]]
    sampler_counter: int
    guided_state: int
    gateway_committed_index: int
    client_acknowledged_index: int
    state_version: int
    lifecycle: SessionLifecycle
    transaction_id: str | None = None

    @classmethod
    def create(
        cls,
        config: HybridDecoderConfig,
        *,
        session_id: str,
        request_id: str,
        tenant_id: str,
        seed: int,
        owner_epoch: int,
        input_token_ids: tuple[int, ...],
    ) -> HybridDecoderState:
        if owner_epoch <= 0:
            raise ValueError("owner_epoch must be positive")
        if not 0 <= seed < 1 << 64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if len(input_token_ids) > config.max_context_tokens:
            raise ValueError("input token history exceeds configured context")
        if any(token < 0 or token >= config.model.vocabulary_size for token in input_token_ids):
            raise ValueError("input token is outside the configured vocabulary")
        state = cls(
            session_id=session_id,
            request_id=request_id,
            tenant_id=tenant_id,
            seed=seed,
            owner_epoch=owner_epoch,
            input_token_ids=list(input_token_ids),
            output_token_ids=[],
            attention_keys=[[] for _ in range(config.layers)],
            attention_values=[[] for _ in range(config.layers)],
            token_dirty_epochs=[],
            recurrent_state=[
                [
                    ((seed & 0xFFFF) + layer * 97 + lane * 13) % _MODULUS
                    for lane in range(config.recurrent_width)
                ]
                for layer in range(config.layers)
            ],
            sampler_counter=0,
            guided_state=0,
            gateway_committed_index=-1,
            client_acknowledged_index=-1,
            state_version=0,
            lifecycle=SessionLifecycle.ACTIVE,
        )
        for position, token_id in enumerate(input_token_ids):
            state._append_model_state(config, token_id=token_id, position=position)
            state.token_dirty_epochs.append(0)
        return state

    def clone(self) -> HybridDecoderState:
        return HybridDecoderState(
            session_id=self.session_id,
            request_id=self.request_id,
            tenant_id=self.tenant_id,
            seed=self.seed,
            owner_epoch=self.owner_epoch,
            input_token_ids=list(self.input_token_ids),
            output_token_ids=list(self.output_token_ids),
            attention_keys=[
                [[list(vector) for vector in token] for token in layer]
                for layer in self.attention_keys
            ],
            attention_values=[
                [[list(vector) for vector in token] for token in layer]
                for layer in self.attention_values
            ],
            token_dirty_epochs=list(self.token_dirty_epochs),
            recurrent_state=[list(layer) for layer in self.recurrent_state],
            sampler_counter=self.sampler_counter,
            guided_state=self.guided_state,
            gateway_committed_index=self.gateway_committed_index,
            client_acknowledged_index=self.client_acknowledged_index,
            state_version=self.state_version,
            lifecycle=self.lifecycle,
            transaction_id=self.transaction_id,
        )

    @property
    def token_count(self) -> int:
        return len(self.input_token_ids) + len(self.output_token_ids)

    def _append_model_state(
        self,
        config: HybridDecoderConfig,
        *,
        token_id: int,
        position: int,
    ) -> None:
        for layer in range(config.layers):
            keys: list[list[int]] = []
            values: list[list[int]] = []
            for head in range(config.kv_heads):
                key_vector = [
                    (
                        token_id * 4099
                        + position * 131
                        + layer * 1009
                        + head * 73
                        + dimension * 17
                        + 19
                    )
                    % _MODULUS
                    for dimension in range(config.head_dimension)
                ]
                value_vector = [
                    (
                        token_id * 6151
                        + position * 193
                        + layer * 1237
                        + head * 89
                        + dimension * 29
                        + 23
                    )
                    % _MODULUS
                    for dimension in range(config.head_dimension)
                ]
                keys.append(key_vector)
                values.append(value_vector)
            self.attention_keys[layer].append(keys)
            self.attention_values[layer].append(values)
            previous = self.recurrent_state[layer]
            self.recurrent_state[layer] = [
                (previous[lane] * 33 + token_id * 17 + position * 7 + layer * 101 + lane * 11)
                % _MODULUS
                for lane in range(config.recurrent_width)
            ]

    def _counter_word(self) -> int:
        value = (self.seed + self.sampler_counter * 0x9E3779B97F4A7C15) & _MASK_64
        value = (value + 0x9E3779B97F4A7C15) & _MASK_64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
        return value ^ (value >> 31)

    def peek_next_token(self, config: HybridDecoderConfig) -> int:
        state_mix = sum(sum(layer) for layer in self.recurrent_state) & _MASK_64
        raw = self._counter_word() ^ state_mix ^ (self.token_count * 0xD6E8FEB86659FD93)
        candidate = raw % config.model.vocabulary_size
        candidate -= candidate % 4
        candidate += self.guided_state
        if candidate >= config.model.vocabulary_size:
            candidate -= 4
        return int(candidate)

    def generate(self, config: HybridDecoderConfig, *, transaction_id: str | None) -> TokenEvent:
        if self.lifecycle is not SessionLifecycle.ACTIVE:
            raise ValueError(f"session is not active: {self.lifecycle.value}")
        if self.token_count >= config.max_context_tokens:
            self.lifecycle = SessionLifecycle.TERMINAL
            raise ValueError("session reached the configured context limit")
        token_id = self.peek_next_token(config)
        position = self.token_count
        self.state_version += 1
        self._append_model_state(config, token_id=token_id, position=position)
        self.token_dirty_epochs.append(self.state_version)
        self.output_token_ids.append(token_id)
        self.sampler_counter += 1
        self.guided_state = (self.guided_state + (token_id // 4) % 3 + 1) % 4
        token_index = len(self.output_token_ids) - 1
        self.transaction_id = transaction_id
        return TokenEvent(
            session_id=self.session_id,
            owner_epoch=self.owner_epoch,
            token_index=token_index,
            token_id=token_id,
            state_commit_version=self.state_version,
            transaction_id=transaction_id,
        )

    def acknowledge_gateway(self, *, token_index: int, owner_epoch: int) -> None:
        if owner_epoch != self.owner_epoch:
            raise ValueError("gateway acknowledgment owner epoch is stale")
        if token_index != self.gateway_committed_index + 1:
            raise ValueError("gateway acknowledgment must advance exactly one token")
        if token_index >= len(self.output_token_ids):
            raise ValueError("gateway cannot acknowledge a token not emitted by the runtime")
        self.gateway_committed_index = token_index
        self.state_version += 1

    def logical_manifest(self, config: HybridDecoderConfig) -> LogicalStateManifest:
        if self.gateway_committed_index != len(self.output_token_ids) - 1:
            raise ValueError("consistent portable state requires all emitted tokens to be resolved")
        last_generated = len(self.output_token_ids) - 1
        return LogicalStateManifest(
            schema_version="continuum.logical.runtime.v1",
            session_id=self.session_id,
            request_id=self.request_id,
            tenant_id=self.tenant_id,
            model=config.model,
            input_token_ids=tuple(self.input_token_ids),
            committed_output_token_ids=tuple(self.output_token_ids),
            uncommitted_speculative_token_ids=(),
            attention_layer_count=config.layers,
            attention_head_count=config.kv_heads,
            attention_kv_head_count=config.kv_heads,
            attention_head_dimension=config.head_dimension,
            positional_encoding_semantics="absolute-position-v1",
            attention_window_semantics="dense-causal-full-context",
            recurrent_state=tuple(tuple(layer) for layer in self.recurrent_state),
            sampler=SamplerSnapshot(
                algorithm="continuum-counter-v1",
                seed=self.seed,
                counter=self.sampler_counter,
                temperature_milli=1000,
                top_k=config.model.vocabulary_size,
                top_p_millionths=1_000_000,
            ),
            guided_decoding=GuidedDecodingSnapshot(
                automaton_id=config.automaton_id,
                automaton_hash=config.automaton_hash,
                state=self.guided_state,
                accepted_prefix_length=len(self.output_token_ids),
            ),
            client_delivery=ClientDeliverySnapshot(
                last_generated_token_index=last_generated,
                last_gateway_committed_token_index=self.gateway_committed_index,
                last_client_acknowledged_token_index=self.client_acknowledged_index,
                stream_owner_epoch=self.owner_epoch,
                terminal_status=(
                    ClientTerminalStatus.CANCELLED
                    if self.lifecycle is SessionLifecycle.CANCELLED
                    else ClientTerminalStatus.COMPLETED
                    if self.lifecycle is SessionLifecycle.TERMINAL
                    else ClientTerminalStatus.OPEN
                ),
            ),
            owner_epoch=self.owner_epoch,
            state_version=self.state_version,
            dirty_epoch=self.state_version,
            continuation_hash=self.continuation_hash(config),
        )

    def continuation_hash(self, config: HybridDecoderConfig) -> str:
        """Hash only state that controls continuation, excluding physical layout and owner."""

        document = {
            "attention_keys": self.attention_keys,
            "attention_values": self.attention_values,
            "guided_state": self.guided_state,
            "input_token_ids": self.input_token_ids,
            "model_hash": config.model.model_hash,
            "output_token_ids": self.output_token_ids,
            "recurrent_state": self.recurrent_state,
            "sampler_counter": self.sampler_counter,
            "seed": self.seed,
            "state_producer_hash": config.model.state_producer_hash,
        }
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()


def state_from_manifest(
    config: HybridDecoderConfig,
    manifest: LogicalStateManifest,
    *,
    destination_session_id: str,
    attention_keys: list[list[list[list[int]]]],
    attention_values: list[list[list[list[int]]]],
    lifecycle: SessionLifecycle = SessionLifecycle.PREPARED,
) -> HybridDecoderState:
    expected_tokens = len(manifest.input_token_ids) + len(manifest.committed_output_token_ids)
    if len(attention_keys) != config.layers or len(attention_values) != config.layers:
        raise ValueError("attention layer count does not match the destination model")
    if any(len(layer) != expected_tokens for layer in (*attention_keys, *attention_values)):
        raise ValueError("attention token coverage does not match token history")
    return HybridDecoderState(
        session_id=destination_session_id,
        request_id=manifest.request_id,
        tenant_id=manifest.tenant_id,
        seed=manifest.sampler.seed,
        owner_epoch=manifest.owner_epoch,
        input_token_ids=list(manifest.input_token_ids),
        output_token_ids=list(manifest.committed_output_token_ids),
        attention_keys=attention_keys,
        attention_values=attention_values,
        token_dirty_epochs=[0] * len(manifest.input_token_ids)
        + [manifest.state_version] * len(manifest.committed_output_token_ids),
        recurrent_state=[list(layer) for layer in manifest.recurrent_state],
        sampler_counter=manifest.sampler.counter,
        guided_state=manifest.guided_decoding.state,
        gateway_committed_index=manifest.client_delivery.last_gateway_committed_token_index,
        client_acknowledged_index=manifest.client_delivery.last_client_acknowledged_token_index,
        state_version=manifest.state_version,
        lifecycle=lifecycle,
    )
