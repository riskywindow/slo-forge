"""Seeded typed unseen-task grammar and pure-Python reference-package generator."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import TypedDict, cast

from sloforge.genesis.ir import canonical_json

from .models import (
    ArchitectureSpec,
    BlockKind,
    BlockSpec,
    GrammarConfiguration,
    HiddenCase,
    TaskDescriptor,
    WorkloadRequest,
)


class _ReferenceState(TypedDict):
    history: list[int]
    recurrent: float
    quantized: int
    expert_loads: list[int]
    speculative: int
    prompt_checksum: int


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seed(base: int, *parts: object) -> int:
    payload = "\0".join((str(base), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _canonical_line(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite synthbench artifact: {path}")
    path.write_bytes(payload)


def _block_effect(
    kind: BlockKind,
    value: float,
    token: int,
    state: _ReferenceState,
    block: BlockSpec,
    bias: float,
) -> float:
    history = list(state["history"])
    recurrent = state["recurrent"]
    quantized = state["quantized"]
    if kind is BlockKind.DENSE_ATTENTION:
        return math.tanh(value + (sum(history) + token) / (len(history) + 1) / 16.0)
    if kind is BlockKind.SLIDING_WINDOW_ATTENTION:
        window = [*history, token][-block.window_size :]
        return math.tanh(value + sum(window) / len(window) / 16.0)
    if kind is BlockKind.GROUPED_QUERY_ATTENTION:
        return math.tanh(value + (token % block.group_count) / block.group_count + bias)
    if kind is BlockKind.GATED_MLP:
        return math.tanh(value * (1.0 / (1.0 + math.exp(-(value + bias)))))
    if kind is BlockKind.SPARSE_MOE:
        route = int(abs(value + bias) * 997) % block.expert_count
        loads = list(state["expert_loads"])
        loads[route] += 1
        state["expert_loads"] = loads
        return math.tanh(value * (1.0 + route / block.expert_count) + bias)
    if kind is BlockKind.STATE_SPACE:
        recurrent = recurrent * 0.82 + value * 0.18
        state["recurrent"] = recurrent
        return recurrent
    if kind is BlockKind.RECURRENT:
        recurrent = math.tanh(recurrent * 0.7 + value)
        state["recurrent"] = recurrent
        return recurrent
    if kind is BlockKind.CONVOLUTIONAL_STATE:
        window = [*history, token][-block.kernel_size :]
        return math.tanh(value + sum((index + 1) * item for index, item in enumerate(window)) / 64)
    if kind is BlockKind.CUSTOM_NORMALIZATION:
        return value / (1.0 + abs(value))
    if kind is BlockKind.QUANTIZED_STATE:
        quantization_bits: int = block.quantization_bits
        limit = 2 ** (quantization_bits - 1) - 1
        scaled: float = quantized * 0.625 + value * float(limit)
        rounded: int = round(scaled)
        quantized = max(-limit, min(limit, rounded))
        state["quantized"] = quantized
        return float(quantized) / float(limit)
    if kind is BlockKind.RESIDUAL_BRANCH:
        return math.tanh(value + token / 32.0 + bias)
    if kind is BlockKind.SPECULATIVE_HEAD:
        state["speculative"] = (int(state["speculative"]) + token + block.state_size) % 17
        return value + int(state["speculative"]) / 64.0
    if kind is BlockKind.CROSS_ATTENTION:
        return math.tanh(value + float(state["prompt_checksum"]) / 257.0)
    return value


def execute_architecture(
    architecture: ArchitectureSpec,
    *,
    request_id: str,
    prompt_tokens: tuple[int, ...],
    maximum_new_tokens: int,
    request_seed: int,
) -> tuple[int, ...]:
    """Independent generator-side reference used to commit public and hidden cases."""

    model_digest = hashlib.sha256(f"model:{architecture.seed}".encode()).digest()
    biases = tuple(
        (model_digest[index % len(model_digest)] / 255.0) - 0.5
        for index in range(len(architecture.blocks))
    )
    state: _ReferenceState = {
        "history": [],
        "recurrent": 0.0,
        "quantized": 0,
        "expert_loads": [0] * max(block.expert_count for block in architecture.blocks),
        "speculative": _seed(request_seed, request_id) % 17,
        "prompt_checksum": sum(prompt_tokens) % 257,
    }

    def advance(token: int) -> float:
        value = token / architecture.vocabulary_size
        for index, block in enumerate(architecture.blocks):
            value = _block_effect(block.kind, value, token, state, block, biases[index])
        history = list(state["history"])
        state["history"] = [*history, token][-architecture.maximum_sequence_length :]
        return value

    for prompt_token in prompt_tokens:
        advance(prompt_token)
    custom_sampler = any(block.kind is BlockKind.CUSTOM_SAMPLER for block in architecture.blocks)
    previous = prompt_tokens[-1]
    generated: list[int] = []
    for position in range(maximum_new_tokens):
        value = advance(previous)
        jitter = _seed(request_seed, position, previous) % architecture.vocabulary_size
        center = int(abs(value) * 1009 + state["quantized"] + jitter) % architecture.vocabulary_size
        logits = tuple(
            -abs(index - center) + value * ((index % 3) - 1)
            for index in range(architecture.vocabulary_size)
        )
        ordered = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
        token = (
            ordered[_seed(request_seed, "sample", position) % 2] if custom_sampler else ordered[0]
        )
        generated.append(token)
        previous = token
    return tuple(generated)


def _architecture(
    task_seed: int, ordinal: int, configuration: GrammarConfiguration
) -> ArchitectureSpec:
    generator = random.Random(task_seed)
    hidden = generator.choice((8, 12, 16, 24, 32))
    vocabulary = generator.choice((32, 48, 64))
    maximum_sequence = generator.choice((24, 32, 48, 64))
    block_count = generator.randint(configuration.minimum_blocks, configuration.maximum_blocks)
    kinds = cast(tuple[BlockKind, ...], tuple(BlockKind))
    state_kinds = (
        BlockKind.STATE_SPACE,
        BlockKind.RECURRENT,
        BlockKind.CONVOLUTIONAL_STATE,
        BlockKind.QUANTIZED_STATE,
    )
    chosen: list[BlockKind] = [kinds[ordinal % len(kinds)], state_kinds[ordinal % len(state_kinds)]]
    chosen.append(BlockKind.CUSTOM_SAMPLER if ordinal % 2 == 0 else BlockKind.SPECULATIVE_HEAD)
    while len(chosen) < block_count:
        chosen.append(generator.choice(kinds))
    blocks = tuple(
        BlockSpec(
            block_id=f"block-{index:02d}-{kind.value.replace('_', '-')}",
            kind=kind,
            hidden_size=hidden,
            window_size=generator.choice((2, 4, 6, 8)),
            expert_count=generator.choice((2, 3, 4)),
            top_k=1,
            group_count=generator.choice(
                tuple(value for value in (1, 2, 4) if hidden % value == 0)
            ),
            kernel_size=generator.choice((3, 5)),
            quantization_bits=generator.choice((4, 8)),
            state_size=generator.choice((2, 4, 8)),
        )
        for index, kind in enumerate(chosen)
    )
    return ArchitectureSpec(
        architecture_id=f"architecture-{task_seed:016x}",
        seed=task_seed,
        vocabulary_size=vocabulary,
        hidden_size=hidden,
        maximum_sequence_length=maximum_sequence,
        blocks=blocks,
    )


def _tokens(generator: random.Random, length: int, vocabulary: int) -> tuple[int, ...]:
    return tuple(generator.randrange(1, vocabulary) for _ in range(length))


def _request(
    architecture: ArchitectureSpec,
    *,
    task_seed: int,
    case_index: int,
    length: int,
    hidden: bool,
) -> WorkloadRequest:
    case_seed = _seed(task_seed, "hidden" if hidden else "public", case_index)
    generator = random.Random(case_seed)
    prompt = _tokens(generator, length, architecture.vocabulary_size)
    maximum_new = 2 + case_index % 4
    request_id = f"{'hidden' if hidden else 'request'}-{case_index:03d}"
    expected = execute_architecture(
        architecture,
        request_id=request_id,
        prompt_tokens=prompt,
        maximum_new_tokens=maximum_new,
        request_seed=case_seed,
    )
    return WorkloadRequest(
        request_id=request_id,
        prompt_tokens=prompt,
        maximum_new_tokens=maximum_new,
        seed=case_seed,
        arrival_offset_ms=float((case_index // 3) * 5),
        priority=case_index % 4,
        deadline_ms=float(100 + length * 4),
        expected_tokens=expected,
    )


_REFERENCE_SOURCE = '''"""Generated ServingSynthBench reference implementation; not a serving adapter."""

import hashlib
import math

ARCHITECTURE_SEED = {architecture_seed}
VOCABULARY_SIZE = {vocabulary_size}
MAXIMUM_SEQUENCE_LENGTH = {maximum_sequence_length}
BLOCKS = {blocks!r}


def _seed(base, *parts):
    payload = "\\0".join((str(base), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_model(*, seed):
    del seed
    digest = hashlib.sha256(f"model:{{ARCHITECTURE_SEED}}".encode()).digest()
    return {{"biases": tuple((digest[index % len(digest)] / 255.0) - 0.5 for index in range(len(BLOCKS)))}}


def allocate_state(*, request_id, prompt_tokens, seed):
    return {{
        "history": [],
        "recurrent": 0.0,
        "quantized": 0,
        "expert_loads": [0] * max(block[3] for block in BLOCKS),
        "speculative": _seed(seed, request_id) % 17,
        "prompt_checksum": sum(prompt_tokens) % 257,
    }}


def custom_normalization(value):
    return value / (1.0 + abs(value))


def quantized_state_update(previous, value, bits):
    limit = 2 ** (bits - 1) - 1
    return max(-limit, min(limit, round(previous * 0.625 + value * limit)))


def _advance(model, token, state):
    value = token / VOCABULARY_SIZE
    history = list(state["history"])
    for index, block in enumerate(BLOCKS):
        kind, window_size, group_count, expert_count, kernel_size, quantization_bits, state_size = block
        bias = model["biases"][index]
        if kind == "dense_attention":
            value = math.tanh(value + (sum(history) + token) / (len(history) + 1) / 16.0)
        elif kind == "sliding_window_attention":
            window = (history + [token])[-window_size:]
            value = math.tanh(value + sum(window) / len(window) / 16.0)
        elif kind == "grouped_query_attention":
            value = math.tanh(value + (token % group_count) / group_count + bias)
        elif kind == "gated_mlp":
            value = math.tanh(value * (1.0 / (1.0 + math.exp(-(value + bias)))))
        elif kind == "sparse_moe":
            route = int(abs(value + bias) * 997) % expert_count
            state["expert_loads"][route] += 1
            value = math.tanh(value * (1.0 + route / expert_count) + bias)
        elif kind == "state_space":
            state["recurrent"] = state["recurrent"] * 0.82 + value * 0.18
            value = state["recurrent"]
        elif kind == "recurrent_state":
            state["recurrent"] = math.tanh(state["recurrent"] * 0.7 + value)
            value = state["recurrent"]
        elif kind == "convolutional_state":
            window = (history + [token])[-kernel_size:]
            value = math.tanh(value + sum((offset + 1) * item for offset, item in enumerate(window)) / 64)
        elif kind == "custom_normalization":
            value = custom_normalization(value)
        elif kind == "quantized_state_transformation":
            state["quantized"] = quantized_state_update(state["quantized"], value, quantization_bits)
            value = state["quantized"] / (2 ** (quantization_bits - 1) - 1)
        elif kind == "residual_branch":
            value = math.tanh(value + token / 32.0 + bias)
        elif kind == "speculative_head":
            state["speculative"] = (state["speculative"] + token + state_size) % 17
            value = value + state["speculative"] / 64.0
        elif kind == "cross_attention":
            value = math.tanh(value + state["prompt_checksum"] / 257.0)
    state["history"] = (history + [token])[-MAXIMUM_SEQUENCE_LENGTH:]
    return value


def prefill(*, model, prompt_tokens, state, seed):
    del seed
    for token in prompt_tokens:
        _advance(model, token, state)
    return state


def decode_step(*, model, previous_token, state, position, seed):
    value = _advance(model, previous_token, state)
    jitter = _seed(seed, position, previous_token) % VOCABULARY_SIZE
    center = int(abs(value) * 1009 + state["quantized"] + jitter) % VOCABULARY_SIZE
    logits = tuple(-abs(index - center) + value * ((index % 3) - 1) for index in range(VOCABULARY_SIZE))
    return {{"logits": logits, "state": state}}


def custom_sampler(*, logits, seed):
    ordered = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
    if any(block[0] == "custom_sampler" for block in BLOCKS):
        return ordered[int(seed) % 2]
    return ordered[0]
'''


_TOKENIZER_SOURCE = '''"""Generated bounded synthetic tokenizer."""

VOCABULARY_SIZE = {vocabulary_size}


def encode(text):
    if not text:
        raise ValueError("text cannot be empty")
    return [(ord(character) % (VOCABULARY_SIZE - 1)) + 1 for character in text]


def decode_token(token_id):
    if not 0 <= token_id < VOCABULARY_SIZE:
        raise ValueError("token outside vocabulary")
    return chr(33 + token_id % 90)
'''


_SAMPLE_SOURCE = '''"""Generated public sample provider."""


def sample_inputs(*, seed):
    return ({{"text": "synthbench", "maximum_new_tokens": 3, "seed": seed}},)
'''


def _manifest(architecture: ArchitectureSpec) -> dict[str, object]:
    custom: list[dict[str, object]] = []
    if any(block.kind is BlockKind.CUSTOM_NORMALIZATION for block in architecture.blocks):
        custom.append(
            {
                "operator_id": "custom-normalization",
                "symbol": "custom_normalization",
                "semantic_description": "bounded scalar normalization value/(1+abs(value))",
                "exact": True,
                "input_domain": ["input_ids"],
                "state_reads": [],
                "state_writes": [],
                "verification_obligations": ["compare finite boundary values"],
            }
        )
    if any(block.kind is BlockKind.QUANTIZED_STATE for block in architecture.blocks):
        custom.append(
            {
                "operator_id": "quantized-state-update",
                "symbol": "quantized_state_update",
                "semantic_description": "saturating signed quantized persistent-state update",
                "exact": True,
                "input_domain": ["input_ids"],
                "state_reads": ["quantized"],
                "state_writes": ["quantized"],
                "verification_obligations": ["enumerate saturation boundaries"],
            }
        )
    dimension = {"name": "batch", "minimum": 1, "maximum": 1, "multiple_of": 1}
    return {
        "schema_version": "1.0.0",
        "package_id": f"synthbench-{architecture.architecture_id}",
        "reference_module": "reference.py",
        "tokenizer_module": "tokenizer.py",
        "sample_generator_module": "sample_inputs.py",
        "sample_corpus": "search_samples.jsonl",
        "entry_points": {
            "load_model": "load_model",
            "allocate_state": "allocate_state",
            "prefill": "prefill",
            "decode_step": "decode_step",
            "sample": "custom_sampler",
            "tokenize": "encode",
            "detokenize": "decode_token",
            "sample_inputs": "sample_inputs",
            "torch_export": None,
        },
        "state_contract": {
            "ownership": "request",
            "fields": [
                {
                    "field_id": "history",
                    "kind": "kv",
                    "dtype": "int64",
                    "shape": [
                        {
                            "name": "sequence",
                            "minimum": 1,
                            "maximum": architecture.maximum_sequence_length,
                            "multiple_of": 1,
                        }
                    ],
                    "mutable": True,
                    "persistent_across_tokens": True,
                    "reset_at_request_boundary": True,
                    "alias_group": None,
                    "quantization": None,
                },
                {
                    "field_id": "recurrent",
                    "kind": "recurrent",
                    "dtype": "float64",
                    "shape": [dimension],
                    "mutable": True,
                    "persistent_across_tokens": True,
                    "reset_at_request_boundary": True,
                    "alias_group": None,
                    "quantization": None,
                },
                {
                    "field_id": "quantized",
                    "kind": "custom",
                    "dtype": "int8",
                    "shape": [dimension],
                    "mutable": True,
                    "persistent_across_tokens": True,
                    "reset_at_request_boundary": True,
                    "alias_group": None,
                    "quantization": "declared-by-block",
                },
                {
                    "field_id": "expert_loads",
                    "kind": "custom",
                    "dtype": "int32",
                    "shape": [{"name": "experts", "minimum": 4, "maximum": 4, "multiple_of": 1}],
                    "mutable": True,
                    "persistent_across_tokens": True,
                    "reset_at_request_boundary": True,
                    "alias_group": None,
                    "quantization": None,
                },
                {
                    "field_id": "speculative",
                    "kind": "speculative",
                    "dtype": "int32",
                    "shape": [dimension],
                    "mutable": True,
                    "persistent_across_tokens": True,
                    "reset_at_request_boundary": True,
                    "alias_group": None,
                    "quantization": None,
                },
            ],
            "mutation_atomicity": "per_token",
            "cancellation_releases_state": True,
            "migration_supported": False,
        },
        "semantic_contract": {
            "token_commitment": "on_emit",
            "deterministic_for_seed": True,
            "batching_axes": ["batch"],
            "batch_isolation": "independent_requests",
            "streaming_order": "strict_token_order",
            "cancellation": "immediate_before_commit",
            "retry_after_first_token": "forbidden",
            "allowed_control_flow": ["if", "for"],
            "required_invariants": [
                "request state is isolated",
                "seeded sampling is deterministic",
            ],
        },
        "quality_contract": {
            "metrics": [{"metric": "exact_token_match", "threshold": 1.0, "comparison": "exact"}],
            "final_evaluation_corpus": "final_holdout_commitment.json",
            "search_corpus": "search_samples.jsonl",
            "permit_approximation": False,
        },
        "supported_input_domain": {
            "tensors": [
                {
                    "name": "input_ids",
                    "dtype": "int64",
                    "dimensions": [
                        dimension,
                        {
                            "name": "sequence",
                            "minimum": 1,
                            "maximum": architecture.maximum_sequence_length,
                            "multiple_of": 1,
                        },
                    ],
                    "contiguous": True,
                    "allowed_strides": [],
                }
            ],
            "scalars": [
                {
                    "name": "seed",
                    "kind": "integer",
                    "minimum": 0.0,
                    "maximum": 18446744073709551615.0,
                    "allowed_values": [],
                }
            ],
            "maximum_prompt_tokens": architecture.maximum_sequence_length,
            "maximum_generated_tokens": 8,
        },
        "custom_operators": custom,
        "workflow": None,
        "software_preconditions": ["Python >=3.11"],
    }


def _package_hash(directory: Path) -> str:
    entries = [
        (path.relative_to(directory).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    return _hash_bytes(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode())


def generate_tasks(
    configuration: GrammarConfiguration, output_directory: Path
) -> tuple[TaskDescriptor, ...]:
    """Generate public packages first and evaluator-only cases separately."""

    output_directory.mkdir(parents=True, exist_ok=True)
    descriptors: list[TaskDescriptor] = []
    for ordinal in range(configuration.count):
        task_seed = _seed(configuration.seed, "task", ordinal)
        architecture = _architecture(task_seed, ordinal, configuration)
        task_id = f"task-{ordinal:03d}-{task_seed:016x}"
        task_directory = output_directory / task_id
        public = task_directory / "public"
        hidden_directory = task_directory / "hidden"
        public_requests = tuple(
            _request(
                architecture,
                task_seed=task_seed,
                case_index=index,
                length=1 + (index * 7) % architecture.maximum_sequence_length,
                hidden=False,
            )
            for index in range(configuration.public_cases_per_task)
        )
        hidden_lengths = (1, architecture.maximum_sequence_length, 7, 11, 13, 17)
        hidden_cases = tuple(
            HiddenCase(
                case_id=f"case-{index:03d}",
                request=_request(
                    architecture,
                    task_seed=task_seed,
                    case_index=index,
                    length=min(
                        hidden_lengths[index % len(hidden_lengths)],
                        architecture.maximum_sequence_length,
                    ),
                    hidden=True,
                ),
                trap=(
                    "minimum_shape",
                    "maximum_shape",
                    "rare_length",
                    "state_reset",
                    "sampler_tie",
                    "burst_priority",
                )[index % 6],
            )
            for index in range(configuration.hidden_cases_per_task)
        )
        hidden_payload = b"".join(_canonical_line(case) + b"\n" for case in hidden_cases)
        hidden_commitment = _hash_bytes(hidden_payload)
        blocks = tuple(
            (
                block.kind.value,
                block.window_size,
                block.group_count,
                block.expert_count,
                block.kernel_size,
                block.quantization_bits,
                block.state_size,
            )
            for block in architecture.blocks
        )
        _write(
            public / "reference.py",
            _REFERENCE_SOURCE.format(
                architecture_seed=architecture.seed,
                vocabulary_size=architecture.vocabulary_size,
                maximum_sequence_length=architecture.maximum_sequence_length,
                blocks=blocks,
            ).encode(),
        )
        _write(
            public / "tokenizer.py",
            _TOKENIZER_SOURCE.format(vocabulary_size=architecture.vocabulary_size).encode(),
        )
        _write(public / "sample_inputs.py", _SAMPLE_SOURCE.encode())
        _write(public / "architecture.json", canonical_json(architecture) + b"\n")
        _write(public / "reference_package.json", _canonical_line(_manifest(architecture)) + b"\n")
        _write(
            public / "search_samples.jsonl",
            b"".join(
                _canonical_line(
                    {
                        "maximum_new_tokens": request.maximum_new_tokens,
                        "seed": request.seed,
                        "text": f"public-{index}",
                    }
                )
                + b"\n"
                for index, request in enumerate(public_requests)
            ),
        )
        _write(
            public / "workload.jsonl",
            b"".join(_canonical_line(request) + b"\n" for request in public_requests),
        )
        _write(
            public / "final_holdout_commitment.json",
            _canonical_line(
                {
                    "algorithm": "sha256",
                    "case_count": len(hidden_cases),
                    "value": hidden_commitment,
                }
            )
            + b"\n",
        )
        _write(hidden_directory / "hidden_cases.jsonl", hidden_payload)
        public_hash = _package_hash(public)
        descriptor = TaskDescriptor(
            task_id=task_id,
            seed=task_seed,
            architecture=architecture,
            public_package_path="public",
            workload_path="public/workload.jsonl",
            hidden_cases_path="hidden/hidden_cases.jsonl",
            hidden_commitment=hidden_commitment,
            public_package_hash=public_hash,
        )
        _write(task_directory / "task.json", canonical_json(descriptor) + b"\n")
        descriptors.append(descriptor)
    _write(
        output_directory / "index.json",
        _canonical_line(
            {
                "schema_version": "1.0.0",
                "generation_seed": configuration.seed,
                "tasks": [descriptor.model_dump(mode="json") for descriptor in descriptors],
            }
        )
        + b"\n",
    )
    return tuple(descriptors)


def load_task(path: Path) -> TaskDescriptor:
    return TaskDescriptor.model_validate_json(path.read_bytes(), strict=True)


def load_workload(task_directory: Path, descriptor: TaskDescriptor) -> tuple[WorkloadRequest, ...]:
    path = task_directory / descriptor.workload_path
    return tuple(
        WorkloadRequest.model_validate_json(line, strict=True)
        for line in path.read_bytes().splitlines()
        if line
    )


def load_hidden_cases(task_directory: Path, descriptor: TaskDescriptor) -> tuple[HiddenCase, ...]:
    path = task_directory / descriptor.hidden_cases_path
    payload = path.read_bytes()
    if _hash_bytes(payload) != descriptor.hidden_commitment:
        raise ValueError("hidden evaluation cases do not match their public commitment")
    return tuple(
        HiddenCase.model_validate_json(line, strict=True) for line in payload.splitlines() if line
    )
