"""Build physical compiler model graphs from auditable model metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

from sloforge.fabric.ir import (
    CommunicationRequirement,
    ExpertSpec,
    LayerKind,
    LayerSpec,
    ModelGraph,
    PrecisionMode,
)
from sloforge.ir import ArtifactDigest
from sloforge.util import sha256_file


def _digest_payload(value: object) -> ArtifactDigest:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return ArtifactDigest(value=hashlib.sha256(encoded).hexdigest())


def synthetic_moe_model_graph() -> ModelGraph:
    """Return the explicitly synthetic MoE graph used by the CPU Fabric demo."""

    layers: list[LayerSpec] = [
        LayerSpec(
            layer_id="embedding",
            ordinal=0,
            kind=LayerKind.EMBEDDING,
            parameter_bytes=256 * 1024 * 1024,
            activation_bytes_per_token=4096,
            kv_bytes_per_token=0,
            indivisible=True,
        )
    ]
    ordinal = 1
    for block in range(8):
        layers.append(
            LayerSpec(
                layer_id=f"attention-{block}",
                ordinal=ordinal,
                kind=LayerKind.ATTENTION,
                parameter_bytes=64 * 1024 * 1024,
                activation_bytes_per_token=8192,
                kv_bytes_per_token=4096,
                communication=(
                    CommunicationRequirement(
                        operation="all_reduce",
                        bytes_per_token=4096.0,
                        synchronization_required=True,
                        parallelism_dimension="tensor",
                    ),
                ),
            )
        )
        ordinal += 1
        experts = tuple(
            ExpertSpec(
                expert_id=f"block-{block}-expert-{expert}",
                parameter_bytes=32 * 1024 * 1024,
                activation_bytes_per_token=4096,
                expected_load=0.30 if expert == 0 else 0.10,
            )
            for expert in range(8)
        )
        layers.append(
            LayerSpec(
                layer_id=f"moe-{block}",
                ordinal=ordinal,
                kind=LayerKind.MOE,
                parameter_bytes=0,
                activation_bytes_per_token=8192,
                kv_bytes_per_token=0,
                experts=experts,
                communication=(
                    CommunicationRequirement(
                        operation="all_to_all",
                        bytes_per_token=8192.0,
                        synchronization_required=True,
                        parallelism_dimension="expert",
                    ),
                ),
            )
        )
        ordinal += 1
    layers.append(
        LayerSpec(
            layer_id="output",
            ordinal=ordinal,
            kind=LayerKind.OUTPUT,
            parameter_bytes=256 * 1024 * 1024,
            activation_bytes_per_token=4096,
            kv_bytes_per_token=0,
            indivisible=True,
        )
    )
    weight_bytes = sum(
        layer.parameter_bytes + sum(expert.parameter_bytes for expert in layer.experts)
        for layer in layers
    )
    identity = {
        "name": "sloforge/synthetic-moe-fabric-v1",
        "layers": [layer.model_dump(mode="json") for layer in layers],
    }
    digest = _digest_payload(identity)
    return ModelGraph(
        model_id="sloforge/synthetic-moe-fabric-v1",
        model_revision="1.0.0",
        model_digest=digest,
        tokenizer_digest=_digest_payload({"tokenizer": "synthetic-tokenizer-v1"}),
        hidden_size=4096,
        attention_heads=32,
        key_value_heads=8,
        maximum_sequence_length=131_072,
        layers=tuple(layers),
        precision_modes=(
            PrecisionMode(
                name="bfloat16",
                weight_bytes=weight_bytes,
                runtime_features=(
                    "tensor_parallel",
                    "expert_parallel",
                    "prefill_decode_disaggregation",
                ),
            ),
        ),
        runtime_features=(
            "synthetic_fixture",
            "tensor_parallel",
            "expert_parallel",
            "prefill_decode_disaggregation",
        ),
    )


def _integer(config: dict[str, Any], *keys: str, minimum: int = 1) -> int:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
            return value
    raise ValueError(f"model config requires integer field from {keys}")


def _weight_identity(model_dir: Path) -> tuple[ArtifactDigest, int]:
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if index_files:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        metadata = index.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("total_size"), int):
            raise ValueError("safetensors index must record metadata.total_size")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index must contain a non-empty weight_map")
        files = sorted({str(value) for value in weight_map.values()})
        digests: list[tuple[str, str]] = []
        for name in files:
            path = model_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"safetensors shard is missing: {path}")
            digests.append((name, sha256_file(path)))
        return _digest_payload(digests), int(metadata["total_size"])
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError("model directory contains no safetensors weights")
    return _digest_payload([(path.name, sha256_file(path)) for path in weights]), sum(
        path.stat().st_size for path in weights
    )


def _tokenizer_digest(model_dir: Path) -> ArtifactDigest:
    candidates = tuple(
        path
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        if (path := model_dir / name).is_file()
    )
    if not candidates:
        raise FileNotFoundError("model directory has no tokenizer artifact")
    return _digest_payload([(path.name, sha256_file(path)) for path in candidates])


def _precision(dtype: object, weight_bytes: int) -> PrecisionMode:
    aliases: dict[str, Literal["float32", "float16", "bfloat16", "fp8", "int8", "int4"]] = {
        "float32": "float32",
        "float16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "fp8": "fp8",
        "int8": "int8",
        "int4": "int4",
    }
    key = str(dtype).lower().removeprefix("torch.")
    name = aliases.get(key)
    if name is None:
        raise ValueError(f"unsupported or missing torch_dtype {dtype!r}")
    return PrecisionMode(name=name, weight_bytes=weight_bytes)


def inspect_local_model(
    model_dir: Path,
    *,
    model_id: str,
    revision: str,
) -> ModelGraph:
    """Inspect a resolved local Hugging Face snapshot without executing model code."""

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config is missing: {config_path}")
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_value, dict):
        raise ValueError("model config must be a JSON object")
    config = cast(dict[str, Any], config_value)
    hidden = _integer(config, "hidden_size", "d_model")
    blocks = _integer(config, "num_hidden_layers", "num_layers", "n_layer")
    heads = _integer(config, "num_attention_heads", "n_head")
    kv_heads_value = config.get("num_key_value_heads", heads)
    if (
        not isinstance(kv_heads_value, int)
        or isinstance(kv_heads_value, bool)
        or kv_heads_value < 1
    ):
        raise ValueError("num_key_value_heads must be a positive integer")
    maximum_sequence = _integer(config, "max_position_embeddings", "model_max_length", minimum=128)
    model_digest, weight_bytes = _weight_identity(model_dir)
    experts_per_layer = config.get("num_local_experts", config.get("num_experts", 0))
    if not isinstance(experts_per_layer, int) or isinstance(experts_per_layer, bool):
        raise ValueError("num_local_experts must be an integer when present")
    if experts_per_layer < 0 or experts_per_layer > 1_024:
        raise ValueError("num_local_experts is outside the supported bounded range")
    layer_weight = max(1, weight_bytes // max(1, blocks + 2))
    layers: list[LayerSpec] = [
        LayerSpec(
            layer_id="embedding",
            ordinal=0,
            kind=LayerKind.EMBEDDING,
            parameter_bytes=layer_weight,
            activation_bytes_per_token=hidden * 2,
            kv_bytes_per_token=0,
            indivisible=True,
        )
    ]
    for block in range(blocks):
        if experts_per_layer:
            expert_bytes = max(1, layer_weight // experts_per_layer)
            experts = tuple(
                ExpertSpec(
                    expert_id=f"layer-{block}-expert-{expert}",
                    parameter_bytes=expert_bytes,
                    activation_bytes_per_token=hidden * 2,
                    expected_load=1.0 / experts_per_layer,
                )
                for expert in range(experts_per_layer)
            )
            kind = LayerKind.MOE
            communication = (
                CommunicationRequirement(
                    operation="all_to_all",
                    bytes_per_token=float(hidden * 2),
                    synchronization_required=True,
                    parallelism_dimension="expert",
                ),
            )
        else:
            experts = ()
            kind = LayerKind.ATTENTION
            communication = (
                CommunicationRequirement(
                    operation="all_reduce",
                    bytes_per_token=float(hidden * 2),
                    synchronization_required=True,
                    parallelism_dimension="tensor",
                ),
            )
        layers.append(
            LayerSpec(
                layer_id=f"block-{block}",
                ordinal=block + 1,
                kind=kind,
                parameter_bytes=0 if experts else layer_weight,
                activation_bytes_per_token=hidden * 2,
                kv_bytes_per_token=2 * kv_heads_value * max(1, hidden // heads) * 2,
                experts=experts,
                communication=communication,
            )
        )
    layers.append(
        LayerSpec(
            layer_id="output",
            ordinal=len(layers),
            kind=LayerKind.OUTPUT,
            parameter_bytes=max(1, weight_bytes - layer_weight * (blocks + 1)),
            activation_bytes_per_token=hidden * 2,
            kv_bytes_per_token=0,
            indivisible=True,
        )
    )
    runtime_features = ["tensor_parallel"]
    if experts_per_layer:
        runtime_features.append("expert_parallel")
    return ModelGraph(
        model_id=model_id,
        model_revision=revision,
        model_digest=model_digest,
        tokenizer_digest=_tokenizer_digest(model_dir),
        hidden_size=hidden,
        attention_heads=heads,
        key_value_heads=kv_heads_value,
        maximum_sequence_length=maximum_sequence,
        layers=tuple(layers),
        precision_modes=(_precision(config.get("torch_dtype"), weight_bytes),),
        runtime_features=tuple(runtime_features),
    )
