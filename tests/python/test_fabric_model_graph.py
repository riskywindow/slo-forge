from __future__ import annotations

import json
from pathlib import Path

import pytest

from sloforge.fabric.ir import LayerKind, canonical_hash
from sloforge.fabric.model_graph import inspect_local_model, synthetic_moe_model_graph


def test_synthetic_moe_graph_is_stable_and_physically_descriptive() -> None:
    first = synthetic_moe_model_graph()
    second = synthetic_moe_model_graph()
    assert canonical_hash(first) == canonical_hash(second)
    moe_layers = tuple(layer for layer in first.layers if layer.kind is LayerKind.MOE)
    assert len(moe_layers) == 8
    assert all(len(layer.experts) == 8 for layer in moe_layers)
    assert all(
        abs(sum(expert.expected_load for expert in layer.experts) - 1.0) < 1e-12
        for layer in moe_layers
    )
    assert "prefill_decode_disaggregation" in first.runtime_features


def test_local_snapshot_inspection_never_executes_model_code(tmp_path: Path) -> None:
    config = {
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
        "num_local_experts": 4,
        "torch_dtype": "bfloat16",
        "auto_map": {"AutoModel": "malicious.py:Model"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"safe-fixture-weights")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "malicious.py").write_text("raise AssertionError('executed')", encoding="utf-8")

    graph = inspect_local_model(tmp_path, model_id="fixture/moe", revision="deadbeef")
    assert graph.model_id == "fixture/moe"
    assert graph.model_revision == "deadbeef"
    assert len(graph.layers) == 4
    assert all(len(layer.experts) == 4 for layer in graph.layers if layer.kind is LayerKind.MOE)


def test_local_inspection_fails_closed_without_weight_identity(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "max_position_embeddings": 4096,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="safetensors"):
        inspect_local_model(tmp_path, model_id="fixture/missing", revision="deadbeef")
