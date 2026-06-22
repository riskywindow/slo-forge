"""Small unseen-style hybrid decoder reference; intentionally not a serving adapter."""

from __future__ import annotations

import hashlib
import math

VOCABULARY_SIZE = 32
WINDOW_SIZE = 6
EXPERT_COUNT = 3


def load_model(*, seed: int) -> dict[str, object]:
    digest = hashlib.sha256(f"hybrid-model:{seed}".encode()).digest()
    expert_bias = tuple((digest[index] / 255.0) - 0.5 for index in range(EXPERT_COUNT))
    return {"expert_bias": expert_bias, "recurrent_decay": 0.75, "speculative_stride": 3}


def allocate_state(
    *, request_id: str, prompt_tokens: tuple[int, ...], seed: int
) -> dict[str, object]:
    state_seed = int.from_bytes(hashlib.sha256(f"{request_id}:{seed}".encode()).digest()[:4], "big")
    return {
        "kv_window": [],
        "recurrent_state": 0.0,
        "quantized_state": 0,
        "expert_loads": [0, 0, 0],
        "speculative_state": state_seed % 7,
        "prompt_length": len(prompt_tokens),
    }


def sliding_window_attention(values: list[int], token: int) -> float:
    window = [*values, token][-WINDOW_SIZE:]
    weights = range(1, len(window) + 1)
    return sum(value * weight for value, weight in zip(window, weights, strict=True)) / sum(weights)


def recurrent_update(previous: float, activation: float, decay: float) -> float:
    return math.tanh(previous * decay + activation / VOCABULARY_SIZE)


def sparse_moe_dispatch(activation: float, expert_bias: tuple[float, ...]) -> tuple[float, int]:
    route = int(abs(activation) * 997) % len(expert_bias)
    transformed = math.tanh(activation * (1.0 + expert_bias[route]))
    return transformed, route


def quantized_state_update(previous: int, activation: float) -> int:
    """Declared custom state transform: symmetric int8 round-to-nearest."""

    combined = previous * 0.625 + activation * 31.0
    return max(-127, min(127, round(combined)))


def _advance(model: dict[str, object], token: int, state: dict[str, object]) -> None:
    kv_window = list(state["kv_window"])
    attention = sliding_window_attention(kv_window, token)
    recurrent = recurrent_update(
        float(state["recurrent_state"]), attention, float(model["recurrent_decay"])
    )
    expert_bias = tuple(float(value) for value in model["expert_bias"])
    activation, route = sparse_moe_dispatch(recurrent, expert_bias)
    expert_loads = list(state["expert_loads"])
    expert_loads[route] += 1
    state["expert_loads"] = expert_loads
    state["recurrent_state"] = activation
    state["quantized_state"] = quantized_state_update(int(state["quantized_state"]), activation)
    state["kv_window"] = [*kv_window, token][-WINDOW_SIZE:]


def prefill(
    *,
    model: dict[str, object],
    prompt_tokens: tuple[int, ...],
    state: dict[str, object],
    seed: int,
) -> dict[str, object]:
    del seed
    for token in prompt_tokens:
        _advance(model, token, state)
    return state


def decode_step(
    *,
    model: dict[str, object],
    previous_token: int,
    state: dict[str, object],
    position: int,
    seed: int,
) -> dict[str, object]:
    _advance(model, previous_token, state)
    recurrent = float(state["recurrent_state"])
    quantized = int(state["quantized_state"])
    speculative_stride = int(model["speculative_stride"])
    speculative = (int(state["speculative_state"]) + speculative_stride + position) % 11
    state["speculative_state"] = speculative
    jitter = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:2], "big") % 13
    center = (previous_token + quantized + speculative + jitter) % VOCABULARY_SIZE
    logits = [-(abs(index - center)) + recurrent * ((index % 3) - 1) for index in range(32)]
    return {"logits": logits, "state": state}


def custom_sampler(*, logits: tuple[float, ...], seed: int) -> int:
    """Nonstandard seeded top-2 sampler with stable tie handling."""

    ordered = sorted(range(len(logits)), key=lambda index: (-logits[index], index))[:2]
    choice = hashlib.sha256(str(seed).encode()).digest()[0] % len(ordered)
    return ordered[choice]


def torch_export_fixture() -> tuple[object, tuple[object, ...], dict[str, object], object]:
    """Return a bounded tensor-only fragment when optional PyTorch inspection is requested."""

    import torch

    class TensorFragment(torch.nn.Module):
        def forward(self, input_ids: object) -> object:
            values = input_ids.to(dtype=torch.float32)
            return torch.tanh(values / 32.0) + values

    sequence = torch.export.Dim("sequence", min=1, max=64)
    example = torch.ones((1, 4), dtype=torch.int64)
    return TensorFragment(), (example,), {}, {"input_ids": {1: sequence}}
