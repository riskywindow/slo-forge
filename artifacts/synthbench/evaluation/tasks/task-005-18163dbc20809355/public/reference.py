"""Generated ServingSynthBench reference implementation; not a serving adapter."""

import hashlib
import math

ARCHITECTURE_SEED = 1735642584606348117
VOCABULARY_SIZE = 64
MAXIMUM_SEQUENCE_LENGTH = 64
BLOCKS = (('state_space', 2, 1, 2, 5, 8, 2), ('recurrent_state', 6, 1, 4, 3, 8, 8), ('speculative_head', 6, 1, 4, 3, 8, 4), ('custom_normalization', 6, 4, 2, 5, 4, 8), ('convolutional_state', 8, 1, 3, 3, 4, 4), ('convolutional_state', 8, 4, 4, 5, 4, 8))


def _seed(base, *parts):
    payload = "\0".join((str(base), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_model(*, seed):
    del seed
    digest = hashlib.sha256(f"model:{ARCHITECTURE_SEED}".encode()).digest()
    return {"biases": tuple((digest[index % len(digest)] / 255.0) - 0.5 for index in range(len(BLOCKS)))}


def allocate_state(*, request_id, prompt_tokens, seed):
    del seed
    return {
        "history": [],
        "recurrent": 0.0,
        "quantized": 0,
        "expert_loads": [0] * max(block[3] for block in BLOCKS),
        "speculative": _seed(ARCHITECTURE_SEED, request_id) % 17,
        "prompt_checksum": sum(prompt_tokens) % 257,
    }


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
    del seed
    value = _advance(model, previous_token, state)
    jitter = _seed(ARCHITECTURE_SEED, position, previous_token) % VOCABULARY_SIZE
    center = int(abs(value) * 1009 + state["quantized"] + jitter) % VOCABULARY_SIZE
    logits = tuple(-abs(index - center) + value * ((index % 3) - 1) for index in range(VOCABULARY_SIZE))
    return {"logits": logits, "state": state}


def custom_sampler(*, logits, seed):
    del seed
    ordered = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
    if any(block[0] == "custom_sampler" for block in BLOCKS):
        return ordered[1]
    return ordered[0]
