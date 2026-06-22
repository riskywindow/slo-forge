raise RuntimeError("static inspection must never execute this module")


def load_model(*, seed):
    return seed


def allocate_state(*, request_id, prompt_tokens, seed):
    return (request_id, prompt_tokens, seed)


def prefill(*, model, prompt_tokens, state, seed):
    return state


def decode_step(*, model, previous_token, state, position, seed):
    return {"logits": [0.0], "state": state}


def custom_sampler(*, logits, seed):
    return 0


def torch_export_fixture():
    return None
