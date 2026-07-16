"""Strict adapter for declared reference-runtime entry points."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Protocol, cast

from sloforge.genesis.frontend.models import EntryPointContract

_SOURCE_IMPORT_LOCK = RLock()


class CallableEntryPoint(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class DecodeResult:
    logits: tuple[float, ...]
    state: object


def _load_source(path: Path, identity: str) -> ModuleType:
    """Compile the declared source bytes directly; never consume adjacent bytecode caches."""

    if not path.is_file():
        raise FileNotFoundError(f"generated runtime source is missing: {path}")
    specification = importlib.util.spec_from_file_location(identity, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load generated runtime source: {path}")
    module = importlib.util.module_from_spec(specification)
    package_root = str(path.parent.resolve(strict=True))
    with _SOURCE_IMPORT_LOCK:
        previous_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, package_root)
        try:
            source = path.read_bytes()
            code = compile(source, str(path), "exec", dont_inherit=True)
            exec(code, module.__dict__)
        finally:
            if sys.path[0] == package_root:
                sys.path.pop(0)
            else:
                sys.path.remove(package_root)
            sys.dont_write_bytecode = previous_bytecode
    return module


def _entry(module: ModuleType, symbol: str) -> CallableEntryPoint:
    value = getattr(module, symbol, None)
    if value is None or not callable(value):
        raise ValueError(f"declared runtime entry point {symbol!r} is not callable")
    return cast(CallableEntryPoint, value)


class ReferenceRuntimeAdapter:
    """Execute only the package entry points named in the validated manifest."""

    def __init__(
        self,
        *,
        reference_path: Path,
        tokenizer_path: Path,
        entry_points: EntryPointContract,
        identity: str,
        seed: int,
    ) -> None:
        reference = _load_source(reference_path, f"{identity}_reference")
        tokenizer = _load_source(tokenizer_path, f"{identity}_tokenizer")
        self._allocate_state = _entry(reference, entry_points.allocate_state)
        self._prefill = _entry(reference, entry_points.prefill)
        self._decode_step = _entry(reference, entry_points.decode_step)
        self._sample = _entry(reference, entry_points.sample)
        self._tokenize = _entry(tokenizer, entry_points.tokenize)
        self._detokenize = _entry(tokenizer, entry_points.detokenize)
        self._model = _entry(reference, entry_points.load_model)(seed=seed)

    def tokenize(self, text: str) -> tuple[int, ...]:
        value = self._tokenize(text)
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value
        ):
            raise TypeError("tokenize must return a sequence of non-negative integers")
        return tuple(value)

    def detokenize(self, token_id: int) -> str:
        value = self._detokenize(token_id)
        if not isinstance(value, str):
            raise TypeError("detokenize must return a string")
        return value

    def allocate_state(self, request_id: str, prompt_tokens: tuple[int, ...], seed: int) -> object:
        return self._allocate_state(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            seed=seed,
        )

    def prefill(self, prompt_tokens: tuple[int, ...], state: object, seed: int) -> object:
        value = self._prefill(
            model=self._model,
            prompt_tokens=prompt_tokens,
            state=state,
            seed=seed,
        )
        if value is None:
            raise TypeError("prefill must return the resulting request state")
        return value

    def decode_step(
        self,
        previous_token: int,
        state: object,
        position: int,
        seed: int,
    ) -> DecodeResult:
        value = self._decode_step(
            model=self._model,
            previous_token=previous_token,
            state=state,
            position=position,
            seed=seed,
        )
        if not isinstance(value, dict) or set(value) != {"logits", "state"}:
            raise TypeError("decode_step must return exactly {'logits': ..., 'state': ...}")
        logits = value["logits"]
        if not isinstance(logits, (list, tuple)) or not logits:
            raise TypeError("decode_step logits must be a non-empty sequence")
        if not all(
            isinstance(item, (float, int)) and not isinstance(item, bool) for item in logits
        ):
            raise TypeError("decode_step logits must contain finite numeric values")
        numeric = tuple(float(item) for item in logits)
        if not all(value == value and abs(value) != float("inf") for value in numeric):
            raise ValueError("decode_step returned non-finite logits outside the runtime contract")
        return DecodeResult(logits=numeric, state=value["state"])

    def sample(self, logits: tuple[float, ...], seed: int) -> int:
        value = self._sample(logits=logits, seed=seed)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < len(logits):
            raise ValueError("sampler returned a token outside the logits domain")
        return value
