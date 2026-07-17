"""Path and secret handling shared by environment capture and branching."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TypeVar, cast

from .models import JsonValue

REDACTED = "[REDACTED]"
_SECRET_KEY_PARTS = ("secret", "token", "password", "passwd", "api_key", "private_key")
_DEFAULT_SECRET_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "**/.env",
    "**/.env.*",
    "**/*secret*",
)


class PathSafetyError(ValueError):
    """An environment path could escape its declared root or namespace."""


def normalize_relative_path(value: str | os.PathLike[str], *, allow_dot: bool = False) -> str:
    raw = os.fspath(value).replace(os.sep, "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or (not allow_dot and raw in {"", "."}):
        raise PathSafetyError(f"unsafe relative path {raw!r}")
    return path.as_posix()


def safe_destination(root: Path, relative: str) -> Path:
    normalized = normalize_relative_path(relative)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    # Lexical validation is intentional: resolving would follow an attacker-controlled symlink.
    if root == candidate or root not in candidate.parents:
        raise PathSafetyError(f"path escapes environment root: {relative!r}")
    current = root
    for component in PurePosixPath(normalized).parts[:-1]:
        current /= component
        if current.is_symlink():
            raise PathSafetyError(f"parent component is a symlink: {relative!r}")
    return candidate


def validate_symlink_target(link_path: str, target: str) -> None:
    target_path = PurePosixPath(target)
    if target_path.is_absolute():
        raise PathSafetyError(f"absolute symlink target is not portable: {target!r}")
    combined = PurePosixPath(link_path).parent.joinpath(target_path)
    depth = 0
    for part in combined.parts:
        if part == "..":
            depth -= 1
        elif part not in {"", "."}:
            depth += 1
        if depth < 0:
            raise PathSafetyError(f"symlink target escapes environment root: {target!r}")


def is_redacted_path(path: str, explicit_patterns: Sequence[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(path, pattern)
        for pattern in (*_DEFAULT_SECRET_PATTERNS, *explicit_patterns)
    )


def redact_text(value: str, secrets: Sequence[str]) -> str:
    redacted = value
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        redacted = redacted.replace(secret, REDACTED)
    return redacted


def redact_mapping(value: Mapping[str, object], *, secrets: Sequence[str]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, item in value.items():
        lowered = key.lower()
        if any(part in lowered for part in _SECRET_KEY_PARTS):
            result[str(key)] = REDACTED
        else:
            result[str(key)] = _redact_value(item, secrets=secrets)
    return result


def _redact_value(value: object, *, secrets: Sequence[str]) -> JsonValue:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, Mapping):
        return redact_mapping(cast(Mapping[str, object], value), secrets=secrets)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | str):
        return [_redact_value(item, secrets=secrets) for item in value]
    return redact_text(str(value), secrets)


T = TypeVar("T")


def bounded_tuple(values: Sequence[T], *, maximum: int, label: str) -> tuple[T, ...]:
    if len(values) > maximum:
        raise ValueError(f"{label} exceeds the configured bound of {maximum}")
    return tuple(values)
