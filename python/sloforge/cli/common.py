"""Shared, side-effect-free helpers for the extension command groups."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from rich.console import Console

from sloforge.ir import ArtifactDigest
from sloforge.util import canonical_json, sha256_bytes, write_json

console = Console()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def json_result(value: object) -> None:
    console.print_json(json.dumps(value, default=str))


def write_model(path: Path, value: BaseModel) -> None:
    write_json(path, value.model_dump(mode="json"))


def load_yaml_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)


def current_git_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def environment_digest() -> ArtifactDigest:
    """Hash stable process facts; capture time belongs in the enclosing IR."""

    facts = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
    }
    return ArtifactDigest(value=sha256_bytes(canonical_json(facts).encode()))
