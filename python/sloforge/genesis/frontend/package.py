"""Reference-package loading with path containment and content identity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from sloforge.ir.canonical import canonical_hash
from sloforge.util import sha256_file

from .models import ReferencePackageManifest


@dataclass(frozen=True)
class LoadedReferencePackage:
    root: Path
    manifest_path: Path
    manifest: ReferencePackageManifest
    manifest_hash: str
    source_hashes: tuple[tuple[str, str], ...]
    package_hash: str

    def resolve(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise ValueError(
                f"reference-package path escapes package root: {relative_path}"
            ) from error
        if not candidate.is_file():
            raise FileNotFoundError(f"reference-package artifact is missing: {candidate}")
        return candidate


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("reference-package manifest must be an object")
    return cast(dict[str, Any], value)


def load_reference_package(path: Path) -> LoadedReferencePackage:
    """Load a package directory or manifest without executing model source."""

    resolved = path.resolve()
    if resolved.is_dir():
        candidates = [
            resolved / "reference_package.json",
            resolved / "reference_package.yaml",
            resolved / "reference_package.yml",
        ]
        manifest_path = next((item for item in candidates if item.is_file()), None)
        if manifest_path is None:
            raise FileNotFoundError(f"no reference_package.json/yaml found under {resolved}")
        root = resolved
    elif resolved.is_file():
        manifest_path = resolved
        root = resolved.parent
    else:
        raise FileNotFoundError(f"reference package does not exist: {resolved}")

    # Pydantic's JSON validation preserves strict scalar typing while accepting
    # JSON arrays for immutable tuple fields. Direct strict Python validation
    # would incorrectly reject every standards-compliant JSON array.
    manifest = ReferencePackageManifest.model_validate_json(
        json.dumps(_read_mapping(manifest_path)), strict=True
    )
    provisional = LoadedReferencePackage(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_hash=canonical_hash(manifest),
        source_hashes=(),
        package_hash="0" * 64,
    )
    relative_paths = sorted(
        {
            manifest.reference_module,
            manifest.tokenizer_module,
            manifest.sample_generator_module,
            manifest.sample_corpus,
            manifest.quality_contract.search_corpus,
            manifest.quality_contract.final_evaluation_corpus,
        }
    )
    source_hashes = tuple(
        (relative, sha256_file(provisional.resolve(relative))) for relative in relative_paths
    )
    identity = {
        "manifest_hash": provisional.manifest_hash,
        "source_hashes": source_hashes,
    }
    package_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return LoadedReferencePackage(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_hash=provisional.manifest_hash,
        source_hashes=source_hashes,
        package_hash=package_hash,
    )
