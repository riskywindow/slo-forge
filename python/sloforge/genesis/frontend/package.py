"""Reference-package loading with path containment and content identity checks."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml

from sloforge.ir.canonical import canonical_hash
from sloforge.util import sha256_file

from .models import ReferencePackageManifest

_MAXIMUM_PACKAGE_ENTRIES = 4_096
_MAXIMUM_PACKAGE_BYTES = 64 * 1024 * 1024
_MAXIMUM_FILE_BYTES = 16 * 1024 * 1024
_DYNAMIC_EXECUTION_CALLS = {"__import__", "compile", "eval", "exec"}
_DYNAMIC_IMPORT_ROOTS = {"importlib", "runpy"}


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

    if path.is_symlink():
        raise ValueError("reference package path must not be a symlink")
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
    relative_paths: list[str] = []
    total_bytes = 0
    for candidate in sorted(root.rglob("*")):
        relative_entry = candidate.relative_to(root)
        if "__pycache__" in relative_entry.parts or candidate.suffix in {".pyc", ".pyo"}:
            raise ValueError(
                "reference package contains executable bytecode outside its source identity: "
                f"{relative_entry.as_posix()}"
            )
        if candidate.is_symlink():
            raise ValueError(f"reference package contains a symlink: {relative_entry.as_posix()}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(
                f"reference package contains a non-regular entry: {relative_entry.as_posix()}"
            )
        size = candidate.stat().st_size
        if size > _MAXIMUM_FILE_BYTES:
            raise ValueError(
                f"reference package file exceeds the per-file bound: {relative_entry.as_posix()}"
            )
        total_bytes += size
        relative_paths.append(relative_entry.as_posix())
        if len(relative_paths) > _MAXIMUM_PACKAGE_ENTRIES or total_bytes > _MAXIMUM_PACKAGE_BYTES:
            raise ValueError("reference package exceeds its bounded executable closure")
    required_paths = {
        manifest_path.relative_to(root).as_posix(),
        manifest.reference_module,
        manifest.tokenizer_module,
        manifest.sample_generator_module,
        manifest.sample_corpus,
        manifest.quality_contract.search_corpus,
        manifest.quality_contract.final_evaluation_corpus,
        *manifest.auxiliary_modules,
    }
    missing = required_paths - set(relative_paths)
    if missing:
        raise FileNotFoundError(f"reference-package artifact is missing: {sorted(missing)}")

    declared_modules = {
        manifest.reference_module,
        manifest.tokenizer_module,
        manifest.sample_generator_module,
        *manifest.auxiliary_modules,
    }
    local_module_paths: dict[str, str] = {}
    for module_path in (item for item in relative_paths if item.endswith(".py")):
        module_name = module_path.removesuffix(".py").replace("/", ".")
        if module_name.endswith(".__init__"):
            module_name = module_name.removesuffix(".__init__")
        local_module_paths[module_name] = module_path
    for relative in sorted(declared_modules):
        source_path = provisional.resolve(relative)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _DYNAMIC_EXECUTION_CALLS:
                    raise ValueError(
                        f"dynamic execution {node.func.id!r} is forbidden in {relative}"
                    )
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in _DYNAMIC_IMPORT_ROOTS
                ):
                    raise ValueError(f"dynamic module loading is forbidden in {relative}")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level > 0:
                    parent_parts = list(PurePosixPath(relative).parent.parts)
                    ascend = node.level - 1
                    if ascend > len(parent_parts):
                        raise ValueError(f"local dependency escapes package root in {relative}")
                    base_parts = parent_parts[: len(parent_parts) - ascend]
                    names = [module] if module else [alias.name for alias in node.names]
                    for name in names:
                        module_parts = [*base_parts, *name.split(".")]
                        local_path = "/".join(module_parts) + ".py"
                        package_path = "/".join([*module_parts, "__init__.py"])
                        declared_path = next(
                            (
                                candidate
                                for candidate in (local_path, package_path)
                                if candidate in declared_modules
                            ),
                            None,
                        )
                        if declared_path is None:
                            raise ValueError(f"undeclared local dependency {name!r} in {relative}")
                elif (
                    module in local_module_paths
                    and local_module_paths[module] not in declared_modules
                ):
                    raise ValueError(f"undeclared local dependency {module!r} in {relative}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_local_path = local_module_paths.get(alias.name)
                    if (
                        imported_local_path is not None
                        and imported_local_path not in declared_modules
                    ):
                        raise ValueError(
                            f"undeclared local dependency {alias.name!r} in {relative}"
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
