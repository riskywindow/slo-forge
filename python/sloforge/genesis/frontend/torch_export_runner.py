"""Sandbox worker for executing a declared ``torch.export`` fixture."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from sloforge.genesis.ir import canonical_json

from .models import TorchExportEvidence
from .package import LoadedReferencePackage, load_reference_package


def _load_module(package: LoadedReferencePackage) -> ModuleType:
    path = package.resolve(package.manifest.reference_module)
    specification = importlib.util.spec_from_file_location(
        f"sloforge_genesis_inspection_{package.package_hash[:12]}", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load reference module at {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _collect(package: LoadedReferencePackage) -> TorchExportEvidence:
    entry_point = package.manifest.entry_points.torch_export
    if entry_point is None:
        raise ValueError("reference package does not declare a torch_export entry point")
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("torch.export was explicitly requested but PyTorch is not installed")
    torch: Any = importlib.import_module("torch")
    module_source = _load_module(package)
    factory = getattr(module_source, entry_point, None)
    if factory is None or not callable(factory):
        raise ValueError(f"declared torch_export entry point {entry_point!r} is not callable")
    fixture = factory()
    if not isinstance(fixture, tuple) or len(fixture) != 4:
        raise TypeError(
            "torch_export entry point must return (module, args, kwargs, dynamic_shapes)"
        )
    model, args, kwargs, dynamic_shapes = fixture
    if not isinstance(args, tuple) or not isinstance(kwargs, dict):
        raise TypeError("torch_export args must be a tuple and kwargs must be a dictionary")
    exported = torch.export.export(
        model,
        args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )
    graph_nodes: list[str] = []
    tensor_metadata: list[str] = []
    for node in exported.graph_module.graph.nodes:
        graph_nodes.append(f"{node.op}:{node.name}:{node.target}")
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            tensor_metadata.append(
                f"{node.name}:shape={tuple(value.shape)}:dtype={value.dtype}:stride={value.stride()}"
            )
    return TorchExportEvidence(
        torch_version=str(torch.__version__),
        graph_nodes=tuple(graph_nodes),
        tensor_metadata=tuple(tensor_metadata),
        range_constraints=(
            f"declared_dynamic_shapes:{dynamic_shapes!r}",
            *(
                f"{symbol}:{constraint}"
                for symbol, constraint in exported.range_constraints.items()
            ),
        ),
    )


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: torch_export_runner PACKAGE_ROOT OUTPUT_PATH")
    package = load_reference_package(Path(sys.argv[1]))
    output = Path(sys.argv[2])
    output.write_bytes(canonical_json(_collect(package)) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
