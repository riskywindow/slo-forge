"""Fail-closed static analysis for previously unseen Python reference models."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypeAlias

from sloforge.ir.canonical import write_canonical

from .models import (
    DiagnosticSeverity,
    DimensionContract,
    InspectionDiagnostic,
    InspectionResult,
    RecoveredAlias,
    RecoveredControlFlow,
    RecoveredGraph,
    RecoveredOperator,
    SourceLocation,
)
from .package import LoadedReferencePackage, load_reference_package
from .torch_adapter import inspect_with_torch_export

_TENSOR_MARKERS = (
    "attention",
    "matmul",
    "linear",
    "softmax",
    "norm",
    "relu",
    "gelu",
    "sigmoid",
    "tanh",
    "conv",
    "quant",
    "dequant",
    "topk",
    "argmax",
    "embedding",
    "expert",
    "moe",
)
_STATE_MARKERS = ("state", "cache", "kv", "recurrent", "prefill", "decode")
_SAMPLING_MARKERS = ("sample", "sampler", "logits")
_SAFE_PYTHON_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
_SAFE_METHODS = {"append", "copy", "extend", "get", "items", "keys", "pop", "values"}
_SAFE_IMPORTED_MODULES = {"hashlib", "math"}

OperatorCategory: TypeAlias = Literal[
    "tensor", "state", "control", "sampling", "custom", "python", "unknown"
]
ControlKind: TypeAlias = Literal["if", "for", "while", "match", "try"]


def _symbol(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _symbol(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _symbol(node.value)
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def _names(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.id
                for item in ast.walk(node)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
            }
        )
    )


def _state_field(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id not in {"state", "request_state"}:
            return None
        value = node.slice
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"state", "request_state"}
    ):
        return node.attr
    return None


def _location(relative_path: str, node: ast.AST) -> SourceLocation:
    return SourceLocation(
        relative_path=relative_path,
        line=max(1, getattr(node, "lineno", 1)),
        column=max(0, getattr(node, "col_offset", 0)),
    )


def _operator_category(symbol: str, custom_symbols: set[str]) -> OperatorCategory:
    lowered = symbol.lower()
    if symbol in custom_symbols:
        return "custom"
    if any(marker in lowered for marker in _SAMPLING_MARKERS):
        return "sampling"
    if any(marker in lowered for marker in _STATE_MARKERS):
        return "state"
    if any(marker in lowered for marker in _TENSOR_MARKERS):
        return "tensor"
    return "python"


class _ModuleInspection:
    def __init__(self, package: LoadedReferencePackage, relative_path: str) -> None:
        self.package = package
        self.relative_path = relative_path
        self.operators: list[RecoveredOperator] = []
        self.control_flow: list[RecoveredControlFlow] = []
        self.aliases: list[RecoveredAlias] = []
        self.diagnostics: list[InspectionDiagnostic] = []

    @property
    def identity(self) -> str:
        return Path(self.relative_path).stem.replace("_", "-")

    def run(self) -> None:
        source_path = self.package.resolve(self.relative_path)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        local_symbols = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        imported_roots = {
            alias.asname or alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        custom_by_symbol = {
            operator.symbol: operator for operator in self.package.manifest.custom_operators
        }
        declared_state = {field.field_id for field in self.package.manifest.state_contract.fields}

        functions = sorted(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        for function in functions:
            category = _operator_category(function.name, set(custom_by_symbol))
            custom = custom_by_symbol.get(function.name)
            if category == "python":
                continue
            self.operators.append(
                RecoveredOperator(
                    operator_id=f"{self.identity}-op-{len(self.operators):05d}",
                    symbol=function.name,
                    category=category,
                    inputs=tuple(
                        argument.arg
                        for argument in (
                            *function.args.posonlyargs,
                            *function.args.args,
                            *function.args.kwonlyargs,
                        )
                    ),
                    location=_location(self.relative_path, function),
                    custom_operator_id=custom.operator_id if custom is not None else None,
                )
            )

        calls = sorted(
            (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
            key=lambda node: (node.lineno, node.col_offset),
        )
        for ordinal, call in enumerate(calls):
            symbol = _symbol(call.func)
            category = _operator_category(symbol, set(custom_by_symbol))
            custom = custom_by_symbol.get(symbol)
            reads = {
                field
                for item in ast.walk(call)
                if (field := _state_field(item)) is not None
                and isinstance(item, (ast.Subscript, ast.Attribute))
                and isinstance(item.ctx, ast.Load)
            }
            writes = {
                field
                for item in ast.walk(call)
                if (field := _state_field(item)) is not None
                and isinstance(item, (ast.Subscript, ast.Attribute))
                and isinstance(item.ctx, ast.Store)
            }
            if custom is not None:
                reads.update(custom.state_reads)
                writes.update(custom.state_writes)
            self.operators.append(
                RecoveredOperator(
                    operator_id=f"{self.identity}-op-{len(self.operators):05d}",
                    symbol=symbol,
                    category=category,
                    inputs=_names(call),
                    state_reads=tuple(sorted(reads & declared_state)),
                    state_writes=tuple(sorted(writes & declared_state)),
                    location=_location(self.relative_path, call),
                    custom_operator_id=custom.operator_id if custom is not None else None,
                )
            )
            root = symbol.split(".", 1)[0]
            leaf = symbol.rsplit(".", 1)[-1]
            known = (
                custom is not None
                or symbol in local_symbols
                or root in local_symbols
                or (root in imported_roots and root in _SAFE_IMPORTED_MODULES)
                or symbol in _SAFE_PYTHON_CALLS
                or leaf in _SAFE_METHODS
                or category != "python"
            )
            if not known:
                self.diagnostics.append(
                    InspectionDiagnostic(
                        diagnostic_id=f"{self.identity}-unknown-call-{ordinal:05d}",
                        severity=DiagnosticSeverity.OBLIGATION,
                        category="unknown_semantics",
                        message=f"call {symbol!r} has no declared operator semantics",
                        location=_location(self.relative_path, call),
                        proof_obligation=f"declare and independently verify semantics for {symbol}",
                    )
                )

        state_assignments = (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        )
        for node in sorted(state_assignments, key=lambda item: (item.lineno, item.col_offset)):
            if isinstance(node, ast.Assign):
                targets = tuple(node.targets)
                value_node: ast.expr | None = node.value
            else:
                targets = (node.target,)
                value_node = node.value
            writes = {
                field
                for target in targets
                for item in ast.walk(target)
                if (field := _state_field(item)) is not None
            }
            if not writes:
                continue
            reads = (
                {
                    field
                    for item in ast.walk(value_node)
                    if (field := _state_field(item)) is not None
                }
                if value_node is not None
                else set()
            )
            self.operators.append(
                RecoveredOperator(
                    operator_id=f"{self.identity}-op-{len(self.operators):05d}",
                    symbol="state_write",
                    category="state",
                    inputs=_names(value_node) if value_node is not None else (),
                    state_reads=tuple(sorted(reads & declared_state)),
                    state_writes=tuple(sorted(writes & declared_state)),
                    location=_location(self.relative_path, node),
                )
            )

        control_types: tuple[tuple[type[ast.AST], ControlKind], ...] = (
            (ast.If, "if"),
            (ast.For, "for"),
            (ast.While, "while"),
            (ast.Match, "match"),
            (ast.Try, "try"),
        )
        declared = set(self.package.manifest.semantic_contract.allowed_control_flow)
        for control_node in sorted(ast.walk(tree), key=lambda item: getattr(item, "lineno", 0)):
            kind = next(
                (name for node_type, name in control_types if isinstance(control_node, node_type)),
                None,
            )
            if kind is None:
                continue
            allowed = kind in declared
            self.control_flow.append(
                RecoveredControlFlow(
                    kind=kind,
                    location=_location(self.relative_path, control_node),
                    declared_semantics=allowed,
                )
            )
            if not allowed:
                ordinal = len(self.diagnostics)
                self.diagnostics.append(
                    InspectionDiagnostic(
                        diagnostic_id=f"{self.identity}-control-flow-{ordinal:05d}",
                        severity=DiagnosticSeverity.UNSUPPORTED,
                        category="dynamic_control_flow",
                        message=f"{kind} control flow is outside the declared semantic contract",
                        location=_location(self.relative_path, control_node),
                        proof_obligation=f"declare bounded {kind} semantics before synthesis",
                    )
                )

        for assignment in (node for node in ast.walk(tree) if isinstance(node, ast.Assign)):
            if len(assignment.targets) != 1:
                continue
            target = assignment.targets[0]
            if isinstance(target, ast.Name) and isinstance(assignment.value, ast.Name):
                self.aliases.append(
                    RecoveredAlias(
                        source=assignment.value.id,
                        target=target.id,
                        location=_location(self.relative_path, assignment),
                        explicit_contract=False,
                    )
                )
                ordinal = len(self.diagnostics)
                self.diagnostics.append(
                    InspectionDiagnostic(
                        diagnostic_id=f"{self.identity}-alias-{ordinal:05d}",
                        severity=DiagnosticSeverity.OBLIGATION,
                        category="aliasing",
                        message=f"assignment may alias {assignment.value.id!r} as {target.id!r}",
                        location=_location(self.relative_path, assignment),
                        proof_obligation="verify that the alias does not violate state ownership",
                    )
                )


def _unique_dimensions(package: LoadedReferencePackage) -> tuple[DimensionContract, ...]:
    by_name: dict[str, DimensionContract] = {}
    tensors = package.manifest.supported_input_domain.tensors
    for tensor in tensors:
        for dimension in tensor.dimensions:
            existing = by_name.get(dimension.name)
            if existing is not None and existing != dimension:
                raise ValueError(f"symbolic dimension {dimension.name!r} has conflicting bounds")
            by_name[dimension.name] = dimension
    return tuple(by_name[name] for name in sorted(by_name))


def _entry_point_diagnostics(
    package: LoadedReferencePackage,
) -> tuple[InspectionDiagnostic, ...]:
    contract = package.manifest.entry_points
    expected = {
        package.manifest.reference_module: (
            contract.load_model,
            contract.allocate_state,
            contract.prefill,
            contract.decode_step,
            contract.sample,
            *((contract.torch_export,) if contract.torch_export is not None else ()),
        ),
        package.manifest.tokenizer_module: (contract.tokenize, contract.detokenize),
        package.manifest.sample_generator_module: (contract.sample_inputs,),
    }
    diagnostics: list[InspectionDiagnostic] = []
    for relative_path, required_symbols in expected.items():
        tree = ast.parse(package.resolve(relative_path).read_text(encoding="utf-8"))
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        asynchronous = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
        }
        for symbol in required_symbols:
            if symbol not in defined:
                diagnostics.append(
                    InspectionDiagnostic(
                        diagnostic_id=(
                            f"missing-entry-{Path(relative_path).stem.replace('_', '-')}-{symbol.replace('_', '-')}"
                        ),
                        severity=DiagnosticSeverity.UNSUPPORTED,
                        category="contract",
                        message=f"declared entry point {symbol!r} is absent from {relative_path}",
                        proof_obligation="provide the declared callable before runtime synthesis",
                    )
                )
            elif symbol in asynchronous:
                diagnostics.append(
                    InspectionDiagnostic(
                        diagnostic_id=(
                            f"async-entry-{Path(relative_path).stem.replace('_', '-')}-{symbol.replace('_', '-')}"
                        ),
                        severity=DiagnosticSeverity.UNSUPPORTED,
                        category="contract",
                        message=f"declared entry point {symbol!r} is asynchronous",
                        proof_obligation="provide a synchronous bounded entry point",
                    )
                )
    return tuple(diagnostics)


def inspect_reference_package(
    path: Path,
    *,
    use_torch_export: bool = False,
    output_path: Path | None = None,
) -> InspectionResult:
    """Recover declared and static semantics without executing source by default."""

    package = load_reference_package(path)
    inspections = [
        _ModuleInspection(package, relative)
        for relative in sorted(
            {
                package.manifest.reference_module,
                package.manifest.tokenizer_module,
                package.manifest.sample_generator_module,
            }
        )
    ]
    for inspection in inspections:
        inspection.run()
    operators = tuple(operator for inspection in inspections for operator in inspection.operators)
    state_dependencies = tuple(
        sorted(
            {
                f"{operator.operator_id}:{field}:{access}"
                for operator in operators
                for access, fields in (
                    ("read", operator.state_reads),
                    ("write", operator.state_writes),
                )
                for field in fields
            }
        )
    )
    torch_evidence = inspect_with_torch_export(package) if use_torch_export else None
    entry_point_diagnostics = _entry_point_diagnostics(package)
    graph = RecoveredGraph(
        operators=operators,
        input_tensors=package.manifest.supported_input_domain.tensors,
        symbolic_dimensions=_unique_dimensions(package),
        state_fields=package.manifest.state_contract.fields,
        state_dependencies=state_dependencies,
        legal_batching_axes=package.manifest.semantic_contract.batching_axes,
        aliases=tuple(alias for inspection in inspections for alias in inspection.aliases),
        control_flow=tuple(flow for inspection in inspections for flow in inspection.control_flow),
        custom_operator_ids=tuple(
            operator.operator_id for operator in package.manifest.custom_operators
        ),
    )
    result = InspectionResult(
        package_id=package.manifest.package_id,
        package_hash=package.package_hash,
        manifest_hash=package.manifest_hash,
        source_hashes=package.source_hashes,
        graph=graph,
        semantic_contract=package.manifest.semantic_contract,
        quality_contract=package.manifest.quality_contract,
        supported_input_domain=package.manifest.supported_input_domain,
        diagnostics=tuple(
            diagnostic for inspection in inspections for diagnostic in inspection.diagnostics
        )
        + entry_point_diagnostics,
        torch_export=torch_evidence,
    )
    if output_path is not None:
        write_canonical(result, output_path)
    return result


def unsupported_obligations(result: InspectionResult) -> Iterable[InspectionDiagnostic]:
    return (
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.severity in {DiagnosticSeverity.OBLIGATION, DiagnosticSeverity.UNSUPPORTED}
    )
