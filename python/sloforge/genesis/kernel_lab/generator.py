"""Deterministic restricted-source generation for the focused CPU kernel."""

from __future__ import annotations

import ast
import hashlib
from typing import Literal

from .models import (
    AliasingConstraint,
    ArchitectureConstraint,
    KernelBackend,
    KernelCandidate,
    KernelParameters,
    NumericalContract,
    OperatorSchema,
    ShapeDimension,
    TensorConstraint,
)


def hybrid_quantized_state_schema() -> OperatorSchema:
    batch = ShapeDimension(name="batch", minimum=1, maximum=128, multiple_of=1)
    return OperatorSchema(
        semantic_contract=(
            "per-request HybridDecoder recurrent state: saturating int8 round-to-nearest "
            "of previous*0.625 + activation*31.0"
        ),
        inputs=(
            TensorConstraint(
                name="previous_state",
                dtype="int8",
                shape=(batch,),
                allowed_strides=(1, 2, 3),
                contiguous_required=False,
                mutable=True,
            ),
            TensorConstraint(
                name="activation",
                dtype="float64",
                shape=(batch,),
                allowed_strides=(1, 2, 3),
                contiguous_required=False,
                mutable=False,
            ),
        ),
        output=TensorConstraint(
            name="updated_state",
            dtype="int8",
            shape=(batch,),
            allowed_strides=(1, 2, 3),
            contiguous_required=False,
            mutable=True,
        ),
        aliasing=AliasingConstraint(output_may_alias=("previous_state",)),
        numerical=NumericalContract(),
        architecture=ArchitectureConstraint(
            backend=KernelBackend.PYTHON_CPU,
            architectures=("cpu-generic",),
            minimum_python="3.11",
            requires_gpu=False,
        ),
    )


def _candidate_source(parameters: KernelParameters) -> str:
    coefficient_setup = (
        "    previous_coefficient = 0.625\n    activation_coefficient = 31.0\n"
        if parameters.cache_coefficients
        else ""
    )
    previous_expression = (
        "previous_value * previous_coefficient"
        if parameters.cache_coefficients
        else "previous_value * 0.625"
    )
    activation_expression = (
        "activation * activation_coefficient"
        if parameters.cache_coefficients
        else "activation * 31.0"
    )
    if parameters.clamp_style == "branches":
        clamp = (
            "            if rounded < -127:\n"
            "                rounded = -127\n"
            "            elif rounded > 127:\n"
            "                rounded = 127\n"
        )
    else:
        clamp = "            rounded = max(-127, min(127, rounded))\n"
    return (
        '"""Generated untrusted HybridDecoder state-update candidate."""\n'
        "\n"
        "import math\n"
        "\n"
        f"UNROLL_FACTOR = {parameters.unroll_factor}\n"
        "\n"
        "def quantized_recurrent_state_update(\n"
        "    previous_storage, activation_storage, count, previous_offset, previous_stride,\n"
        "    activation_offset, activation_stride, output_alias_previous,\n"
        "):\n"
        "    if count < 1 or count > 128:\n"
        "        raise ValueError('count outside supported domain')\n"
        "    if previous_stride not in (1, 2, 3) or activation_stride not in (1, 2, 3):\n"
        "        raise ValueError('stride outside supported domain')\n"
        f"{coefficient_setup}"
        "    output = previous_storage if output_alias_previous else [0] * count\n"
        "    logical = [0] * count\n"
        "    position = 0\n"
        "    while position < count:\n"
        "        stop = min(count, position + UNROLL_FACTOR)\n"
        "        while position < stop:\n"
        "            previous_value = previous_storage[previous_offset + position * previous_stride]\n"
        "            activation = activation_storage[activation_offset + position * activation_stride]\n"
        "            if previous_value < -127 or previous_value > 127:\n"
        "                raise ValueError('previous state outside symmetric int8 domain')\n"
        "            if not math.isfinite(activation):\n"
        "                raise ValueError('activation must be finite')\n"
        f"            rounded = round({previous_expression} + {activation_expression})\n"
        f"{clamp}"
        "            if output_alias_previous:\n"
        "                output[previous_offset + position * previous_stride] = rounded\n"
        "            else:\n"
        "                output[position] = rounded\n"
        "            logical[position] = rounded\n"
        "            position += 1\n"
        "    return logical\n"
    )


def generate_candidates(*, seed: int) -> tuple[tuple[KernelCandidate, str], ...]:
    """Return a deterministic local search population and its exact sources."""

    if seed < 0:
        raise ValueError("kernel generation seed must be non-negative")
    schema = hybrid_quantized_state_schema()
    unroll_factors: tuple[Literal[1, 2, 4], ...] = (1, 2, 4)
    clamp_styles: tuple[Literal["branches", "builtins"], ...] = ("branches", "builtins")
    parameter_space = tuple(
        KernelParameters(unroll_factor=unroll, clamp_style=clamp, cache_coefficients=cache)
        for unroll in unroll_factors
        for clamp in clamp_styles
        for cache in (True, False)
    )
    generated = []
    for parameters in parameter_space:
        source = _candidate_source(parameters)
        digest = hashlib.sha256(source.encode()).hexdigest()
        candidate = KernelCandidate(
            candidate_id=(
                f"qstate-u{parameters.unroll_factor}-{parameters.clamp_style}-"
                f"{'cached' if parameters.cache_coefficients else 'literal'}"
            ),
            operator_schema=schema,
            backend=KernelBackend.PYTHON_CPU,
            target_architecture="cpu-generic",
            parameters=parameters,
            source_sha256=digest,
            deterministic_seed=seed,
        )
        generated.append((candidate, source))
    return tuple(generated)


_ALLOWED_CALLS = {
    "ValueError",
    "len",
    "max",
    "min",
    "range",
    "round",
}
_BANNED_NAMES = {
    "__import__",
    "builtins",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "os",
    "setattr",
    "subprocess",
    "sys",
}


def validate_generated_source(candidate: KernelCandidate, source: str) -> tuple[str, ...]:
    """Statically enforce the restricted candidate language before sandboxing."""

    diagnostics: list[str] = []
    if len(source.encode()) > 64 * 1024:
        diagnostics.append("source_size_limit")
    if hashlib.sha256(source.encode()).hexdigest() != candidate.source_sha256:
        diagnostics.append("source_digest_mismatch")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return (*diagnostics, "source_syntax_error")
    function_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if function_names != ["quantized_recurrent_state_update"]:
        diagnostics.append("unexpected_top_level_function")
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.Await,
                ast.ClassDef,
                ast.Global,
                ast.Lambda,
                ast.Nonlocal,
            ),
        ):
            diagnostics.append(f"forbidden_syntax:{type(node).__name__}")
        elif isinstance(node, ast.Import):
            if [(alias.name, alias.asname) for alias in node.names] != [("math", None)]:
                diagnostics.append("forbidden_import")
        elif isinstance(node, ast.ImportFrom):
            diagnostics.append("forbidden_import")
        elif isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            diagnostics.append(f"forbidden_name:{node.id}")
        elif isinstance(node, ast.Attribute):
            allowed_attribute = (
                isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id == "math"
                and node.attr == "isfinite"
            )
            if not allowed_attribute:
                diagnostics.append("forbidden_attribute_access")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in _ALLOWED_CALLS:
                if node.func.id != "quantized_recurrent_state_update":
                    diagnostics.append(f"forbidden_call:{node.func.id}")
            elif isinstance(node.func, ast.Attribute) and not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
                and node.func.attr == "isfinite"
            ):
                diagnostics.append("forbidden_call:attribute")
    return tuple(sorted(set(diagnostics)))
