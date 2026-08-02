"""Auditable built-in tensor rewrite rules and their verification obligations."""

from __future__ import annotations

from .model import DType, RewriteKind, RewriteRule, SemanticCategory

ALL_DTYPES = tuple(DType)
INTEGER_DTYPES = (DType.INT64, DType.INT32, DType.INT8)
FLOAT_DTYPES = (DType.FLOAT64, DType.FLOAT32, DType.BFLOAT16, DType.FLOAT16)

BUILTIN_RULES = (
    RewriteRule(
        "tensor/redundant-cast/v1",
        RewriteKind.REDUNDANT_CAST,
        SemanticCategory.EXACT,
        ALL_DTYPES,
        0.0,
        False,
        ("shape_domain", "dtype_domain", "aliasing"),
    ),
    RewriteRule(
        "tensor/double-transpose/v1",
        RewriteKind.DOUBLE_TRANSPOSE,
        SemanticCategory.EXACT,
        ALL_DTYPES,
        0.0,
        False,
        ("shape_domain", "stride_domain", "aliasing"),
    ),
    RewriteRule(
        "tensor/add-zero-integer/v1",
        RewriteKind.ADD_ZERO,
        SemanticCategory.EXACT,
        INTEGER_DTYPES,
        0.0,
        False,
        ("integer_exactness", "shape_domain"),
    ),
    RewriteRule(
        "tensor/add-zero-float/v1",
        RewriteKind.ADD_ZERO,
        SemanticCategory.EXACT,
        FLOAT_DTYPES,
        0.0,
        True,
        ("signed_zero_contract", "nan_and_infinity", "hardware_mode"),
    ),
    RewriteRule(
        "tensor/mul-one/v1",
        RewriteKind.MUL_ONE,
        SemanticCategory.EXACT,
        ALL_DTYPES,
        0.0,
        False,
        ("nan_and_infinity", "denormal_mode", "shape_domain"),
    ),
    RewriteRule(
        "tensor/reassociate-add/v1",
        RewriteKind.REASSOCIATE_ADD,
        SemanticCategory.APPROXIMATE,
        FLOAT_DTYPES,
        1e-5,
        False,
        ("differential", "quality_budget", "nan_and_infinity", "determinism"),
    ),
    RewriteRule(
        "tensor/state-update-fusion/v1",
        RewriteKind.STATE_UPDATE_FUSION,
        SemanticCategory.EXACT,
        ALL_DTYPES,
        0.0,
        False,
        ("state_ownership", "aliasing", "cancellation", "rollback"),
    ),
    RewriteRule(
        "tensor/quantize-output-int8/v1",
        RewriteKind.QUANTIZE_OUTPUT,
        SemanticCategory.APPROXIMATE,
        FLOAT_DTYPES,
        0.01,
        False,
        ("quality_dataset", "calibration_split", "differential", "fallback"),
    ),
)
