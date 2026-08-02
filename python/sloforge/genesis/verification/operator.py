"""Schema-aware differential and adversarial operator verification."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .model import (
    EvidenceStatus,
    OperatorContract,
    OperatorCounterexample,
    OperatorVerificationResult,
    VerificationError,
    VerificationLevel,
)

Array = npt.NDArray[np.generic]
Operator = Callable[[Array], Array]

_DTYPES: dict[str, npt.DTypeLike] = {
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "int8": np.int8,
    "bool": np.bool_,
}


def _digest(array: Array) -> str:
    value = np.ascontiguousarray(array)
    header = f"{value.dtype}:{value.shape}".encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def _error_digest(error: Exception) -> str:
    payload = f"{type(error).__name__}:{error}".encode()
    return hashlib.sha256(payload).hexdigest()


def _shape(
    contract: OperatorContract, index: int, generator: np.random.Generator
) -> tuple[int, ...]:
    result: list[int] = []
    for dimension_index, bound in enumerate(contract.shape):
        if bound.minimum <= 0 or bound.maximum < bound.minimum:
            raise VerificationError("shape bounds must be positive and ordered")
        selector = (index + dimension_index) % 3
        if selector == 0:
            result.append(bound.minimum)
        elif selector == 1:
            result.append(bound.maximum)
        else:
            result.append(int(generator.integers(bound.minimum, bound.maximum + 1)))
    return tuple(result)


def _values(
    shape: tuple[int, ...], dtype_name: str, index: int, generator: np.random.Generator
) -> Array:
    try:
        dtype = np.dtype(_DTYPES[dtype_name])
    except KeyError as error:
        raise VerificationError(f"unsupported verifier dtype {dtype_name!r}") from error
    if np.issubdtype(dtype, np.bool_):
        value = generator.integers(0, 2, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        value = generator.integers(-8, 9, size=shape).astype(dtype)
        flat = value.reshape(-1)
        if flat.size and index % 5 == 0:
            flat[0] = np.iinfo(dtype).min
        if flat.size > 1 and index % 5 == 0:
            flat[1] = np.iinfo(dtype).max
    else:
        value = generator.normal(0.0, 3.0, size=shape).astype(dtype)
        flat = value.reshape(-1)
        if flat.size and index % 7 == 0:
            flat[0] = np.nan
        if flat.size > 1 and index % 11 == 0:
            flat[1] = np.inf
        if flat.size > 2 and index % 13 == 0:
            flat[2] = -0.0
        limits = np.finfo(dtype)  # type: ignore[arg-type]
        if flat.size > 3 and index % 5 == 0:
            flat[3] = limits.max
        if flat.size > 4 and index % 5 == 0:
            flat[4] = limits.smallest_subnormal
    return value


def _non_contiguous(value: Array) -> Array:
    expanded_shape = (*value.shape[:-1], value.shape[-1] * 2)
    expanded = np.empty(expanded_shape, dtype=value.dtype)
    expanded[..., ::2] = value
    expanded[..., 1::2] = value
    result = expanded[..., ::2]
    if result.flags.c_contiguous:
        raise VerificationError("failed to construct non-contiguous verifier input")
    return result


def _compare(expected: Array, observed: Array, contract: OperatorContract) -> str | None:
    if expected.shape != observed.shape:
        return "output_shape"
    if expected.dtype != observed.dtype:
        return "output_dtype"
    if contract.preserve_nan and not np.array_equal(np.isnan(expected), np.isnan(observed)):
        return "nan_behavior"
    if contract.preserve_infinity and not np.array_equal(np.isinf(expected), np.isinf(observed)):
        return "infinity_behavior"
    if contract.preserve_signed_zero and np.issubdtype(expected.dtype, np.floating):
        expected_zero = expected == 0
        observed_zero = observed == 0
        shared_zero = expected_zero & observed_zero
        if not np.array_equal(
            np.signbit(expected)[shared_zero], np.signbit(observed)[shared_zero]
        ):
            return "signed_zero_behavior"
    if contract.exact:
        if not np.array_equal(expected, observed, equal_nan=True):
            return "exact_value"
    elif not np.allclose(
        expected,
        observed,
        atol=contract.maximum_absolute_error,
        rtol=contract.maximum_relative_error,
        equal_nan=True,
    ):
        return "numerical_tolerance"
    return None


def _minimized_values(value: Array, *, limit: int = 32) -> tuple[float | int | bool, ...]:
    result: list[float | int | bool] = []
    for item in value.reshape(-1)[:limit]:
        scalar = item.item()
        if isinstance(scalar, (bool, int, float)):
            result.append(scalar)
        else:
            result.append(float(scalar))
    return tuple(result)


def verify_operator(
    reference: Operator,
    candidate: Operator,
    contract: OperatorContract,
    *,
    seed: int,
) -> OperatorVerificationResult:
    if not contract.operator_id or not contract.shape or not contract.dtypes:
        raise VerificationError("operator identity, shape and dtype domains are required")
    if not 1 <= contract.maximum_cases <= 100_000:
        raise VerificationError("maximum_cases must be in [1, 100000]")
    if seed < 0:
        raise VerificationError("operator verification seed must be non-negative")
    if contract.maximum_absolute_error < 0 or contract.maximum_relative_error < 0:
        raise VerificationError("numerical tolerances must be non-negative")
    generator = np.random.default_rng(seed)
    executed = 0
    for index in range(contract.maximum_cases):
        dtype = contract.dtypes[index % len(contract.dtypes)]
        shape = _shape(contract, index, generator)
        value = _values(shape, dtype, index, generator)
        variant = "non_contiguous" if contract.allow_non_contiguous and index % 2 else "contiguous"
        if variant == "non_contiguous":
            value = _non_contiguous(value)
        original = value.copy()
        try:
            expected = np.asarray(reference(value.copy()))
        except Exception as error:
            raise VerificationError(
                f"reference implementation failed inside its declared domain: "
                f"{type(error).__name__}: {error}"
            ) from error
        try:
            observed = np.asarray(candidate(value))
        except Exception as error:
            executed += 1
            exception_violation = f"candidate_exception:{type(error).__name__}"
            if not contract.allow_aliasing and not np.array_equal(value, original, equal_nan=True):
                exception_violation = "unexpected_input_mutation_before_exception"
            return OperatorVerificationResult(
                claim=f"operator:{contract.operator_id}",
                level=VerificationLevel.PROPERTY,
                status=EvidenceStatus.FAILED,
                seed=seed,
                cases_executed=executed,
                domain=contract,
                counterexample=OperatorCounterexample(
                    case_index=index,
                    seed=seed,
                    shape=shape,
                    dtype=dtype,
                    stride_variant=variant,
                    violation=exception_violation,
                    expected_digest=_digest(expected),
                    observed_digest=_error_digest(error),
                    minimized_values=_minimized_values(original),
                ),
                assumptions=("numpy_reference", "bounded_generated_domain"),
            )
        executed += 1
        violation = _compare(expected, observed, contract)
        if not contract.allow_aliasing and not np.array_equal(value, original, equal_nan=True):
            violation = violation or "unexpected_input_mutation"
        if contract.deterministic:
            try:
                repeated = np.asarray(candidate(original.copy()))
            except Exception:
                violation = violation or "nondeterministic_exception"
            else:
                if _compare(observed, repeated, contract) is not None:
                    violation = violation or "nondeterministic_output"
        if violation is not None:
            return OperatorVerificationResult(
                claim=f"operator:{contract.operator_id}",
                level=VerificationLevel.PROPERTY,
                status=EvidenceStatus.FAILED,
                seed=seed,
                cases_executed=executed,
                domain=contract,
                counterexample=OperatorCounterexample(
                    case_index=index,
                    seed=seed,
                    shape=shape,
                    dtype=dtype,
                    stride_variant=variant,
                    violation=violation,
                    expected_digest=_digest(expected),
                    observed_digest=_digest(observed),
                    minimized_values=_minimized_values(original),
                ),
                assumptions=("numpy_reference", "bounded_generated_domain"),
            )
    return OperatorVerificationResult(
        claim=f"operator:{contract.operator_id}",
        level=VerificationLevel.PROPERTY,
        status=EvidenceStatus.PASSED,
        seed=seed,
        cases_executed=executed,
        domain=contract,
        counterexample=None,
        assumptions=("numpy_reference", "bounded_generated_domain"),
    )
