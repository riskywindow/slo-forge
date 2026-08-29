#!/usr/bin/env python3
"""Execute one bounded BranchFabric vLLM GPU-validation invocation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from gpu_validation_runner import (
    VllmGpuValidationInvocation,
    build_vllm_adapter_factory,
    run_experiment,
)

_MAX_INVOCATION_BYTES = 16 * 1024 * 1024


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--invocation",
        required=True,
        type=Path,
        help="fully materialized, versioned invocation JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new output directory; existing paths are rejected",
    )
    return parser.parse_args(argv)


def _read_invocation(path: Path) -> VllmGpuValidationInvocation:
    size = path.stat().st_size
    if size > _MAX_INVOCATION_BYTES:
        raise ValueError(f"invocation is {size} bytes; limit is {_MAX_INVOCATION_BYTES} bytes")
    return VllmGpuValidationInvocation.model_validate_json(path.read_bytes())


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    invocation = _read_invocation(args.invocation)
    result = run_experiment(
        invocation.experiment,
        build_vllm_adapter_factory(invocation),
        args.output,
        launcher_configuration=invocation,
    )
    print(
        json.dumps(
            result.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
