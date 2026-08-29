#!/usr/bin/env python3
"""Bounded CUDA-owning peer-capability probe for Experiment 004."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-gpu-uuid",
        required=True,
        action="append",
        dest="expected_gpu_uuids",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    expected = tuple(str(item) for item in args.expected_gpu_uuids)
    if len(expected) != 2 or len(set(expected)) != 2:
        raise ValueError("peer probe requires exactly two distinct physical GPU UUIDs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != ",".join(expected):
        raise RuntimeError("peer probe CUDA visibility differs from the exact physical GPU pair")

    # This fresh child is the only preflight process allowed to import torch.
    import torch  # type: ignore[import-not-found]

    from sloforge.helix.characterization.gpu_reclamation_instrumentation import (
        capture_cuda_peer_access,
    )

    matrix = capture_cuda_peer_access(expected_gpu_count=2, torch_module=torch)
    payload = {
        "schema_version": "sloforge.branchfabric.experiment-004-cuda-peer-probe/v1",
        "expected_physical_gpu_uuids": expected,
        "cuda_visible_devices": visible,
        "torch_cuda_version": torch.version.cuda,
        "matrix": matrix.model_dump(mode="json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(
            (
                json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
            ).encode("utf-8")
        )
        handle.flush()
        os.fsync(handle.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
