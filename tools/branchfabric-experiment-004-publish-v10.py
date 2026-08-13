#!/usr/bin/env python3
"""Publish plots, reports, and plans for one canonically successful v10 run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))


def main() -> int:
    from sloforge.helix.characterization.gpu_reclamation_v10_publish import (
        publish_v10_success,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--experiment-root",
        default=Path("artifacts/branchfabric/gpu-validation/experiment-004"),
        type=Path,
    )
    parser.add_argument("--repository-root", default=Path.cwd(), type=Path)
    args = parser.parse_args()
    manifest = publish_v10_success(
        repository_root=args.repository_root,
        run_root=args.run_root,
        experiment_root=args.experiment_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
