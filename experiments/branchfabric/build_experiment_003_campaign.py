"""Build deterministic, bounded Modal campaign inputs for Experiment 003."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from gpu_validation_campaign import CampaignConfig, plan_primary_pairs

Stage = Literal["baseline", "optimized"]
ORDER_SEED = 20_260_811
PRIMARY_SEEDS = (41, 73, 113)


def _trial(
    *,
    campaign_id: str,
    attempt_id: str,
    seed: int,
    fanout: int,
    mode: str,
    implementation: Stage,
    trace: str,
    pair_id: str | None,
    pair_order_position: int | None,
    pair_order_rule: str | None,
    pair_order_seed: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": "sloforge.branchfabric.modal-metadata-characterization-config/v1",
        "attempt_id": attempt_id,
        "campaign_id": campaign_id,
        "pair_id": pair_id,
        "pair_order_position": pair_order_position,
        "pair_order_rule": pair_order_rule,
        "pair_order_seed": pair_order_seed,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "tokenizer_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "runtime": "vllm",
        "runtime_version": "0.23.0",
        "fanout": fanout,
        "prefix_length": 16_384,
        "suffix_length": 256,
        "seed": seed,
        "baseline_mode": mode,
        "implementation": implementation,
        "tracing_level": trace,
        "maximum_wall_seconds": 240,
        "initialization_timeout_seconds": 300,
        "cleanup_timeout_seconds": 60,
        "gpu_memory_utilization": 0.8,
        "gpu_sample_interval_seconds": 1.0,
    }


def build_campaign(stage: Stage, *, campaign_id: str) -> dict[str, Any]:
    paired = CampaignConfig(
        campaign_id=campaign_id,
        randomized_order_seed=ORDER_SEED,
    )
    trials: list[dict[str, Any]] = []
    for pair in plan_primary_pairs(paired):
        for position, mode in enumerate(pair.realized_order, start=1):
            trials.append(
                _trial(
                    campaign_id=campaign_id,
                    attempt_id=(
                        f"exp003-{stage}-f8-s{pair.seed}-p{position}-"
                        f"{'ind' if mode == 'independent' else 'shared'}"
                    ),
                    seed=pair.seed,
                    fanout=8,
                    mode=mode,
                    implementation=stage,
                    trace="minimal",
                    pair_id=pair.pair_id,
                    pair_order_position=position,
                    pair_order_rule=pair.order_rule,
                    pair_order_seed=pair.order_seed,
                )
            )
    for fanout in (1, 16, 32):
        trials.append(
            _trial(
                campaign_id=campaign_id,
                attempt_id=f"exp003-{stage}-scaling-f{fanout}-s149",
                seed=149,
                fanout=fanout,
                mode="shared_root",
                implementation=stage,
                trace="minimal",
                pair_id=None,
                pair_order_position=None,
                pair_order_rule=None,
                pair_order_seed=None,
            )
        )
    # Keep the diagnostic controls in the optimized campaign too.  The first
    # bounded baseline campaign may use its tail on scaling, and overhead must
    # be measured on one internally consistent instrumented image.
    for trace in ("disabled", "minimal", "full"):
        trials.append(
            _trial(
                campaign_id=campaign_id,
                attempt_id=f"exp003-{stage}-trace-{trace}-f8-s191",
                seed=191,
                fanout=8,
                mode="shared_root",
                implementation=stage,
                trace=trace,
                pair_id=None,
                pair_order_position=None,
                pair_order_rule="instrumentation_control",
                pair_order_seed=None,
            )
        )
    return {
        "schema_version": "sloforge.branchfabric.modal-campaign-003/v1",
        "campaign_id": campaign_id,
        "maximum_campaign_seconds": 1800 if stage == "baseline" else 2300,
        "trials": trials,
    }


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable campaign input: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            build_campaign(args.stage, campaign_id=args.campaign_id),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ORDER_SEED", "PRIMARY_SEEDS", "build_campaign"]
