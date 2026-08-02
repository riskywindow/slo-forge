from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_kernel_module() -> ModuleType:
    path = Path(__file__).parents[2] / "kernels" / "fused_logits.py"
    specification = importlib.util.spec_from_file_location("sloforge_fused_logits", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load kernel experiment from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Opt-in Triton fused logits preprocessing correctness and timing experiment"
    )
    parser.add_argument("--enable-triton-experiment", action="store_true")
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocabulary", type=int, default=32_000)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seen-probability", type=float, default=0.05)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if not arguments.enable_triton_experiment:
        raise SystemExit(
            "refusing to run: pass --enable-triton-experiment; this kernel is never enabled by default"
        )
    kernel: Any = _load_kernel_module()
    try:
        kernel.run_randomized_correctness(
            seed=arguments.seed,
            device=arguments.device,
            device_index=arguments.device_index,
        )
        result = kernel.benchmark_fused_logits(
            batch=arguments.batch,
            vocabulary=arguments.vocabulary,
            dtype_name=arguments.dtype,
            temperature=arguments.temperature,
            repetition_penalty=arguments.repetition_penalty,
            seen_probability=arguments.seen_probability,
            warmups=arguments.warmups,
            samples=arguments.samples,
            seed=arguments.seed,
            enable_triton_experiment=True,
            device=arguments.device,
            device_index=arguments.device_index,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
