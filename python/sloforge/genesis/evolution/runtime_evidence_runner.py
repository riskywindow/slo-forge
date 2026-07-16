"""Trusted bounded runner for one extracted, untrusted Genesis runtime bundle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType


def _load_runtime(bundle_root: Path) -> ModuleType:
    runtime_path = bundle_root / "runtime.py"
    spec = importlib.util.spec_from_file_location(
        "genesis_evolution_candidate_runtime", runtime_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate runtime module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(bundle_root))
    code = compile(runtime_path.read_bytes(), str(runtime_path), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


def _write_once(path: Path, value: object) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")


def _generation_seed(bundle_root: Path) -> int:
    config = json.loads((bundle_root / "runtime_config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("generated runtime configuration must be an object")
    value = config.get("generation_seed")
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 1 << 64:
        raise ValueError("generated runtime configuration has an invalid generation seed")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    arguments = parser.parse_args(argv)
    trace = json.loads(arguments.trace.read_text(encoding="utf-8"))
    if not isinstance(trace, dict) or not isinstance(trace.get("requests"), list):
        raise ValueError("evolution trace must contain a bounded requests list")
    if not 1 <= len(trace["requests"]) <= 256:
        raise ValueError("evolution trace request count must be in [1, 256]")

    runtime_seed = _generation_seed(arguments.bundle)
    module = _load_runtime(arguments.bundle)
    application = module.application(seed=runtime_seed)
    cases: list[dict[str, object]] = []
    application.start()
    try:
        for request in trace["requests"]:
            if not isinstance(request, dict):
                raise TypeError("trace request must be an object")
            request_id = str(request["request_id"])
            started_ns = time.perf_counter_ns()
            first_token_ns: int | None = None
            token_ids: list[int] = []
            error: str | None = None
            try:
                handle = application.runtime.submit_text(
                    request_id=request_id,
                    text=str(request["text"]),
                    maximum_new_tokens=int(request["maximum_new_tokens"]),
                    seed=int(request["seed"]),
                    timeout_seconds=arguments.timeout_seconds,
                    batching_eligible=bool(request.get("batching_eligible", True)),
                )
                for event in handle.events(arguments.timeout_seconds):
                    if event.token_id is not None:
                        if first_token_ns is None:
                            first_token_ns = time.perf_counter_ns()
                        token_ids.append(int(event.token_id))
            except (RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            completed_ns = time.perf_counter_ns()
            elapsed_ns = max(1, completed_ns - started_ns)
            ttft_ns = max(1, (first_token_ns or completed_ns) - started_ns)
            cases.append(
                {
                    "request_id": request_id,
                    "request_seed": int(request["seed"]),
                    "token_ids": token_ids,
                    "token_count": len(token_ids),
                    "ttft_ns": ttft_ns,
                    "mean_tpot_ns": max(1, elapsed_ns // max(1, len(token_ids))),
                    "completion_ns": elapsed_ns,
                    "error": error,
                }
            )
    finally:
        application.shutdown()
    _write_once(
        arguments.output,
        {
            "schema_version": "sloforge.genesis.evolution.runtime-observation/v1",
            "seed": arguments.seed,
            "runtime_seed": runtime_seed,
            "request_count": len(cases),
            "cases": cases,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
