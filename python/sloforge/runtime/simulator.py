from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sloforge.ir import DeploymentPlan
from sloforge.models.service_curve import CalibratedModels, CurvePoint
from sloforge.trace import TraceRequest
from sloforge.util import write_json


class SimulatorRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_output: dict[str, object]
    summary: dict[str, float | int]
    command: list[str]
    scenario_path: str
    chrome_trace_path: str


def _slope(points: Sequence[CurvePoint]) -> tuple[float, float]:
    first = points[0]
    last = points[-1]
    first_x = float(first.x)
    last_x = float(last.x)
    first_y = float(first.median_ms)
    last_y = float(last.median_ms)
    slope = max(0.0, (last_y - first_y) / max(last_x - first_x, 1e-9))
    intercept = max(0.0, first_y - first_x * slope)
    return intercept, slope


def replay_simulator(
    *,
    plan: DeploymentPlan,
    models: CalibratedModels,
    trace: list[TraceRequest],
    output_path: Path,
    chrome_trace_path: Path,
    repository_root: Path,
    seed: int,
    inject_required_faults: bool = True,
    control_actions: list[dict[str, object]] | None = None,
    routing_policy_override: str | None = None,
) -> SimulatorRun:
    if inject_required_faults and control_actions:
        raise ValueError("fault injection and controller actions must use separate simulator runs")
    candidate_id = plan.routing.targets[0].variant
    model = next(item for item in models.candidates if item.candidate_id == candidate_id)
    prefill_intercept, prefill_slope = _slope(model.prefill.points)
    decode_intercept, decode_slope = _slope(model.decode.points)
    replicas = [
        {
            "id": f"replica-{index}",
            "initially_warm": index < plan.cold_start.minimum_warm_replicas,
            "max_queue": plan.admission.queue_capacity,
            "max_active_sequences": plan.batching.maximum_active_sequences,
            "service_rate_multiplier": 1.0,
            "hourly_price_usd": plan.hardware.hourly_price_usd,
            "canary": index == plan.replica_topology.initial_replicas - 1 and plan.canary.enabled,
        }
        for index in range(plan.replica_topology.initial_replicas)
    ]
    if not replicas:
        raise ValueError("simulator requires at least one replica")
    start = int(trace[0].arrival_ms)
    end = int(trace[-1].arrival_ms)
    span = max(100, end - start)
    actions: list[dict[str, object]] = list(control_actions or [])
    if inject_required_faults:
        actions = [
            {
                "at_ms": start + int(span * 0.25),
                "action": {"type": "backend_slowdown", "replica_id": "replica-0", "factor": 2.8},
            },
            {
                "at_ms": start + int(span * 0.42),
                "action": {"type": "backend_crash", "replica_id": "replica-0"},
            },
            {
                "at_ms": start + int(span * 0.52),
                "action": {"type": "backend_recover", "replica_id": "replica-0", "warm": False},
            },
            {
                "at_ms": start + int(span * 0.65),
                "action": {"type": "startup_slowdown", "replica_id": "replica-0", "factor": 3.0},
            },
        ]
    requests = [
        {
            "id": item.request_id,
            "arrival_ms": round(item.arrival_ms),
            "prompt_tokens": item.prompt_tokens,
            "output_tokens": item.output_tokens,
            "priority": 10 - item.priority,
            "request_class": item.request_class,
            "deadline_ms": int(item.deadline_ms) if item.deadline_ms else None,
            "cancel_after_ms": int(item.cancelled_at_ms - item.arrival_ms)
            if item.cancelled_at_ms
            else None,
            "canary_eligible": True,
            "adapter_id": item.adapter_id,
            "prefix_group": item.prefix_group,
        }
        for item in trace
    ]
    scenario = {
        "schema_version": "1.0",
        "seed": seed,
        "service_curve": {
            "id": f"{candidate_id}/calibrated-v1",
            "measurement_artifact": plan.provenance.evidence_bundle_uri,
            "prefill_intercept_ms": prefill_intercept,
            "prefill_ms_per_prompt_token": prefill_slope,
            "prefill_ms_per_batch_item": 0.05,
            "chunk_overhead_ms": 0.1,
            "chunk_size_tokens": plan.engine.chunked_prefill.chunk_tokens
            if plan.engine.chunked_prefill.enabled
            else None,
            "decode_intercept_ms": decode_intercept,
            "decode_ms_per_active_sequence": decode_slope,
            "decode_ms_per_context_token": 0.0005,
            "startup": {
                "kind": "empirical",
                "samples_ms": [model.startup_median_ms, model.startup_p95_ms],
            },
        },
        "replicas": replicas,
        "requests": requests,
        "actions": actions,
        "routing_policy": routing_policy_override
        or {
            "round_robin": "round_robin",
            "least_outstanding": "least_outstanding",
            "earliest_finish": "earliest_finish",
            "slo_slack": "slo_slack_aware",
        }[plan.routing.kind.value],
        "canary_weight": plan.canary.initial_weight if plan.canary.enabled else 0.0,
        "max_events": 5_000_000,
    }
    scenario_path = output_path.with_name("simulator-scenario.json")
    write_json(scenario_path, scenario)
    binary = shutil.which("sloforge-sim")
    command = [binary] if binary else ["cargo", "run", "-q", "-p", "sloforge-sim", "--"]
    command += [
        "simulate",
        "--config",
        str(scenario_path),
        "--output",
        str(output_path.with_suffix(".raw.json")),
        "--chrome-trace",
        str(chrome_trace_path),
    ]
    completed = subprocess.run(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Rust simulator failed ({completed.returncode}): {completed.stderr}")
    raw = json.loads(output_path.with_suffix(".raw.json").read_text(encoding="utf-8"))
    metrics = raw["metrics"]
    request_count = int(metrics["request_count"])
    completed_count = int(metrics["completed_count"])
    deadline_misses = int(metrics["deadline_miss_count"])
    summary: dict[str, float | int] = {
        "request_count": request_count,
        "completed_count": completed_count,
        "deadline_miss_count": deadline_misses,
        "p95_ttft_ms": float(metrics["p95_ttft_ms"] or 0.0),
        "p99_itl_ms": float(metrics["p99_itl_ms"] or 0.0),
        "availability": float(metrics["availability"]),
        "cost_usd": float(metrics["cost_usd"]),
        "slo_attainment": max(0.0, (request_count - deadline_misses) / max(request_count, 1)),
        "processed_events": int(metrics["processed_events"]),
    }
    envelope = {
        "schema_version": "sloforge.replay/v1",
        "summary": summary,
        "simulation": raw,
        "command": command,
    }
    write_json(output_path, envelope)
    return SimulatorRun(
        raw_output=raw,
        summary=summary,
        command=command,
        scenario_path=str(scenario_path),
        chrome_trace_path=str(chrome_trace_path),
    )
