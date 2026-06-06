"""Deterministic causal-hypothesis generation from differential evidence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .models import (
    AlignmentQuality,
    AutopsyRun,
    BottleneckKind,
    CausalHypothesis,
    DiagnosisRecord,
    DifferentialComparison,
    EventType,
    EvidenceStatement,
)


@dataclass(frozen=True)
class SignalRule:
    kind: BottleneckKind
    signal: str
    threshold: float
    relation: str
    target: str
    causal_specificity: float = 0.0


RULES: tuple[SignalRule, ...] = (
    SignalRule(
        BottleneckKind.ARRIVAL_OVERLOAD, "arrival_capacity_ratio", 1.0, "greater", "deployment"
    ),
    SignalRule(BottleneckKind.GATEWAY_QUEUEING, "gateway_queue_delta", 0.20, "greater", "gateway"),
    SignalRule(
        BottleneckKind.BACKEND_QUEUEING, "backend_queue_delta", 0.20, "greater", "backend_pool"
    ),
    SignalRule(
        BottleneckKind.INSUFFICIENT_WARM_CAPACITY,
        "warm_fraction_drop",
        0.15,
        "greater",
        "worker_pool",
    ),
    SignalRule(
        BottleneckKind.COLD_START_REGRESSION, "startup_delta", 0.20, "greater", "worker_startup"
    ),
    SignalRule(
        BottleneckKind.MODEL_LOADING_REGRESSION, "model_load_delta", 0.20, "greater", "model_loader"
    ),
    SignalRule(
        BottleneckKind.CPU_LAUNCH_BOTTLENECK, "cpu_launch_delta", 0.20, "greater", "cpu_launch"
    ),
    SignalRule(
        BottleneckKind.EXCESSIVE_KERNEL_LAUNCHES,
        "kernel_count_delta",
        0.20,
        "greater",
        "gpu_runtime",
    ),
    SignalRule(
        BottleneckKind.GPU_COMPUTE_REGRESSION, "gpu_compute_delta", 0.20, "greater", "gpu_compute"
    ),
    SignalRule(
        BottleneckKind.GPU_MEMORY_BANDWIDTH_REGRESSION,
        "gpu_memory_delta",
        0.20,
        "greater",
        "gpu_hbm",
    ),
    SignalRule(BottleneckKind.GPU_CLOCK_THROTTLING, "gpu_clock_drop", 0.10, "greater", "gpu_clock"),
    SignalRule(
        BottleneckKind.NUMA_MISPLACEMENT, "numa_remote_delta", 0.15, "greater", "numa_affinity"
    ),
    SignalRule(BottleneckKind.PCIE_BOTTLENECK, "pcie_delta", 0.20, "greater", "pcie_path"),
    SignalRule(BottleneckKind.NVLINK_DEGRADATION, "nvlink_delta", 0.20, "greater", "nvlink_path"),
    SignalRule(
        BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
        "network_bandwidth_drop",
        0.15,
        "greater",
        "network_rail",
        0.08,
    ),
    SignalRule(
        BottleneckKind.NETWORK_LATENCY_DEGRADATION,
        "network_latency_delta",
        0.20,
        "greater",
        "network_rail",
    ),
    SignalRule(
        BottleneckKind.RANK_STRAGGLER,
        "rank_skew",
        1.20,
        "greater",
        "rank_group",
        0.03,
    ),
    SignalRule(
        BottleneckKind.COLLECTIVE_IMBALANCE,
        "collective_wait_delta",
        0.20,
        "greater",
        "collective_group",
    ),
    SignalRule(
        BottleneckKind.COLLECTIVE_ALGORITHM_REGRESSION,
        "collective_delta",
        0.20,
        "greater",
        "collective_algorithm",
    ),
    SignalRule(
        BottleneckKind.EXPERT_LOAD_IMBALANCE, "expert_imbalance", 0.15, "greater", "expert_group"
    ),
    SignalRule(
        BottleneckKind.PREFILL_POOL_SATURATION, "prefill_delta", 0.20, "greater", "prefill_pool"
    ),
    SignalRule(
        BottleneckKind.DECODE_POOL_SATURATION, "decode_delta", 0.20, "greater", "decode_pool"
    ),
    SignalRule(
        BottleneckKind.KV_TRANSFER_BOTTLENECK, "kv_transfer_delta", 0.20, "greater", "kv_transfer"
    ),
    SignalRule(BottleneckKind.UNHEALTHY_WORKER, "unhealthy_worker", 0.5, "greater", "worker"),
    SignalRule(BottleneckKind.WORKER_CRASH, "worker_crash", 0.5, "greater", "worker"),
    SignalRule(
        BottleneckKind.TOPOLOGY_MISMATCH, "topology_mismatch", 0.5, "greater", "physical_plan"
    ),
    SignalRule(
        BottleneckKind.INVALID_PHYSICAL_PLAN,
        "invalid_plan_assumption",
        0.5,
        "greater",
        "physical_plan",
    ),
)


def _stage_signal(comparison: DifferentialComparison, event_types: set[EventType]) -> float:
    values = [
        item.relative_delta
        for item in comparison.stage_deltas
        if item.event_type in event_types and item.matched_count > 0
    ]
    return max(values, default=0.0)


def _counter_map(comparison: DifferentialComparison) -> dict[str, float]:
    counters: dict[str, tuple[str, float]] = {}
    for item in comparison.counter_deltas:
        previous = counters.get(item.name)
        if previous is not None and previous[0] != item.unit:
            raise ValueError(f"counter {item.name!r} has conflicting units in comparison")
        counters[item.name] = (item.unit, item.relative_delta)
    return {name: value for name, (_, value) in counters.items()}


_SIGNAL_EVENT_TYPES: dict[str, frozenset[EventType]] = {
    "gateway_queue_delta": frozenset({EventType.GATEWAY_QUEUE}),
    "backend_queue_delta": frozenset({EventType.BACKEND_QUEUE}),
    "startup_delta": frozenset({EventType.STARTUP, EventType.READINESS}),
    "cpu_launch_delta": frozenset({EventType.CPU_LAUNCH}),
    "gpu_compute_delta": frozenset({EventType.GPU_COMPUTE}),
    "gpu_memory_delta": frozenset({EventType.GPU_MEMORY}),
    "pcie_delta": frozenset({EventType.PCIE_TRANSFER}),
    "nvlink_delta": frozenset({EventType.NVLINK_TRANSFER}),
    "network_latency_delta": frozenset({EventType.NETWORK_TRANSFER}),
    "collective_wait_delta": frozenset({EventType.COLLECTIVE_WAIT}),
    "collective_delta": frozenset({EventType.COLLECTIVE}),
    "expert_imbalance": frozenset({EventType.EXPERT_DISPATCH, EventType.EXPERT_COMBINE}),
    "prefill_delta": frozenset({EventType.PREFILL}),
    "decode_delta": frozenset({EventType.DECODE}),
    "kv_transfer_delta": frozenset({EventType.KV_TRANSFER}),
}

_SIGNAL_COUNTERS: dict[str, frozenset[str]] = {
    "arrival_capacity_ratio": frozenset({"arrival_capacity_ratio"}),
    "warm_fraction_drop": frozenset({"warm_fraction"}),
    "model_load_delta": frozenset({"model_load_time_ms"}),
    "kernel_count_delta": frozenset({"kernel_count"}),
    "gpu_clock_drop": frozenset({"gpu_clock_mhz"}),
    "numa_remote_delta": frozenset({"numa_remote_fraction"}),
    "network_bandwidth_drop": frozenset({"network_bandwidth_gbps"}),
    "network_latency_delta": frozenset({"network_latency_us"}),
    "expert_imbalance": frozenset({"expert_token_cv"}),
    "unhealthy_worker": frozenset({"worker_unhealthy"}),
    "worker_crash": frozenset({"worker_crash"}),
    "topology_mismatch": frozenset({"topology_mismatch"}),
    "invalid_plan_assumption": frozenset({"plan_assumption_error"}),
}


def _signal_sample_count(comparison: DifferentialComparison, signal: str) -> int:
    if signal == "rank_skew":
        return sum(item.matched_count for item in comparison.stage_deltas if item.rank is not None)
    event_types = _SIGNAL_EVENT_TYPES.get(signal, frozenset())
    stage_count = sum(
        item.matched_count for item in comparison.stage_deltas if item.event_type in event_types
    )
    counter_names = _SIGNAL_COUNTERS.get(signal, frozenset())
    counter_count = sum(
        max(1, min(item.healthy_count, item.degraded_count))
        for item in comparison.counter_deltas
        if item.name in counter_names
    )
    return stage_count + counter_count


def extract_signals(comparison: DifferentialComparison) -> dict[str, float]:
    counters = _counter_map(comparison)
    return {
        "arrival_capacity_ratio": counters.get("arrival_capacity_ratio", 0.0) + 1.0,
        "gateway_queue_delta": _stage_signal(comparison, {EventType.GATEWAY_QUEUE}),
        "backend_queue_delta": _stage_signal(comparison, {EventType.BACKEND_QUEUE}),
        "warm_fraction_drop": max(0.0, -counters.get("warm_fraction", 0.0)),
        "startup_delta": _stage_signal(comparison, {EventType.STARTUP, EventType.READINESS}),
        "model_load_delta": counters.get("model_load_time_ms", 0.0),
        "cpu_launch_delta": _stage_signal(comparison, {EventType.CPU_LAUNCH}),
        "kernel_count_delta": counters.get("kernel_count", 0.0),
        "gpu_compute_delta": _stage_signal(comparison, {EventType.GPU_COMPUTE}),
        "gpu_memory_delta": _stage_signal(comparison, {EventType.GPU_MEMORY}),
        "gpu_clock_drop": max(0.0, -counters.get("gpu_clock_mhz", 0.0)),
        "numa_remote_delta": counters.get("numa_remote_fraction", 0.0),
        "pcie_delta": _stage_signal(comparison, {EventType.PCIE_TRANSFER}),
        "nvlink_delta": _stage_signal(comparison, {EventType.NVLINK_TRANSFER}),
        "network_bandwidth_drop": max(0.0, -counters.get("network_bandwidth_gbps", 0.0)),
        "network_latency_delta": max(
            _stage_signal(comparison, {EventType.NETWORK_TRANSFER}),
            counters.get("network_latency_us", 0.0),
        ),
        "rank_skew": comparison.maximum_rank_skew,
        "collective_wait_delta": _stage_signal(comparison, {EventType.COLLECTIVE_WAIT}),
        "collective_delta": _stage_signal(comparison, {EventType.COLLECTIVE}),
        "expert_imbalance": max(
            counters.get("expert_token_cv", 0.0),
            _stage_signal(comparison, {EventType.EXPERT_DISPATCH, EventType.EXPERT_COMBINE}),
        ),
        "prefill_delta": _stage_signal(comparison, {EventType.PREFILL}),
        "decode_delta": _stage_signal(comparison, {EventType.DECODE}),
        "kv_transfer_delta": _stage_signal(comparison, {EventType.KV_TRANSFER}),
        "unhealthy_worker": max(0.0, counters.get("worker_unhealthy", 0.0)),
        "worker_crash": max(0.0, counters.get("worker_crash", 0.0)),
        "topology_mismatch": max(0.0, counters.get("topology_mismatch", 0.0)),
        "invalid_plan_assumption": max(0.0, counters.get("plan_assumption_error", 0.0)),
    }


def _confidence(
    observed: float, threshold: float, matched: int, causal_specificity: float
) -> float:
    if observed <= threshold:
        return max(0.01, min(0.20, observed / max(threshold, 1e-9) * 0.20))
    effect = min(1.0, (observed - threshold) / max(threshold, 0.1))
    sample_factor = min(1.0, math.log2(matched + 1) / 5.0)
    # Direct physical-counter changes are more causally specific than downstream
    # stage delays, which can be propagation effects. Keep this bounded so a
    # counter cannot overwhelm contradictory evidence or sample quality.
    return min(0.99, 0.45 + 0.30 * effect + 0.12 * sample_factor + causal_specificity)


def diagnose(
    degraded: AutopsyRun,
    *,
    comparison: DifferentialComparison,
    baseline: AutopsyRun | None = None,
) -> DiagnosisRecord:
    if comparison.degraded_run_id != degraded.run_id:
        raise ValueError("comparison does not reference the degraded run")
    if baseline is not None and comparison.healthy_run_id != baseline.run_id:
        raise ValueError("comparison does not reference the supplied baseline")
    signals = extract_signals(comparison)
    event_ids = (
        (comparison.first_divergence_event_id,)
        if comparison.first_divergence_event_id is not None
        else ()
    )
    hypotheses: list[CausalHypothesis] = []
    for index, rule in enumerate(RULES):
        observed = signals[rule.signal]
        supports = observed > rule.threshold
        statement = EvidenceStatement(
            metric=rule.signal,
            observed=observed,
            threshold=rule.threshold,
            relation="greater_than",
            supports_hypothesis=supports,
            event_ids=event_ids,
            explanation=(
                f"{rule.signal}={observed:.6g} exceeds {rule.threshold:.6g}"
                if supports
                else f"{rule.signal}={observed:.6g} does not exceed {rule.threshold:.6g}"
            ),
        )
        confidence = _confidence(
            observed,
            rule.threshold,
            _signal_sample_count(comparison, rule.signal),
            rule.causal_specificity,
        )
        hypotheses.append(
            CausalHypothesis(
                hypothesis_id=f"hypothesis-{index:02d}-{rule.kind.value}",
                kind=rule.kind,
                target=rule.target,
                supporting_evidence=(statement,) if supports else (),
                contradicting_evidence=() if supports else (statement,),
                confidence=confidence,
                rejected_reason=None if supports else "primary signal did not cross its threshold",
            )
        )
    hypotheses.sort(
        key=lambda item: (item.rejected_reason is not None, -item.confidence, item.kind.value)
    )
    ranked = tuple(item.kind for item in hypotheses[:3])
    top = hypotheses[0]
    runs = (degraded,) if baseline is None else (degraded, baseline)
    alignments = tuple(estimate for run in runs for estimate in run.alignments)
    event_hosts = {event.host for event in degraded.events}
    if baseline is not None:
        event_hosts.update(event.host for event in baseline.events)
    complete_alignment = all(
        {event.host for event in run.events}.issubset(
            {estimate.host for estimate in run.alignments}
        )
        for run in runs
    )
    sufficient_alignment = len(event_hosts) <= 1 or (
        complete_alignment
        and all(estimate.quality is not AlignmentQuality.INSUFFICIENT for estimate in alignments)
    )
    warnings = list(comparison.warnings)
    if not sufficient_alignment:
        warnings.append("diagnosis confidence excludes fine-grained cross-host ordering")
    if not any(hypothesis.supporting_evidence for hypothesis in hypotheses):
        warnings.append("no causal hypothesis crossed its evidence threshold")
    warnings.append(
        "diagnosis confidence is a deterministic evidence score, not a calibrated probability"
    )
    identifier = hashlib.sha256(
        f"{degraded.run_id}\0{comparison.comparison_id}\0{top.kind.value}".encode()
    ).hexdigest()[:16]
    evidence = tuple(
        dict.fromkeys((*degraded.artifacts, *(() if baseline is None else baseline.artifacts)))
    )
    return DiagnosisRecord(
        diagnosis_id=f"diagnosis-{identifier}",
        degraded_run_id=degraded.run_id,
        baseline_run_id=baseline.run_id if baseline is not None else None,
        comparison_id=comparison.comparison_id,
        top_hypothesis=top.kind,
        top_three=ranked,
        hypotheses=tuple(hypotheses),
        first_divergence_event_id=comparison.first_divergence_event_id,
        first_divergence_ns=comparison.first_divergence_ns,
        confidence=top.confidence,
        sufficient_alignment=sufficient_alignment,
        warnings=tuple(warnings),
        evidence=evidence,
    )
