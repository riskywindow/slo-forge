"""Executable unsafe candidate used to prove the red-team rejection path."""

from __future__ import annotations

from sloforge.genesis.ir import Precision, RequestEventCase

from .models import (
    BenchmarkComparison,
    BenchmarkRunManifest,
    CacheRegime,
    RedTeamSurface,
    ResourceAdversarialCase,
    ScheduleAdversarialCase,
    TargetDescriptor,
    TensorAdversarialCase,
    TimerKind,
    TopologyAdversarialCase,
    ViolationObservation,
)


def _digest(character: str) -> str:
    return character * 64


class UnsafeStreamingCandidate:
    """A fast-path candidate with independently observable contract defects.

    This fixture models implementation mistakes that an optimizer can plausibly
    introduce: implicit contiguous loads, eager cancellation release, static
    route selection, and an off-by-one queue admission check.
    """

    descriptor = TargetDescriptor(
        candidate_id="unsafe-fastpath-v1",
        transformation_id="cross-layer-fastpath-v1",
        description="contiguous tensor fast path with eager cancellation and static routing",
        queue_capacity=8,
        device_capacity_bytes=4096,
        host_capacity_bytes=8192,
        process_limit=4,
        selected_links=("nvlink-0-1",),
    )

    def evaluate_tensor(self, case: TensorAdversarialCase) -> ViolationObservation | None:
        contiguous = _contiguous_strides(case.input.shape)
        if case.input.non_contiguous or case.input.strides != contiguous:
            return ViolationObservation(
                surface=RedTeamSurface.TENSOR,
                violated_contract="tensor.stride_semantics",
                expected_behavior="operator honors every declared legal tensor stride",
                observed_behavior="fast path indexes storage as though the tensor were contiguous",
                learned_precondition="tensor.non_contiguous == false and strides == contiguous(shape)",
            )
        return None

    def evaluate_schedule(self, case: ScheduleAdversarialCase) -> ViolationObservation | None:
        allocated: set[str] = set()
        externally_visible: set[str] = set()
        eagerly_released: set[str] = set()
        for event in case.events:
            if event.action == "admit":
                allocated.add(event.request_id)
            elif event.action == "emit" and event.request_id in allocated:
                externally_visible.add(event.request_id)
            elif event.action in {"cancel", "disconnect"} and event.request_id in allocated:
                allocated.remove(event.request_id)
                eagerly_released.add(event.request_id)
            elif (
                event.action == "retry"
                and event.request_id in externally_visible
                and event.request_id in eagerly_released
            ):
                return ViolationObservation(
                    surface=RedTeamSurface.PROTOCOL,
                    violated_contract="stream.retry_after_visible_output",
                    expected_behavior=(
                        "a request with externally visible output is never retried after its state is released"
                    ),
                    observed_behavior=(
                        "retry reads state that cancellation eagerly released after a committed token"
                    ),
                    learned_precondition=(
                        "retry requires externally_visible == false or retained_idempotent_state == true"
                    ),
                )
        return None

    def evaluate_topology(self, case: TopologyAdversarialCase) -> ViolationObservation | None:
        failed = set(case.topology.failed_links)
        selected = set(self.descriptor.selected_links)
        if failed & selected:
            return ViolationObservation(
                surface=RedTeamSurface.TOPOLOGY,
                violated_contract="distributed.route_uses_live_links",
                expected_behavior="the selected communication route contains no failed link",
                observed_behavior=f"static route selects failed links {sorted(failed & selected)!r}",
                learned_precondition="selected_links is disjoint from topology.failed_links",
            )
        return None

    def evaluate_resource(self, case: ResourceAdversarialCase) -> ViolationObservation | None:
        resource = case.resource
        if resource.queue_depth >= self.descriptor.queue_capacity:
            return ViolationObservation(
                surface=RedTeamSurface.RESOURCE,
                violated_contract="resource.queue_bound",
                expected_behavior="admission stops before reserving beyond the bounded queue",
                observed_behavior=(
                    "candidate tests queue_depth <= capacity and reserves one additional queue slot"
                ),
                learned_precondition="admit only when queue_depth < queue_capacity",
            )
        if resource.device_bytes > self.descriptor.device_capacity_bytes:
            return ViolationObservation(
                surface=RedTeamSurface.RESOURCE,
                violated_contract="resource.device_memory_bound",
                expected_behavior="candidate allocation remains within declared device capacity",
                observed_behavior="requested device allocation exceeds the declared capacity",
                learned_precondition="device_bytes <= device_capacity_bytes",
            )
        if resource.host_bytes > self.descriptor.host_capacity_bytes:
            return ViolationObservation(
                surface=RedTeamSurface.RESOURCE,
                violated_contract="resource.host_memory_bound",
                expected_behavior="candidate allocation remains within declared host capacity",
                observed_behavior="requested host allocation exceeds the declared capacity",
                learned_precondition="host_bytes <= host_capacity_bytes",
            )
        if resource.process_count > self.descriptor.process_limit:
            return ViolationObservation(
                surface=RedTeamSurface.RESOURCE,
                violated_contract="resource.process_bound",
                expected_behavior="generated runtime remains within its process limit",
                observed_behavior="candidate process count exceeds the sandbox limit",
                learned_precondition="process_count <= process_limit",
            )
        return None


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    reversed_strides: list[int] = []
    for dimension in reversed(shape):
        reversed_strides.append(stride)
        stride *= dimension
    return tuple(reversed(reversed_strides))


def unsafe_benchmark_comparison() -> BenchmarkComparison:
    """Return evidence that exercises every benchmark-integrity defense."""

    baseline = BenchmarkRunManifest(
        run_id="baseline-run",
        candidate_id="reference-runtime",
        benchmark_definition_hash=_digest("a"),
        input_fingerprints=(_digest("b"), _digest("c")),
        synchronized=True,
        timer_kind=TimerKind.MONOTONIC,
        warmup_iterations=10,
        cache_regime=CacheRegime.COLD,
        cache_reset_between_trials=True,
        fallback_invocations=0,
        precision=Precision.FLOAT32,
        quality_contract_hash=_digest("d"),
        quality_score=1.0,
        failures_included=True,
        hardware_clock_mhz=1500,
        cpu_affinity=(0, 1),
        background_processes=("gateway",),
        raw_samples=(10.0, 10.2, 9.9, 10.1),
    )
    candidate = BenchmarkRunManifest(
        run_id="unsafe-candidate-run",
        candidate_id="unsafe-fastpath-v1",
        benchmark_definition_hash=_digest("e"),
        input_fingerprints=(_digest("f"),),
        synchronized=False,
        timer_kind=TimerKind.WALL_CLOCK,
        warmup_iterations=1,
        cache_regime=CacheRegime.WARM,
        cache_reset_between_trials=False,
        fallback_invocations=2,
        precision=Precision.FLOAT16,
        quality_contract_hash=_digest("e"),
        quality_score=0.8,
        failures_included=False,
        hardware_clock_mhz=1800,
        cpu_affinity=(2, 3),
        background_processes=("gateway", "load-interference"),
        raw_samples=(5.0, 5.1, 20.0, 4.9),
        discarded_sample_indices=(2,),
    )
    return BenchmarkComparison(
        baseline=baseline,
        candidate=candidate,
        required_precision=Precision.FLOAT32,
        quality_contract_hash=_digest("d"),
        minimum_quality_score=0.99,
        maximum_clock_delta_mhz=15,
        allowed_background_processes=("gateway",),
    )


def renumber_events(events: tuple[RequestEventCase, ...]) -> tuple[RequestEventCase, ...]:
    return tuple(
        RequestEventCase(
            at_step=index,
            request_id=event.request_id,
            action=event.action,
            worker_id=event.worker_id,
        )
        for index, event in enumerate(events)
    )


__all__ = ["UnsafeStreamingCandidate", "renumber_events", "unsafe_benchmark_comparison"]
