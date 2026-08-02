"""Conservative bounded streaming runtime used by generated model adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import monotonic

from sloforge.genesis.policy_dsl import BytecodeProgram, ScalarType, execute_bytecode

from .adapter import ReferenceRuntimeAdapter
from .models import (
    EventKind,
    RequestControl,
    RequestLifecycle,
    RuntimeLimits,
    RuntimeMetrics,
    RuntimeRequest,
    StreamEvent,
)


class QueueSaturatedError(RuntimeError):
    """The bounded admission queue could not accept a request in time."""


class StreamHandle:
    def __init__(self, control: RequestControl, events: Queue[StreamEvent]) -> None:
        self._control = control
        self._events = events

    @property
    def request_id(self) -> str:
        return self._control.request.request_id

    @property
    def lifecycle(self) -> RequestLifecycle:
        with self._control.lifecycle_lock:
            return self._control.lifecycle

    def cancel(self) -> bool:
        return self._control.cancel()

    def next_event(self, timeout_seconds: float) -> StreamEvent:
        if timeout_seconds <= 0:
            raise ValueError("stream timeout must be positive")
        try:
            event = self._events.get(timeout=timeout_seconds)
        except Empty as error:
            raise TimeoutError(f"no event received for request {self.request_id}") from error
        if event.kind is not EventKind.TOKEN:
            # The producer publishes the terminal event and lifecycle while
            # holding this lock. This barrier prevents a consumer from seeing
            # the event before the matching lifecycle/finished state.
            with self._control.commit_lock:
                if not self._control.finished.is_set():
                    raise RuntimeError("terminal event was published without terminal lifecycle")
        return event

    def events(self, timeout_seconds: float) -> Iterator[StreamEvent]:
        while True:
            event = self.next_event(timeout_seconds)
            yield event
            if event.kind != EventKind.TOKEN:
                return

    def wait(self, timeout_seconds: float) -> bool:
        if timeout_seconds <= 0:
            raise ValueError("wait timeout must be positive")
        return self._control.finished.wait(timeout_seconds)


class _StateOwners:
    def __init__(self) -> None:
        self._owners: dict[str, str] = {}
        self._lock = Lock()

    def allocate(self, request_id: str, owner: str) -> None:
        with self._lock:
            if request_id in self._owners:
                raise RuntimeError(f"state for request {request_id!r} already has an owner")
            self._owners[request_id] = owner

    def release(self, request_id: str, owner: str) -> None:
        with self._lock:
            actual = self._owners.get(request_id)
            if actual != owner:
                raise RuntimeError(
                    f"state owner mismatch for {request_id!r}: expected {owner!r}, got {actual!r}"
                )
            del self._owners[request_id]

    def count(self) -> int:
        with self._lock:
            return len(self._owners)


class BaselineStreamingRuntime:
    """Single-worker deterministic runtime with bounded batching and state ownership."""

    def __init__(
        self,
        adapter: ReferenceRuntimeAdapter,
        *,
        limits: RuntimeLimits,
        runtime_seed: int,
        policy: BytecodeProgram | None = None,
    ) -> None:
        self._adapter = adapter
        self._limits = limits
        self._runtime_seed = runtime_seed
        self._policy = policy
        self._admission: Queue[tuple[RequestControl, Queue[StreamEvent]]] = Queue(
            maxsize=limits.maximum_queue_depth
        )
        self._state_owners = _StateOwners()
        self._metrics = RuntimeMetrics()
        self._metrics_lock = Lock()
        self._active: dict[str, RequestControl] = {}
        self._active_lock = Lock()
        self._stopping = False
        self._started = False
        self._lifecycle_lock = Lock()
        self._worker: Thread | None = None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("runtime has already been started")
            if self._stopping:
                raise RuntimeError("a stopped runtime cannot be restarted")
            self._validate_policy_contract()
            self._started = True
            self._worker = Thread(target=self._worker_loop, name="genesis-baseline", daemon=False)
            self._worker.start()

    def __enter__(self) -> BaselineStreamingRuntime:
        self.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.shutdown()

    def submit(self, request: RuntimeRequest) -> StreamHandle:
        request.validate(
            maximum_prompt_tokens=self._limits.maximum_prompt_tokens,
            maximum_generated_tokens=self._limits.maximum_generated_tokens,
        )
        control = RequestControl(request=request)
        events: Queue[StreamEvent] = Queue(maxsize=self._limits.maximum_output_events_per_request)
        with self._lifecycle_lock:
            if not self._started or self._stopping:
                raise RuntimeError("runtime is not accepting requests")
            if self._worker is None or not self._worker.is_alive():
                raise RuntimeError("runtime worker is unavailable")
            with self._active_lock:
                if request.request_id in self._active:
                    raise ValueError(f"request identifier is already active: {request.request_id}")
                self._active[request.request_id] = control
            control.transition(RequestLifecycle.QUEUED)
            try:
                self._admission.put(
                    (control, events), timeout=self._limits.queue_operation_timeout_seconds
                )
            except Full as error:
                with self._active_lock:
                    self._active.pop(request.request_id, None)
                with self._metrics_lock:
                    self._metrics.rejected_queue_full += 1
                raise QueueSaturatedError("bounded admission queue is saturated") from error
        with self._metrics_lock:
            self._metrics.submitted += 1
        return StreamHandle(control, events)

    def submit_text(
        self,
        *,
        request_id: str,
        text: str,
        maximum_new_tokens: int,
        seed: int,
        timeout_seconds: float,
        batching_eligible: bool = True,
    ) -> StreamHandle:
        return self.submit(
            RuntimeRequest(
                request_id=request_id,
                prompt_tokens=self._adapter.tokenize(text),
                maximum_new_tokens=maximum_new_tokens,
                seed=seed,
                timeout_seconds=timeout_seconds,
                batching_eligible=batching_eligible,
            )
        )

    def health(self) -> dict[str, object]:
        with self._lifecycle_lock:
            accepting = (
                self._started
                and not self._stopping
                and self._worker is not None
                and self._worker.is_alive()
            )
        return {
            "status": "ready" if accepting else "stopped",
            "accepting_requests": accepting,
            "queue_depth": self._admission.qsize(),
            "queue_capacity": self._limits.maximum_queue_depth,
            "owned_state_count": self._state_owners.count(),
            "policy": self._policy.name if self._policy is not None else None,
        }

    def metrics(self) -> dict[str, int]:
        with self._metrics_lock:
            return self._metrics.snapshot()

    def shutdown(self, timeout_seconds: float | None = None) -> None:
        timeout = timeout_seconds or self._limits.shutdown_timeout_seconds
        if timeout <= 0:
            raise ValueError("shutdown timeout must be positive")
        with self._lifecycle_lock:
            if not self._started:
                return
            self._stopping = True
        with self._active_lock:
            controls = tuple(self._active.values())
        for control in controls:
            control.cancel()
        worker = self._worker
        if worker is not None:
            worker.join(timeout)
            if worker.is_alive():
                raise TimeoutError("baseline runtime worker did not stop before its deadline")
        self._terminalize_abandoned_requests()
        with self._lifecycle_lock:
            self._started = False
        if self._state_owners.count() != 0:
            raise RuntimeError("runtime shutdown left owned request state")
        with self._active_lock:
            if self._active:
                raise RuntimeError("runtime shutdown left active requests")
        if not self._admission.empty():
            raise RuntimeError("runtime shutdown left queued requests")

    def _terminalize_abandoned_requests(self) -> None:
        """Cancel queued work if a worker exited before consuming it."""

        while True:
            try:
                control, events = self._admission.get_nowait()
            except Empty:
                return
            try:
                if control.lifecycle not in {
                    RequestLifecycle.COMPLETED,
                    RequestLifecycle.CANCELLED,
                    RequestLifecycle.TIMED_OUT,
                    RequestLifecycle.FAILED,
                }:
                    self._terminal(
                        control,
                        events,
                        lifecycle=RequestLifecycle.CANCELLED,
                        kind=EventKind.CANCELLED,
                        sequence=0,
                        message="runtime shutdown cancelled queued request",
                    )
                with self._active_lock:
                    self._active.pop(control.request.request_id, None)
            finally:
                self._admission.task_done()

    def _worker_loop(self) -> None:
        pending: tuple[RequestControl, Queue[StreamEvent]] | None = None
        while True:
            if pending is not None:
                first = pending
                pending = None
            else:
                try:
                    first = self._admission.get(timeout=self._limits.worker_poll_seconds)
                except Empty:
                    with self._lifecycle_lock:
                        if self._stopping and self._admission.empty():
                            return
                    continue
            batch = [first]
            try:
                batch_limit = self._batch_limit(first[0])
            except BaseException as error:
                self._terminal(
                    first[0],
                    first[1],
                    lifecycle=RequestLifecycle.FAILED,
                    kind=EventKind.ERROR,
                    sequence=0,
                    message=f"{type(error).__name__}: {error}",
                )
                with self._active_lock:
                    self._active.pop(first[0].request.request_id, None)
                self._admission.task_done()
                continue
            if batch_limit == 0:
                self._terminal(
                    first[0],
                    first[1],
                    lifecycle=RequestLifecycle.CANCELLED,
                    kind=EventKind.CANCELLED,
                    sequence=0,
                    message="synthesized policy suppressed a cancelled request before scheduling",
                )
                with self._active_lock:
                    self._active.pop(first[0].request.request_id, None)
                self._admission.task_done()
                continue
            if first[0].request.batching_eligible and batch_limit > 1:
                deadline = monotonic() + self._limits.batch_wait_seconds
                while len(batch) < batch_limit:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    try:
                        candidate = self._admission.get(timeout=remaining)
                    except Empty:
                        break
                    if not candidate[0].request.batching_eligible:
                        pending = candidate
                        break
                    batch.append(candidate)
            if self._policy is not None and self._policy.name in {
                "deadline_batch",
                "deadline_cancel_batch",
            }:
                original_order = tuple(item[0].request.request_id for item in batch)
                batch.sort(
                    key=lambda item: (
                        item[0].created_at + item[0].request.timeout_seconds,
                        item[0].request.request_id,
                    )
                )
                if original_order != tuple(item[0].request.request_id for item in batch):
                    with self._metrics_lock:
                        self._metrics.deadline_reorders += 1
            with self._metrics_lock:
                self._metrics.batches += 1
                self._metrics.maximum_observed_batch = max(
                    self._metrics.maximum_observed_batch, len(batch)
                )
            for control, events in batch:
                self._process(control, events)
                self._admission.task_done()
            with self._lifecycle_lock:
                if self._stopping and self._admission.empty() and pending is None:
                    return

    def _batch_limit(self, control: RequestControl) -> int:
        if self._policy is None:
            return self._limits.maximum_batch_size
        elapsed = monotonic() - control.created_at
        remaining_ms = int(
            max(0.0, min(1000.0, (control.request.timeout_seconds - elapsed) * 1000.0))
        )
        available: dict[str, int | bool] = {
            "queue_length": min(32, self._admission.qsize() + 1),
            "slo_slack_ms": remaining_ms,
            "cancellation_pending": control.cancellation.is_set(),
        }
        names = {item.name for item in self._policy.inputs}
        unsupported = names - available.keys()
        if unsupported:
            raise RuntimeError(f"runtime policy requires unsupported inputs: {sorted(unsupported)}")
        result = execute_bytecode(self._policy, {name: available[name] for name in names})
        if type(result) is not int:
            raise RuntimeError("runtime batching policy must return an integer")
        if result == 0:
            if not control.cancellation.is_set():
                raise RuntimeError("runtime policy may return zero only for a cancelled request")
            return 0
        return min(self._limits.maximum_batch_size, result)

    def _validate_policy_contract(self) -> None:
        if self._policy is None:
            return
        supported = {
            "queue_length": ScalarType.INT,
            "slo_slack_ms": ScalarType.INT,
            "cancellation_pending": ScalarType.BOOL,
        }
        inputs = {item.name: item.scalar_type for item in self._policy.inputs}
        unsupported = inputs.keys() - supported.keys()
        if unsupported:
            raise RuntimeError(f"runtime policy requires unsupported inputs: {sorted(unsupported)}")
        mismatched = sorted(
            name for name, scalar_type in inputs.items() if supported[name] is not scalar_type
        )
        if mismatched:
            raise RuntimeError(f"runtime policy input types are incompatible: {mismatched}")
        if self._policy.output.scalar_type is not ScalarType.INT:
            raise RuntimeError("runtime batching policy must return an integer")

    def _seed(self, request: RuntimeRequest, phase: str, position: int) -> int:
        identity = (
            f"{self._runtime_seed}\0{request.seed}\0{request.request_id}\0{phase}\0{position}"
        ).encode()
        return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big", signed=False)

    def _emit(self, events: Queue[StreamEvent], event: StreamEvent) -> None:
        try:
            events.put(event, timeout=self._limits.queue_operation_timeout_seconds)
        except Full as error:
            raise RuntimeError(
                "bounded output queue exhausted despite declared request bound"
            ) from error

    def _terminal(
        self,
        control: RequestControl,
        events: Queue[StreamEvent],
        *,
        lifecycle: RequestLifecycle,
        kind: EventKind,
        sequence: int,
        message: str | None = None,
    ) -> None:
        with control.commit_lock:
            if lifecycle is RequestLifecycle.COMPLETED:
                if control.cancellation.is_set():
                    lifecycle = RequestLifecycle.CANCELLED
                    kind = EventKind.CANCELLED
                elif control.expired():
                    lifecycle = RequestLifecycle.TIMED_OUT
                    kind = EventKind.TIMED_OUT
            self._emit(
                events,
                StreamEvent(
                    request_id=control.request.request_id,
                    kind=kind,
                    sequence=sequence,
                    message=message,
                ),
            )
            control.transition(lifecycle)
        with self._metrics_lock:
            if lifecycle == RequestLifecycle.COMPLETED:
                self._metrics.completed += 1
            elif lifecycle == RequestLifecycle.CANCELLED:
                self._metrics.cancelled += 1
            elif lifecycle == RequestLifecycle.TIMED_OUT:
                self._metrics.timed_out += 1
            else:
                self._metrics.failed += 1

    def _abort_if_needed(
        self,
        control: RequestControl,
        events: Queue[StreamEvent],
        sequence: int,
    ) -> bool:
        if control.cancellation.is_set():
            self._terminal(
                control,
                events,
                lifecycle=RequestLifecycle.CANCELLED,
                kind=EventKind.CANCELLED,
                sequence=sequence,
            )
            return True
        if control.expired():
            self._terminal(
                control,
                events,
                lifecycle=RequestLifecycle.TIMED_OUT,
                kind=EventKind.TIMED_OUT,
                sequence=sequence,
            )
            return True
        return False

    def _commit_token(
        self,
        control: RequestControl,
        events: Queue[StreamEvent],
        event: StreamEvent,
    ) -> bool:
        """Atomically choose token emission or cancellation/deadline termination.

        A cancellation that returns before this method acquires the commit lock
        wins and produces a terminal event. If token enqueue wins, it completes
        before a concurrent cancellation call can return.
        """

        with control.commit_lock:
            if self._abort_if_needed(control, events, event.sequence):
                return False
            self._emit(events, event)
            return True

    def _process(self, control: RequestControl, events: Queue[StreamEvent]) -> None:
        request = control.request
        owner = f"worker:{request.request_id}"
        owns_state = False
        sequence = 0
        try:
            if self._abort_if_needed(control, events, sequence):
                return
            control.transition(RequestLifecycle.PREFILLING)
            self._state_owners.allocate(request.request_id, owner)
            owns_state = True
            with self._metrics_lock:
                self._metrics.state_allocations += 1
            state = self._adapter.allocate_state(
                request.request_id,
                request.prompt_tokens,
                self._seed(request, "allocate", 0),
            )
            state = self._adapter.prefill(
                request.prompt_tokens,
                state,
                self._seed(request, "prefill", 0),
            )
            if self._abort_if_needed(control, events, sequence):
                return
            control.transition(RequestLifecycle.DECODING)
            previous = request.prompt_tokens[-1]
            for position in range(request.maximum_new_tokens):
                if self._abort_if_needed(control, events, sequence):
                    return
                result = self._adapter.decode_step(
                    previous,
                    state,
                    position,
                    self._seed(request, "decode", position),
                )
                state = result.state
                if self._abort_if_needed(control, events, sequence):
                    return
                token = self._adapter.sample(
                    result.logits,
                    self._seed(request, "sample", position),
                )
                if self._abort_if_needed(control, events, sequence):
                    return
                token_event = StreamEvent(
                    request_id=request.request_id,
                    kind=EventKind.TOKEN,
                    sequence=sequence,
                    token_id=token,
                    text=self._adapter.detokenize(token),
                )
                if not self._commit_token(
                    control,
                    events,
                    token_event,
                ):
                    return
                sequence += 1
                previous = token
                with self._metrics_lock:
                    self._metrics.emitted_tokens += 1
            self._terminal(
                control,
                events,
                lifecycle=RequestLifecycle.COMPLETED,
                kind=EventKind.COMPLETED,
                sequence=sequence,
            )
        except BaseException as error:
            if control.lifecycle not in {
                RequestLifecycle.COMPLETED,
                RequestLifecycle.CANCELLED,
                RequestLifecycle.TIMED_OUT,
                RequestLifecycle.FAILED,
            }:
                self._terminal(
                    control,
                    events,
                    lifecycle=RequestLifecycle.FAILED,
                    kind=EventKind.ERROR,
                    sequence=sequence,
                    message=f"{type(error).__name__}: {error}",
                )
        finally:
            if owns_state:
                self._state_owners.release(request.request_id, owner)
                with self._metrics_lock:
                    self._metrics.state_releases += 1
            with self._active_lock:
                self._active.pop(request.request_id, None)
