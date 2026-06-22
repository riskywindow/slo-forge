"""Bounded runtime request, lifecycle, streaming, and metrics types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from threading import Event, Lock
from time import monotonic


class RequestLifecycle(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class EventKind(StrEnum):
    TOKEN = "token"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    ERROR = "error"


@dataclass(frozen=True)
class RuntimeRequest:
    request_id: str
    prompt_tokens: tuple[int, ...]
    maximum_new_tokens: int
    seed: int
    timeout_seconds: float
    batching_eligible: bool = True

    def validate(self, *, maximum_prompt_tokens: int, maximum_generated_tokens: int) -> None:
        if not self.request_id or any(ord(character) < 32 for character in self.request_id):
            raise ValueError("request_id must be a non-empty printable string")
        if not self.prompt_tokens:
            raise ValueError("prompt_tokens cannot be empty")
        if len(self.prompt_tokens) > maximum_prompt_tokens:
            raise ValueError("prompt exceeds the declared maximum_prompt_tokens")
        if not 1 <= self.maximum_new_tokens <= maximum_generated_tokens:
            raise ValueError("maximum_new_tokens is outside the declared bounded domain")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if any(token < 0 for token in self.prompt_tokens):
            raise ValueError("prompt token identifiers must be non-negative")


@dataclass(frozen=True)
class StreamEvent:
    request_id: str
    kind: EventKind
    sequence: int
    token_id: int | None = None
    text: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class RuntimeLimits:
    maximum_queue_depth: int = 32
    maximum_batch_size: int = 4
    maximum_prompt_tokens: int = 2048
    maximum_generated_tokens: int = 256
    maximum_output_events_per_request: int = 258
    batch_wait_seconds: float = 0.002
    queue_operation_timeout_seconds: float = 0.5
    worker_poll_seconds: float = 0.05
    shutdown_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        integer_bounds = (
            self.maximum_queue_depth,
            self.maximum_batch_size,
            self.maximum_prompt_tokens,
            self.maximum_generated_tokens,
            self.maximum_output_events_per_request,
        )
        if any(value <= 0 for value in integer_bounds):
            raise ValueError("runtime integer limits must be positive")
        if self.maximum_output_events_per_request < self.maximum_generated_tokens + 1:
            raise ValueError("output event bound must fit generated tokens and a terminal event")
        if self.maximum_batch_size > self.maximum_queue_depth:
            raise ValueError("batch size cannot exceed queue depth")
        timeouts = (
            self.batch_wait_seconds,
            self.queue_operation_timeout_seconds,
            self.worker_poll_seconds,
            self.shutdown_timeout_seconds,
        )
        if any(value <= 0 for value in timeouts):
            raise ValueError("runtime timeouts must be positive")


@dataclass
class RequestControl:
    request: RuntimeRequest
    lifecycle: RequestLifecycle = RequestLifecycle.CREATED
    created_at: float = field(default_factory=monotonic)
    cancellation: Event = field(default_factory=Event)
    finished: Event = field(default_factory=Event)
    lifecycle_lock: Lock = field(default_factory=Lock)

    def transition(self, target: RequestLifecycle) -> None:
        allowed = {
            RequestLifecycle.CREATED: {RequestLifecycle.QUEUED},
            RequestLifecycle.QUEUED: {
                RequestLifecycle.PREFILLING,
                RequestLifecycle.CANCELLED,
                RequestLifecycle.TIMED_OUT,
            },
            RequestLifecycle.PREFILLING: {
                RequestLifecycle.DECODING,
                RequestLifecycle.CANCELLED,
                RequestLifecycle.TIMED_OUT,
                RequestLifecycle.FAILED,
            },
            RequestLifecycle.DECODING: {
                RequestLifecycle.COMPLETED,
                RequestLifecycle.CANCELLED,
                RequestLifecycle.TIMED_OUT,
                RequestLifecycle.FAILED,
            },
            RequestLifecycle.COMPLETED: set(),
            RequestLifecycle.CANCELLED: set(),
            RequestLifecycle.TIMED_OUT: set(),
            RequestLifecycle.FAILED: set(),
        }
        with self.lifecycle_lock:
            if target not in allowed[self.lifecycle]:
                raise RuntimeError(f"illegal request transition {self.lifecycle} -> {target}")
            self.lifecycle = target
            if target in {
                RequestLifecycle.COMPLETED,
                RequestLifecycle.CANCELLED,
                RequestLifecycle.TIMED_OUT,
                RequestLifecycle.FAILED,
            }:
                self.finished.set()

    def expired(self) -> bool:
        return monotonic() - self.created_at >= self.request.timeout_seconds


@dataclass
class RuntimeMetrics:
    submitted: int = 0
    rejected_queue_full: int = 0
    completed: int = 0
    cancelled: int = 0
    timed_out: int = 0
    failed: int = 0
    emitted_tokens: int = 0
    batches: int = 0
    maximum_observed_batch: int = 0
    state_allocations: int = 0
    state_releases: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "submitted": self.submitted,
            "rejected_queue_full": self.rejected_queue_full,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "timed_out": self.timed_out,
            "failed": self.failed,
            "emitted_tokens": self.emitted_tokens,
            "batches": self.batches,
            "maximum_observed_batch": self.maximum_observed_batch,
            "state_allocations": self.state_allocations,
            "state_releases": self.state_releases,
        }
