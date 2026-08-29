"""Production engine-step synchronization for the vLLM 0.23 v11 adapter.

vLLM's in-process synchronous engine does not expose a public transaction lock
for out-of-band KV import/export.  This binding installs one bounded re-entrant
critical section *on the production mutation entry points themselves*.  It is
therefore not a side mutex: every ``LLMEngine.step`` and every scheduler/KV
mutation used by vLLM 0.23 or the Continuum restore stager enters the same
critical section.

The binding is deliberately version- and topology-scoped.  It accepts only the
validated V1 ``InprocClient`` hierarchy, rejects asynchronous scheduling, and
fails closed if any installed wrapper is displaced.  Export capture and import
admission witnesses can only be issued by the thread that currently owns the
installed critical section.
"""

from __future__ import annotations

import functools
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

VLLM_V11_SYNCHRONIZATION_SCHEMA_VERSION: Literal[
    "sloforge.continuum.vllm-v11-engine-step-binding/v1"
] = "sloforge.continuum.vllm-v11-engine-step-binding/v1"
VLLM_V11_SYNCHRONIZATION_RUNTIME_VERSION: Literal["0.23.0"] = "0.23.0"

V11CriticalOperation = Literal["EXPORT_CAPTURE", "IMPORT_ADMISSION"]

_V11_PRODUCTION_GATE_SEAL = object()
_WRAPPER_BINDING_ATTRIBUTE = "__sloforge_v11_engine_step_binding_id__"
_WRAPPER_SURFACE_ATTRIBUTE = "__sloforge_v11_engine_step_surface__"


class V11EngineStepBindingError(RuntimeError):
    """The production vLLM synchronization binding is missing or invalid."""


class V11EngineStepTimeout(TimeoutError):
    """A production engine-step critical-section acquisition timed out."""


@dataclass(frozen=True, slots=True)
class V11AdmissionGateWitness:
    """Unforgeable, single-critical-section capability for v11 state access."""

    gate_id: str
    binding_id: str
    operation: V11CriticalOperation
    scheduler_object_id: int
    manager_object_id: int
    owner_thread_id: int
    acquisition_generation: int
    issued_at_monotonic_ns: int
    _binding: Vllm0230EngineStepBinding = field(repr=False, compare=False)
    _proof_seal: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class V11EngineStepBindingEvidence:
    """Offline-serializable description of the exact installed binding."""

    schema_version: str
    binding_id: str
    runtime_version: str
    engine_core_client_type: str
    asynchronous_scheduling: bool
    scheduler_object_id: int
    manager_object_id: int
    wrapped_surfaces: tuple[str, ...]
    acquire_timeout_seconds: float
    installed_at_monotonic_ns: int
    active: bool
    wrappers_intact: bool

    @property
    def passed(self) -> bool:
        return (
            self.runtime_version == VLLM_V11_SYNCHRONIZATION_RUNTIME_VERSION
            and self.engine_core_client_type.startswith("vllm.v1.")
            and self.engine_core_client_type.endswith(".InprocClient")
            and not self.asynchronous_scheduling
            and bool(self.wrapped_surfaces)
            and self.active
            and self.wrappers_intact
        )


@dataclass(slots=True)
class _InstalledSurface:
    label: str
    target: Any
    name: str
    original: Callable[..., Any]
    wrapper: Callable[..., Any]


class Vllm0230EngineStepBinding:
    """Bind v11 state transactions to the real synchronous vLLM step path.

    Construct this only from ``VllmLiveInternalView`` for the live engine.  The
    constructor patches the current instance methods (and therefore composes
    with already-installed measurement wrappers) rather than a test-only lock.
    """

    _REQUIRED_SURFACES = (
        ("llm_engine", "step"),
        ("llm_engine", "add_request"),
        ("scheduler", "add_request"),
        ("scheduler", "_enqueue_waiting_request"),
        ("scheduler", "schedule"),
        ("scheduler", "update_from_output"),
        ("manager", "allocate_slots"),
        ("manager", "free"),
        ("manager", "cache_blocks"),
        ("manager", "reset_prefix_cache"),
        ("manager", "take_new_block_ids"),
        ("waiting", "add_request"),
        ("waiting", "remove_requests"),
        ("skipped_waiting", "add_request"),
        ("skipped_waiting", "remove_requests"),
    )
    _OPTIONAL_MUTATION_SURFACES = (
        ("llm_engine", "abort_request"),
        ("engine_core_client", "add_request"),
        ("engine_core_client", "abort_requests"),
        ("engine_core", "add_request"),
        ("engine_core", "step"),
        ("scheduler", "finish_requests"),
    )

    def __init__(
        self,
        view: Any,
        *,
        acquire_timeout_seconds: float = 30.0,
    ) -> None:
        if not 0 < acquire_timeout_seconds <= 300:
            raise ValueError("v11 engine-step acquire timeout must be in (0, 300] seconds")
        self._validate_production_view(view)
        self.view = view
        self.llm_engine = view.llm_engine
        self.engine_core_client = view.engine_core_client
        self.engine_core = view.engine_core
        self.scheduler = view.scheduler
        self.manager = view.manager
        self.waiting = self.scheduler.waiting
        self.skipped_waiting = self.scheduler.skipped_waiting
        self.binding_id = f"vllm-0.23-engine-step-{uuid.uuid4()}"
        self.acquire_timeout_seconds = float(acquire_timeout_seconds)
        self.installed_at_monotonic_ns = time.monotonic_ns()
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._owner_thread_id: int | None = None
        self._owner_depth = 0
        self._acquisition_generation = 0
        self._closed = False
        self._surfaces: list[_InstalledSurface] = []
        self._install()

    @staticmethod
    def _validate_production_view(view: Any) -> None:
        if getattr(view, "runtime_version", None) != VLLM_V11_SYNCHRONIZATION_RUNTIME_VERSION:
            raise V11EngineStepBindingError("v11 synchronization requires exactly vLLM 0.23.0")
        required = (
            "llm_engine",
            "engine_core_client",
            "engine_core",
            "scheduler",
            "manager",
            "vllm_config",
        )
        if any(getattr(view, name, None) is None for name in required):
            raise V11EngineStepBindingError("v11 synchronization requires the validated live view")
        client = view.engine_core_client
        client_type = type(client)
        if client_type.__name__ != "InprocClient" or not client_type.__module__.startswith(
            "vllm.v1."
        ):
            raise V11EngineStepBindingError(
                "v11 synchronization requires the production V1 InprocClient"
            )
        if getattr(view.llm_engine, "engine_core", None) is not client:
            raise V11EngineStepBindingError("LLMEngine is not bound to the validated InprocClient")
        if getattr(client, "engine_core", None) is not view.engine_core:
            raise V11EngineStepBindingError("InprocClient is not bound to the validated EngineCore")
        if getattr(view.engine_core, "scheduler", None) is not view.scheduler:
            raise V11EngineStepBindingError("EngineCore is not bound to the validated Scheduler")
        if getattr(view.scheduler, "kv_cache_manager", None) is not view.manager:
            raise V11EngineStepBindingError("Scheduler is not bound to the validated KV manager")
        scheduler_config = getattr(view.vllm_config, "scheduler_config", None)
        if scheduler_config is None or bool(getattr(scheduler_config, "async_scheduling", False)):
            raise V11EngineStepBindingError("v11 synchronization rejects async scheduling")
        internal_config = getattr(view.scheduler, "scheduler_config", scheduler_config)
        if bool(getattr(internal_config, "async_scheduling", False)):
            raise V11EngineStepBindingError("v11 scheduler unexpectedly enables async scheduling")
        if (
            getattr(view.scheduler, "waiting", None) is None
            or getattr(view.scheduler, "skipped_waiting", None) is None
        ):
            raise V11EngineStepBindingError("v11 scheduler queues are unavailable")

    def _surface_target(self, owner_name: str) -> Any:
        return getattr(self, owner_name)

    def _make_wrapper(
        self,
        label: str,
        original: Callable[..., Any],
    ) -> Callable[..., Any]:
        @functools.wraps(original)
        def synchronized(*args: Any, **kwargs: Any) -> Any:
            with self._hold_lock(label):
                return original(*args, **kwargs)

        setattr(synchronized, _WRAPPER_BINDING_ATTRIBUTE, self.binding_id)
        setattr(synchronized, _WRAPPER_SURFACE_ATTRIBUTE, label)
        return synchronized

    def _patch(self, owner_name: str, method_name: str, *, required: bool) -> None:
        target = self._surface_target(owner_name)
        original = getattr(target, method_name, None)
        if not callable(original):
            if required:
                raise V11EngineStepBindingError(
                    f"required vLLM 0.23 mutation surface is absent: {owner_name}.{method_name}"
                )
            return
        label = f"{owner_name}.{method_name}"
        wrapper = self._make_wrapper(label, original)
        try:
            setattr(target, method_name, wrapper)
        except (AttributeError, TypeError) as error:
            raise V11EngineStepBindingError(
                f"cannot bind production mutation surface: {label}"
            ) from error
        if getattr(target, method_name, None) is not wrapper:
            raise V11EngineStepBindingError(f"production mutation wrapper did not install: {label}")
        self._surfaces.append(
            _InstalledSurface(
                label=label,
                target=target,
                name=method_name,
                original=original,
                wrapper=wrapper,
            )
        )

    def _install(self) -> None:
        try:
            for owner_name, method_name in self._REQUIRED_SURFACES:
                self._patch(owner_name, method_name, required=True)
            for owner_name, method_name in self._OPTIONAL_MUTATION_SURFACES:
                self._patch(owner_name, method_name, required=False)
        except BaseException:
            self._restore_installed_surfaces()
            self._closed = True
            raise
        self.assert_bound()

    def _restore_installed_surfaces(self) -> None:
        for surface in reversed(self._surfaces):
            if getattr(surface.target, surface.name, None) is surface.wrapper:
                setattr(surface.target, surface.name, surface.original)
        self._surfaces.clear()

    def assert_bound(self) -> None:
        if self._closed:
            raise V11EngineStepBindingError("v11 engine-step binding is closed")
        if not self._surfaces:
            raise V11EngineStepBindingError("v11 engine-step binding installed no surfaces")
        displaced = [
            surface.label
            for surface in self._surfaces
            if getattr(surface.target, surface.name, None) is not surface.wrapper
            or getattr(surface.wrapper, _WRAPPER_BINDING_ATTRIBUTE, None) != self.binding_id
        ]
        if displaced:
            raise V11EngineStepBindingError(
                "v11 engine-step binding was displaced: " + ", ".join(displaced)
            )

    @contextmanager
    def _hold_lock(self, operation: str) -> Iterator[None]:
        del operation
        if self._closed:
            raise V11EngineStepBindingError("v11 engine-step binding is closed")
        acquired = self._lock.acquire(timeout=self.acquire_timeout_seconds)
        if not acquired:
            raise V11EngineStepTimeout("timed out acquiring the production engine-step gate")
        thread_id = threading.get_ident()
        with self._state_lock:
            if self._owner_thread_id is None:
                self._owner_thread_id = thread_id
                self._owner_depth = 1
                self._acquisition_generation += 1
            elif self._owner_thread_id == thread_id:
                self._owner_depth += 1
            else:  # pragma: no cover - RLock excludes this state.
                self._lock.release()
                raise V11EngineStepBindingError("engine-step lock ownership became inconsistent")
        try:
            yield
        finally:
            with self._state_lock:
                if self._owner_thread_id != thread_id or self._owner_depth <= 0:
                    raise V11EngineStepBindingError("engine-step lock release ownership is invalid")
                self._owner_depth -= 1
                if self._owner_depth == 0:
                    self._owner_thread_id = None
            self._lock.release()

    @contextmanager
    def critical_section(
        self,
        operation: V11CriticalOperation,
        *,
        gate_id: str,
    ) -> Iterator[V11AdmissionGateWitness]:
        """Exclude production engine steps while one v11 transaction is active."""

        if operation not in ("EXPORT_CAPTURE", "IMPORT_ADMISSION"):
            raise ValueError("unsupported v11 engine-step critical operation")
        if not gate_id or len(gate_id) > 512:
            raise ValueError("v11 engine-step gate ID must contain 1..512 characters")
        self.assert_bound()
        with self._hold_lock(operation):
            self.assert_bound()
            witness = self._issue_witness(operation=operation, gate_id=gate_id)
            try:
                yield witness
            finally:
                self.require_witness(
                    witness,
                    scheduler=self.scheduler,
                    manager=self.manager,
                    operation=operation,
                )

    def _issue_witness(
        self,
        *,
        operation: V11CriticalOperation,
        gate_id: str,
    ) -> V11AdmissionGateWitness:
        thread_id = threading.get_ident()
        with self._state_lock:
            if self._owner_thread_id != thread_id or self._owner_depth <= 0:
                raise V11EngineStepBindingError(
                    "v11 witness requested without owning the production engine-step gate"
                )
            generation = self._acquisition_generation
        return V11AdmissionGateWitness(
            gate_id=gate_id,
            binding_id=self.binding_id,
            operation=operation,
            scheduler_object_id=id(self.scheduler),
            manager_object_id=id(self.manager),
            owner_thread_id=thread_id,
            acquisition_generation=generation,
            issued_at_monotonic_ns=time.monotonic_ns(),
            _binding=self,
            _proof_seal=_V11_PRODUCTION_GATE_SEAL,
        )

    def require_witness(
        self,
        witness: V11AdmissionGateWitness,
        *,
        scheduler: Any,
        manager: Any,
        operation: V11CriticalOperation,
    ) -> None:
        self.assert_bound()
        thread_id = threading.get_ident()
        with self._state_lock:
            owned = self._owner_thread_id == thread_id and self._owner_depth > 0
            generation = self._acquisition_generation
        if (
            not isinstance(witness, V11AdmissionGateWitness)
            or witness._proof_seal is not _V11_PRODUCTION_GATE_SEAL
            or witness._binding is not self
            or witness.binding_id != self.binding_id
            or witness.operation != operation
            or witness.scheduler_object_id != id(scheduler)
            or witness.manager_object_id != id(manager)
            or witness.owner_thread_id != thread_id
            or witness.acquisition_generation != generation
            or scheduler is not self.scheduler
            or manager is not self.manager
            or not owned
        ):
            raise V11EngineStepBindingError(
                "v11 production engine-step witness is stale, foreign, or not held"
            )

    def evidence(self) -> V11EngineStepBindingEvidence:
        client_type = type(self.engine_core_client)
        try:
            self.assert_bound()
        except V11EngineStepBindingError:
            intact = False
        else:
            intact = True
        scheduler_config = getattr(self.view.vllm_config, "scheduler_config", None)
        return V11EngineStepBindingEvidence(
            schema_version=VLLM_V11_SYNCHRONIZATION_SCHEMA_VERSION,
            binding_id=self.binding_id,
            runtime_version=str(self.view.runtime_version),
            engine_core_client_type=f"{client_type.__module__}.{client_type.__name__}",
            asynchronous_scheduling=bool(getattr(scheduler_config, "async_scheduling", False)),
            scheduler_object_id=id(self.scheduler),
            manager_object_id=id(self.manager),
            wrapped_surfaces=tuple(surface.label for surface in self._surfaces),
            acquire_timeout_seconds=self.acquire_timeout_seconds,
            installed_at_monotonic_ns=self.installed_at_monotonic_ns,
            active=not self._closed,
            wrappers_intact=intact,
        )

    def close(self) -> None:
        if self._closed:
            return
        with self._state_lock:
            if self._owner_thread_id is not None or self._owner_depth:
                raise V11EngineStepBindingError("cannot close a held engine-step binding")
        self.assert_bound()
        self._restore_installed_surfaces()
        self._closed = True

    def __enter__(self) -> Vllm0230EngineStepBinding:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


def require_production_gate_v11(
    witness: V11AdmissionGateWitness,
    scheduler: Any,
    manager: Any,
    *,
    operation: V11CriticalOperation,
) -> None:
    """Validate a witness without accepting a caller-provided boolean probe."""

    if not isinstance(witness, V11AdmissionGateWitness):
        raise V11EngineStepBindingError("v11 production engine-step witness is required")
    witness._binding.require_witness(
        witness,
        scheduler=scheduler,
        manager=manager,
        operation=operation,
    )


def require_production_gate_operation_v11(
    witness: V11AdmissionGateWitness,
    *,
    operation: V11CriticalOperation,
) -> None:
    """Validate a gate for CUDA helpers that do not otherwise own runtime objects."""

    if not isinstance(witness, V11AdmissionGateWitness):
        raise V11EngineStepBindingError("v11 production engine-step witness is required")
    binding = witness._binding
    binding.require_witness(
        witness,
        scheduler=binding.scheduler,
        manager=binding.manager,
        operation=operation,
    )


__all__ = [
    "VLLM_V11_SYNCHRONIZATION_RUNTIME_VERSION",
    "VLLM_V11_SYNCHRONIZATION_SCHEMA_VERSION",
    "V11AdmissionGateWitness",
    "V11EngineStepBindingError",
    "V11EngineStepBindingEvidence",
    "V11EngineStepTimeout",
    "Vllm0230EngineStepBinding",
    "require_production_gate_operation_v11",
    "require_production_gate_v11",
]
