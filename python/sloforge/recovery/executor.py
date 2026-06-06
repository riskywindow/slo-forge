"""Bounded deterministic recovery executor for local and simulated targets."""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Collection
from pathlib import Path
from typing import Protocol

from sloforge.fabric.ir import RecoveryAction, RecoveryPlan
from sloforge.util import canonical_json

from .models import (
    ActionAttempt,
    RecoveryMachineConfig,
    RecoveryObservation,
    RecoverySnapshot,
    RecoveryState,
)
from .state_machine import RecoveryStateMachine

MAX_RECOVERY_SNAPSHOT_BYTES = 8 * 1024 * 1024
MAX_ACTION_DETAIL_CHARS = 4_096


def _bounded_detail(value: str) -> str:
    if len(value) <= MAX_ACTION_DETAIL_CHARS:
        return value
    return value[: MAX_ACTION_DETAIL_CHARS - len("...[truncated]")] + "...[truncated]"


class ActionDriver(Protocol):
    """Synchronous, bounded driver contract.

    ``apply`` runs under the executor lock so a driver must enforce its own I/O
    timeout and return only after the action has reached an idempotent outcome.
    SLOForge ships only the non-mutating simulated implementation.
    """

    allow_external_mutation: bool

    def apply(self, action: RecoveryAction, *, now_ms: int) -> ActionAttempt: ...


class SimulatedActionDriver:
    """Deterministic driver that records intent and never mutates external state."""

    allow_external_mutation = False

    def __init__(self, *, fail_action_ids: Collection[str] = ()) -> None:
        self._failures = frozenset(fail_action_ids)
        self._attempted: set[str] = set()

    @property
    def attempted_action_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._attempted))

    def apply(self, action: RecoveryAction, *, now_ms: int) -> ActionAttempt:
        if action.action_id in self._attempted:
            raise RuntimeError(f"driver refused duplicate action {action.action_id}")
        if action.requires_external_mutation:
            raise PermissionError("simulated action driver cannot perform external mutations")
        self._attempted.add(action.action_id)
        succeeded = action.action_id not in self._failures
        return ActionAttempt(
            action_id=action.action_id,
            idempotency_key=action.idempotency_key,
            attempted_at_ms=now_ms,
            succeeded=succeeded,
            detail=(
                f"simulated action {action.kind.value} completed"
                if succeeded
                else f"deterministic injected failure for {action.kind.value}"
            ),
        )


class DeterministicRecoveryExecutor:
    """Thread-safe local executor with explicit snapshot persistence.

    One transition is evaluated per observation. Recovery actions are issued
    exactly once after simulation validation and are represented as auditable
    simulated effects unless a separately authorized driver is supplied.
    """

    def __init__(
        self,
        plan: RecoveryPlan,
        *,
        now_ms: int,
        config: RecoveryMachineConfig | None = None,
        driver: ActionDriver | None = None,
        snapshot: RecoverySnapshot | None = None,
    ) -> None:
        self.machine = RecoveryStateMachine(plan, config)
        self.driver = driver or SimulatedActionDriver()
        self._lock = threading.RLock()
        self.snapshot = snapshot or self.machine.start(now_ms=now_ms)
        if self.snapshot.recovery_plan_hash != self.machine.plan_hash:
            raise ValueError("executor snapshot does not match recovery plan")

    def _execute_pending_actions(self, *, now_ms: int) -> None:
        if self.snapshot.state is not RecoveryState.BUILDING_REPLACEMENT:
            return
        attempted = {item.action_id for item in self.snapshot.action_attempts}
        pending = tuple(
            action for action in self.machine.plan.actions if action.action_id not in attempted
        )
        if not pending:
            return
        attempts: list[ActionAttempt] = []
        for action in pending:
            if action.requires_external_mutation and not self.driver.allow_external_mutation:
                attempts.append(
                    ActionAttempt(
                        action_id=action.action_id,
                        idempotency_key=action.idempotency_key,
                        attempted_at_ms=now_ms,
                        succeeded=False,
                        detail="action driver lacks explicit external mutation authorization",
                    )
                )
                break
            try:
                attempt = self.driver.apply(action, now_ms=now_ms)
                if (
                    attempt.action_id != action.action_id
                    or attempt.idempotency_key != action.idempotency_key
                ):
                    attempts.append(
                        ActionAttempt(
                            action_id=action.action_id,
                            idempotency_key=action.idempotency_key,
                            attempted_at_ms=now_ms,
                            succeeded=False,
                            detail="action driver returned a mismatched action identity",
                        )
                    )
                    break
                attempts.append(
                    attempt.model_copy(update={"detail": _bounded_detail(attempt.detail)})
                )
            except Exception as error:
                attempts.append(
                    ActionAttempt(
                        action_id=action.action_id,
                        idempotency_key=action.idempotency_key,
                        attempted_at_ms=now_ms,
                        succeeded=False,
                        detail=_bounded_detail(str(error)),
                    )
                )
                break
        self.snapshot = self.machine.record_action_attempts(self.snapshot, attempts=tuple(attempts))

    def tick(self, observation: RecoveryObservation) -> RecoverySnapshot:
        with self._lock:
            self.snapshot = self.machine.advance(self.snapshot, observation)
            self._execute_pending_actions(now_ms=observation.observed_at_ms)
            return self.snapshot

    def dump_state(self) -> bytes:
        with self._lock:
            payload = canonical_json(self.snapshot.model_dump(mode="json"))
            encoded = (payload + "\n").encode()
            if len(encoded) > MAX_RECOVERY_SNAPSHOT_BYTES:
                raise ValueError("recovery snapshot exceeds bounded serialized size")
            return encoded

    def persist_state(self, path: Path) -> None:
        """Atomically persist a mode-0600 snapshot without following a final symlink."""

        with self._lock:
            payload = self.dump_state()
            path.parent.mkdir(parents=True, exist_ok=True)
            parent = path.parent.resolve()
            destination = parent / path.name
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, destination)
                temporary_name = ""
                if hasattr(os, "O_DIRECTORY"):
                    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            finally:
                if temporary_name:
                    Path(temporary_name).unlink(missing_ok=True)

    @classmethod
    def restore(
        cls,
        plan: RecoveryPlan,
        payload: str | bytes,
        *,
        config: RecoveryMachineConfig | None = None,
        driver: ActionDriver | None = None,
    ) -> DeterministicRecoveryExecutor:
        payload_bytes = payload.encode() if isinstance(payload, str) else payload
        if len(payload_bytes) > MAX_RECOVERY_SNAPSHOT_BYTES:
            raise ValueError("recovery snapshot exceeds bounded input size")
        machine = RecoveryStateMachine(plan, config)
        snapshot = machine.restore(payload_bytes)
        return cls(
            plan,
            now_ms=snapshot.started_at_ms,
            config=config,
            driver=driver,
            snapshot=snapshot,
        )
