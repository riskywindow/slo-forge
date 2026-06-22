"""Auditable candidate lifecycle and bounded canonical event persistence."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from sloforge.genesis.ir import (
    BudgetUsage,
    Candidate,
    CandidateState,
    CandidateSuccessState,
    LifecycleEvent,
    SearchBudget,
    canonical_json,
    write_canonical,
)

from .models import CandidateDesign, SearchEvent

MAXIMUM_EVENT_BYTES = 1024 * 1024


class CanonicalEventStore:
    def __init__(self, path: Path, *, maximum_events: int) -> None:
        if maximum_events <= 0:
            raise ValueError("maximum_events must be positive")
        self.path = path
        self.maximum_events = maximum_events
        self._events: list[SearchEvent] = []
        if path.is_file():
            payload = path.read_bytes()
            if len(payload) > maximum_events * MAXIMUM_EVENT_BYTES:
                raise ValueError("existing search event log exceeds its configured bound")
            for line in payload.splitlines():
                if len(line) > MAXIMUM_EVENT_BYTES:
                    raise ValueError("search event exceeds maximum canonical event size")
                self._events.append(SearchEvent.model_validate_json(line, strict=True))
            if any(event.sequence != index for index, event in enumerate(self._events)):
                raise ValueError("search event sequence is not contiguous")

    @property
    def events(self) -> tuple[SearchEvent, ...]:
        return tuple(self._events)

    def append(self, event: SearchEvent) -> None:
        if len(self._events) >= self.maximum_events:
            raise RuntimeError("bounded search event store is full")
        if event.sequence != len(self._events):
            raise ValueError("search event sequence must be contiguous")
        encoded = canonical_json(event)
        if len(encoded) > MAXIMUM_EVENT_BYTES:
            raise ValueError("search event exceeds maximum canonical event size")
        self._events.append(event)
        self._write_all()

    def _write_all(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                for event in self._events:
                    handle.write(canonical_json(event) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class CandidateRepository:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, candidate_id: str) -> Path:
        name = hashlib.sha256(candidate_id.encode()).hexdigest()
        return self.directory / f"candidate-{name}.json"

    def save(self, candidate: Candidate) -> Path:
        path = self.path_for(candidate.candidate_id)
        write_canonical(candidate, path)
        return path


def proposed_candidate(
    design: CandidateDesign,
    *,
    budget: SearchBudget,
    usage: BudgetUsage,
) -> Candidate:
    event = LifecycleEvent(
        sequence=0,
        from_state=None,
        to_state=CandidateSuccessState.PROPOSED,
        reason=f"proposed by deterministic {design.proposal_engine} engine",
    )
    return Candidate(
        candidate_id=design.candidate_id,
        seed=design.seed,
        genome_hash=design.genome_hash,
        parent_candidate_ids=design.parent_candidate_ids,
        transformation_ids=tuple(mutation.transformation_id for mutation in design.mutations),
        state=CandidateSuccessState.PROPOSED,
        lifecycle=(event,),
        budget=budget,
        usage=usage,
    )


def with_usage(candidate: Candidate, usage: BudgetUsage) -> Candidate:
    return Candidate(
        candidate_id=candidate.candidate_id,
        seed=candidate.seed,
        genome_hash=candidate.genome_hash,
        parent_candidate_ids=candidate.parent_candidate_ids,
        transformation_ids=candidate.transformation_ids,
        state=candidate.state,
        lifecycle=candidate.lifecycle,
        budget=candidate.budget,
        usage=usage,
        extensions=candidate.extensions,
    )


def transition(candidate: Candidate, target: CandidateState, reason: str) -> Candidate:
    event = LifecycleEvent(
        sequence=len(candidate.lifecycle),
        from_state=candidate.state,
        to_state=target,
        reason=reason,
    )
    return Candidate(
        candidate_id=candidate.candidate_id,
        seed=candidate.seed,
        genome_hash=candidate.genome_hash,
        parent_candidate_ids=candidate.parent_candidate_ids,
        transformation_ids=candidate.transformation_ids,
        state=target,
        lifecycle=(*candidate.lifecycle, event),
        budget=candidate.budget,
        usage=candidate.usage,
        extensions=candidate.extensions,
    )
