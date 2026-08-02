"""Canonical learned-constraint storage and repeated-failure suppression."""

from __future__ import annotations

from pathlib import Path

from sloforge.genesis.ir import write_canonical
from sloforge.genesis.search import CandidateDesign

from .models import ConstraintDocument, GeneralizedConstraint


class ConstraintStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.is_file():
            self._document = ConstraintDocument.model_validate_json(path.read_bytes(), strict=True)
        else:
            self._document = ConstraintDocument()

    @property
    def constraints(self) -> tuple[GeneralizedConstraint, ...]:
        return self._document.constraints

    def add(self, constraint: GeneralizedConstraint) -> None:
        if any(
            item.learned.constraint_id == constraint.learned.constraint_id
            for item in self._document.constraints
        ):
            return
        self._document = ConstraintDocument(constraints=(*self._document.constraints, constraint))
        write_canonical(self._document, self.path)

    def rejecting_constraint(self, candidate: CandidateDesign) -> GeneralizedConstraint | None:
        return next((item for item in self.constraints if item.rejects(candidate)), None)
