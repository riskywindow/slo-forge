"""Explicit adapters from Fabric resource identifiers to Helix capacity units."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from sloforge.fabric.simulation import FabricSimulationRequest

from .models import Identifier, ResourceVector, SchedulerModel


class FabricResourceBinding(SchedulerModel):
    """Caller-declared unit conversion; the adapter never guesses physical units."""

    fabric_resource_id: Identifier
    helix_capacity: ResourceVector

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.helix_capacity.is_zero():
            raise ValueError("Fabric binding must contribute nonzero Helix capacity")
        return self


class FabricCapacityMapping(SchedulerModel):
    bindings: Annotated[tuple[FabricResourceBinding, ...], Field(min_length=1, max_length=100_000)]

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        identifiers = [binding.fabric_resource_id for binding in self.bindings]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Fabric resource bindings must be unique")
        return self


def capacity_from_fabric(
    request: FabricSimulationRequest, mapping: FabricCapacityMapping
) -> ResourceVector:
    """Validate resource identities and sum an explicit, non-inferential mapping."""

    known = {resource.id for resource in request.resources}
    bound = {binding.fabric_resource_id for binding in mapping.bindings}
    unknown = sorted(bound - known)
    if unknown:
        raise ValueError("Fabric capacity mapping references unknown physical resources")
    total = ResourceVector.zero()
    for binding in sorted(mapping.bindings, key=lambda item: item.fabric_resource_id):
        total = total.add(binding.helix_capacity)
    return total


__all__ = ["FabricCapacityMapping", "FabricResourceBinding", "capacity_from_fabric"]
