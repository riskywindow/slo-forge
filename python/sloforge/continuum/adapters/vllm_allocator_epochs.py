"""Allocator-side lifetime identities for the pinned vLLM 0.23 runtime.

vLLM 0.23 exposes reusable integer block IDs but no allocation generation.
SLOForge therefore installs this source *inside* the manager's
``allocate_slots``/``free`` interception path.  Import code has read/consume
access only: it cannot choose an epoch.  A generation names one native block
allocation lifetime, remains tombstoned after free, advances before a reused
block ID can be admitted, and is consumed at most once by a restore.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_ALLOCATOR_EPOCH_SOURCE_ATTRIBUTE = "_sloforge_vllm0230_allocator_epoch_source"
_PROOF_SEAL = object()


class VllmAllocatorEpochError(RuntimeError):
    """Allocator lifetime evidence was absent, stale, forged, or reused."""


@dataclass(frozen=True, slots=True)
class VllmAllocatorIssuedEpoch:
    block_id: int
    allocation_epoch: int
    issued_for_request_id: str
    issued_at_monotonic_ns: int

    def __post_init__(self) -> None:
        if self.block_id < 0 or self.allocation_epoch <= 0:
            raise ValueError("allocator-issued block identity is invalid")
        if not self.issued_for_request_id or self.issued_at_monotonic_ns <= 0:
            raise ValueError("allocator-issued block identity lacks provenance")


@dataclass(frozen=True, slots=True)
class VllmAllocatorEpochProof:
    records: tuple[VllmAllocatorIssuedEpoch, ...]
    owner_request_ids: tuple[str, ...]
    issued_by_runtime_allocation_path: bool
    _source: object = field(repr=False, compare=False)
    _proof_seal: object = field(repr=False, compare=False)

    @property
    def allocation_epoch_by_block(self) -> dict[int, int]:
        return {record.block_id: record.allocation_epoch for record in self.records}


class Vllm0230AllocatorEpochSource:
    """Monotonic, allocator-bound generation source attached to one manager."""

    def __init__(self, manager: Any) -> None:
        if getattr(manager, _ALLOCATOR_EPOCH_SOURCE_ATTRIBUTE, None) is not None:
            raise ValueError("vLLM manager already has an allocator epoch source")
        self._manager = manager
        self._next_epoch = 0
        self._records: dict[int, VllmAllocatorIssuedEpoch] = {}
        self._live: set[tuple[int, int]] = set()
        self._consumed: set[tuple[int, int]] = set()
        self._request_blocks: dict[str, tuple[int, ...]] = {}
        setattr(manager, _ALLOCATOR_EPOCH_SOURCE_ATTRIBUTE, self)

    @staticmethod
    def _request_id(request: Any) -> str:
        request_id = getattr(request, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            raise VllmAllocatorEpochError("runtime allocation lacks a request identity")
        return request_id

    def _runtime_table(self, request_id: str) -> tuple[int, ...]:
        try:
            groups = self._manager.get_blocks(request_id).blocks
        except (AttributeError, KeyError, TypeError) as error:
            raise VllmAllocatorEpochError(
                f"runtime block table is unavailable for {request_id!r}"
            ) from error
        result: list[int] = []
        for group in groups:
            for block in group:
                if bool(getattr(block, "is_null", False)):
                    continue
                block_id = getattr(block, "block_id", None)
                if not isinstance(block_id, int) or block_id < 0:
                    raise VllmAllocatorEpochError("runtime returned an invalid block ID")
                result.append(block_id)
        if len(result) != len(set(result)):
            raise VllmAllocatorEpochError("runtime block table aliases one physical block")
        return tuple(result)

    def issue_from_runtime_allocation(
        self,
        request: Any,
        *,
        previous_block_ids: Sequence[int],
        cached_block_ids: Sequence[int],
        allocator_returned_block_ids: Sequence[int],
    ) -> tuple[VllmAllocatorIssuedEpoch, ...]:
        """Issue generations while unwinding the real ``allocate_slots`` call.

        Full-table delta is authoritative because vLLM's external-computed
        path may return an empty block collection after allocating pages.
        """

        request_id = self._request_id(request)
        table = self._runtime_table(request_id)
        previous = set(previous_block_ids)
        cached = set(cached_block_ids)
        returned = tuple(dict.fromkeys(int(item) for item in allocator_returned_block_ids))
        new_ids = list(returned)
        new_ids.extend(
            block_id
            for block_id in table
            if block_id not in previous and block_id not in cached and block_id not in new_ids
        )
        if any(block_id not in table for block_id in new_ids):
            raise VllmAllocatorEpochError(
                "allocator returned a block absent from the installed request table"
            )
        issued: list[VllmAllocatorIssuedEpoch] = []
        for block_id in new_ids:
            prior = self._records.get(block_id)
            if prior is not None:
                self._live.discard((block_id, prior.allocation_epoch))
            self._next_epoch += 1
            record = VllmAllocatorIssuedEpoch(
                block_id=block_id,
                allocation_epoch=self._next_epoch,
                issued_for_request_id=request_id,
                issued_at_monotonic_ns=time.monotonic_ns(),
            )
            self._records[block_id] = record
            self._live.add((block_id, record.allocation_epoch))
            issued.append(record)
        self._request_blocks[request_id] = table
        return tuple(issued)

    def observe_runtime_free(
        self,
        request: Any,
        *,
        released_block_ids: Sequence[int],
    ) -> None:
        """Tombstone allocations no longer owned or retained by native cache."""

        request_id = self._request_id(request)
        self._request_blocks.pop(request_id, None)
        still_owned = {
            block_id for block_ids in self._request_blocks.values() for block_id in block_ids
        }
        pool_blocks = getattr(getattr(self._manager, "block_pool", None), "blocks", ())
        for block_id in released_block_ids:
            record = self._records.get(int(block_id))
            if record is None or block_id in still_owned:
                continue
            retained_by_cache = False
            try:
                block = pool_blocks[int(block_id)]
                retained_by_cache = getattr(block, "block_hash", None) is not None
                native_refcount = int(block.ref_cnt)
            except (IndexError, TypeError, ValueError):
                native_refcount = 0
            if native_refcount <= 0 and not retained_by_cache:
                self._live.discard((record.block_id, record.allocation_epoch))

    def observe_prefix_cache_reset(self) -> None:
        """Tombstone every generation made allocator-available by cache reset."""

        pool_blocks = getattr(getattr(self._manager, "block_pool", None), "blocks", ())
        for block_id, record in self._records.items():
            try:
                block = pool_blocks[block_id]
                available = (
                    int(block.ref_cnt) == 0
                    and getattr(block, "block_hash", None) is None
                    and not bool(getattr(block, "is_null", False))
                )
            except (IndexError, TypeError, ValueError):
                available = False
            if available:
                self._live.discard((block_id, record.allocation_epoch))

    def records_for_blocks(self, block_ids: Sequence[int]) -> tuple[VllmAllocatorIssuedEpoch, ...]:
        result: list[VllmAllocatorIssuedEpoch] = []
        for block_id in block_ids:
            record = self._records.get(int(block_id))
            if record is None:
                raise VllmAllocatorEpochError(
                    f"block {block_id} has no allocator-issued allocation epoch"
                )
            result.append(record)
        return tuple(result)

    def require_released(self, records: Sequence[VllmAllocatorIssuedEpoch]) -> None:
        """Prove exact allocation lifetimes are tombstoned and native-free."""

        if not records:
            raise VllmAllocatorEpochError("release proof cannot be empty")
        pool_blocks = getattr(getattr(self._manager, "block_pool", None), "blocks", ())
        for record in records:
            current = self._records.get(record.block_id)
            if current != record:
                raise VllmAllocatorEpochError(
                    f"block {record.block_id} changed generation before release proof"
                )
            if (record.block_id, record.allocation_epoch) in self._live:
                raise VllmAllocatorEpochError(
                    f"block {record.block_id} allocation epoch remains live"
                )
            try:
                block = pool_blocks[record.block_id]
                native_released = (
                    int(block.ref_cnt) == 0
                    and getattr(block, "block_hash", None) is None
                    and not bool(getattr(block, "is_null", False))
                )
            except (IndexError, TypeError, ValueError):
                native_released = False
            if not native_released:
                raise VllmAllocatorEpochError(
                    f"block {record.block_id} is not allocator-available after release"
                )

    def proof_for_live_blocks(
        self,
        block_ids: Sequence[int],
        *,
        owner_request_ids: Sequence[str],
    ) -> VllmAllocatorEpochProof:
        ordered_ids = tuple(dict.fromkeys(int(item) for item in block_ids))
        owners = tuple(dict.fromkeys(owner_request_ids))
        if not ordered_ids or len(ordered_ids) != len(tuple(block_ids)):
            raise VllmAllocatorEpochError("allocator epoch proof requires unique blocks")
        if not owners or any(not item for item in owners):
            raise VllmAllocatorEpochError("allocator epoch proof requires runtime owners")
        runtime_owned: set[int] = set()
        for request_id in owners:
            current = self._runtime_table(request_id)
            self._request_blocks[request_id] = current
            runtime_owned.update(current)
        if not set(ordered_ids).issubset(runtime_owned):
            raise VllmAllocatorEpochError("allocator epoch proof includes an unowned block")
        records: list[VllmAllocatorIssuedEpoch] = []
        for block_id in ordered_ids:
            record = self._records.get(block_id)
            if record is None:
                raise VllmAllocatorEpochError(
                    f"block {block_id} has no allocator-issued allocation epoch"
                )
            key = (block_id, record.allocation_epoch)
            if key not in self._live:
                raise VllmAllocatorEpochError(f"block {block_id} allocation epoch is stale")
            if key in self._consumed:
                raise VllmAllocatorEpochError(f"block {block_id} allocation epoch was already used")
            records.append(record)
        return VllmAllocatorEpochProof(
            records=tuple(records),
            owner_request_ids=owners,
            issued_by_runtime_allocation_path=True,
            _source=self,
            _proof_seal=_PROOF_SEAL,
        )

    def require_current(
        self,
        proof: VllmAllocatorEpochProof,
        *,
        expected_block_ids: Sequence[int],
    ) -> None:
        if (
            not isinstance(proof, VllmAllocatorEpochProof)
            or proof._source is not self
            or proof._proof_seal is not _PROOF_SEAL
            or not proof.issued_by_runtime_allocation_path
        ):
            raise VllmAllocatorEpochError("allocator epoch proof is not runtime-issued")
        expected = tuple(dict.fromkeys(int(item) for item in expected_block_ids))
        actual = tuple(record.block_id for record in proof.records)
        if actual != expected:
            raise VllmAllocatorEpochError("allocator epoch proof differs from destination blocks")
        refreshed = self.proof_for_live_blocks(
            expected,
            owner_request_ids=proof.owner_request_ids,
        )
        if refreshed.records != proof.records:
            raise VllmAllocatorEpochError("allocator epoch proof became stale")

    def consume_once(
        self,
        proof: VllmAllocatorEpochProof,
        *,
        expected_block_ids: Sequence[int],
    ) -> None:
        self.require_current(proof, expected_block_ids=expected_block_ids)
        keys = {(record.block_id, record.allocation_epoch) for record in proof.records}
        if keys & self._consumed:
            raise VllmAllocatorEpochError("allocator epoch proof was already consumed")
        self._consumed.update(keys)

    def epoch_by_block(self) -> Mapping[int, int]:
        return {block_id: record.allocation_epoch for block_id, record in self._records.items()}


def install_vllm_allocator_epoch_source(manager: Any) -> Vllm0230AllocatorEpochSource:
    existing = getattr(manager, _ALLOCATOR_EPOCH_SOURCE_ATTRIBUTE, None)
    if existing is not None:
        if not isinstance(existing, Vllm0230AllocatorEpochSource):
            raise VllmAllocatorEpochError("vLLM manager epoch source has an invalid type")
        return existing
    return Vllm0230AllocatorEpochSource(manager)


def require_vllm_allocator_epoch_source(manager: Any) -> Vllm0230AllocatorEpochSource:
    source = getattr(manager, _ALLOCATOR_EPOCH_SOURCE_ATTRIBUTE, None)
    if not isinstance(source, Vllm0230AllocatorEpochSource):
        raise VllmAllocatorEpochError(
            "v11 production import requires the runtime allocator epoch source"
        )
    return source


__all__ = [
    "Vllm0230AllocatorEpochSource",
    "VllmAllocatorEpochError",
    "VllmAllocatorEpochProof",
    "VllmAllocatorIssuedEpoch",
    "install_vllm_allocator_epoch_source",
    "require_vllm_allocator_epoch_source",
]
