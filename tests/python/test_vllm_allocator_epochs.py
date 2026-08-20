from __future__ import annotations

from types import SimpleNamespace

import pytest

from sloforge.continuum.adapters.vllm_allocator_epochs import (
    VllmAllocatorEpochError,
    install_vllm_allocator_epoch_source,
    require_vllm_allocator_epoch_source,
)


class _Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash = None
        self.is_null = False


class _Allocator:
    def __init__(self) -> None:
        self.block_pool = SimpleNamespace(blocks=[_Block(index) for index in range(8)])
        self.tables: dict[str, tuple[_Block, ...]] = {}
        self.epochs = install_vllm_allocator_epoch_source(self)

    def get_blocks(self, request_id: str) -> object:
        return SimpleNamespace(blocks=(self.tables[request_id],))

    def allocate(self, request_id: str, block_ids: tuple[int, ...]) -> object:
        request = SimpleNamespace(request_id=request_id)
        blocks = tuple(self.block_pool.blocks[item] for item in block_ids)
        for block in blocks:
            block.ref_cnt += 1
        self.tables[request_id] = blocks
        self.epochs.issue_from_runtime_allocation(
            request,
            previous_block_ids=(),
            cached_block_ids=(),
            allocator_returned_block_ids=block_ids,
        )
        return request

    def free(self, request: object) -> None:
        blocks = self.tables.pop(request.request_id)
        for block in blocks:
            block.ref_cnt -= 1
            block.block_hash = None
        self.epochs.observe_runtime_free(
            request,
            released_block_ids=tuple(block.block_id for block in blocks),
        )


def test_valid_allocator_allocation_is_runtime_issued_and_consumed_once() -> None:
    allocator = _Allocator()
    allocator.allocate("restore.1", (2, 3))
    proof = allocator.epochs.proof_for_live_blocks((2, 3), owner_request_ids=("restore.1",))

    assert proof.issued_by_runtime_allocation_path
    assert proof.allocation_epoch_by_block[2] > 0
    assert proof.allocation_epoch_by_block[3] > proof.allocation_epoch_by_block[2]
    allocator.epochs.consume_once(proof, expected_block_ids=(2, 3))
    with pytest.raises(VllmAllocatorEpochError, match=r"already used|already consumed"):
        allocator.epochs.consume_once(proof, expected_block_ids=(2, 3))


def test_freed_allocation_and_stale_proof_fail_closed() -> None:
    allocator = _Allocator()
    request = allocator.allocate("restore.freed", (4,))
    proof = allocator.epochs.proof_for_live_blocks((4,), owner_request_ids=("restore.freed",))
    record = proof.records[0]

    allocator.free(request)

    allocator.epochs.require_released((record,))
    with pytest.raises(VllmAllocatorEpochError, match=r"unavailable|unowned|stale"):
        allocator.epochs.require_current(proof, expected_block_ids=(4,))


def test_reused_physical_block_id_receives_a_different_epoch() -> None:
    allocator = _Allocator()
    first_request = allocator.allocate("restore.old", (5,))
    old_proof = allocator.epochs.proof_for_live_blocks((5,), owner_request_ids=("restore.old",))
    allocator.free(first_request)

    allocator.allocate("restore.new", (5,))
    new_proof = allocator.epochs.proof_for_live_blocks((5,), owner_request_ids=("restore.new",))

    assert new_proof.records[0].allocation_epoch > old_proof.records[0].allocation_epoch
    with pytest.raises(VllmAllocatorEpochError, match=r"stale|unavailable"):
        allocator.epochs.require_current(old_proof, expected_block_ids=(5,))


def test_missing_runtime_epoch_source_is_rejected() -> None:
    with pytest.raises(VllmAllocatorEpochError, match="runtime allocator epoch source"):
        require_vllm_allocator_epoch_source(SimpleNamespace())
