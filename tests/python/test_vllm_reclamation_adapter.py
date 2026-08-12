from __future__ import annotations

from collections.abc import Iterator

import pytest

from sloforge.continuum.adapters.vllm_reclamation import (
    CanonicalBranchTable,
    CanonicalKvTransportManifest,
    CanonicalKvTransportState,
    NativeKvGeometry,
    Vllm0230RestoreStager,
    copy_host_transport_to_canonical_device,
    native_pages_to_host_transport,
    restore_host_transport_to_native_pages,
    token_history_sha256,
)

torch = pytest.importorskip("torch")


def _identity() -> dict[str, str]:
    return {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a" * 40,
        "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
        "tokenizer_revision": "a" * 40,
        "dtype": "bfloat16",
        "policy_epoch": "policy-1",
    }


def _geometry(*, block_dims: tuple[int, int] = (0, 1)) -> NativeKvGeometry:
    return NativeKvGeometry(
        layer_names=("model.layers.0.attn", "model.layers.1.attn"),
        num_blocks=6,
        block_size_tokens=2,
        kv_heads=2,
        head_size=2,
        element_size_bytes=2,
        block_dimensions=block_dims,
        source_axis_labels=("kv", "token", "head", "dim"),
    )


def _tensors() -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.arange(6 * 2 * 2 * 2 * 2, dtype=torch.bfloat16).view(6, 2, 2, 2, 2)
    # Exercise a non-leading native block dimension and non-contiguous view.
    second_base = (1000 + torch.arange(6 * 2 * 2 * 2 * 2)).to(torch.bfloat16)
    second = second_base.view(2, 6, 2, 2, 2)
    return first, second


def _branches() -> tuple[CanonicalBranchTable, CanonicalBranchTable]:
    left_tokens = (10, 11, 12, 13)
    right_tokens = (10, 11, 20, 21)
    return (
        CanonicalBranchTable(
            logical_branch_id="branch.0",
            parent_logical_branch_id="root",
            token_ids=left_tokens,
            token_history_sha256=token_history_sha256(left_tokens),
            computed_tokens=4,
            logical_page_ids=("page.root", "page.left"),
        ),
        CanonicalBranchTable(
            logical_branch_id="branch.1",
            parent_logical_branch_id="root",
            token_ids=right_tokens,
            token_history_sha256=token_history_sha256(right_tokens),
            computed_tokens=3,
            logical_page_ids=("page.root", "page.right"),
        ),
    )


def test_native_canonical_fresh_native_round_trip_with_partial_page() -> None:
    geometry = _geometry()
    source = _tensors()
    state = native_pages_to_host_transport(
        source,
        geometry,
        page_order=(
            ("page.root", 1, 2, ("branch.0", "branch.1")),
            ("page.left", 2, 2, ("branch.0",)),
            ("page.right", 3, 1, ("branch.1",)),
        ),
        branch_tables=_branches(),
        identity=_identity(),
        pin_memory=False,
    )
    state.verify()
    assert state.manifest.logical_state_bytes == (2 + 2 + 1) * geometry.logical_token_bytes
    assert state.manifest.physical_source_bytes == 3 * geometry.physical_page_bytes
    assert all("vllm" not in page.logical_page_id for page in state.manifest.pages)

    first_destination = torch.full_like(source[0], -1)
    second_destination = torch.full_like(source[1], -1)
    restore_host_transport_to_native_pages(
        state,
        (first_destination, second_destination),
        geometry,
        destination_block_indices={"page.root": 4, "page.left": 5, "page.right": 0},
        expected_identity=_identity(),
    )
    assert torch.equal(first_destination[4], source[0][1])
    assert torch.equal(first_destination[5], source[0][2])
    assert torch.equal(first_destination[0, :, 0], source[0][3, :, 0])
    assert torch.count_nonzero(first_destination[0, :, 1]) == 0
    assert torch.equal(second_destination[:, 4], source[1][:, 1])
    assert torch.equal(second_destination[:, 5], source[1][:, 2])
    assert torch.equal(second_destination[:, 0, 0], source[1][:, 3, 0])
    assert torch.count_nonzero(second_destination[:, 0, 1]) == 0


def test_transport_corruption_fails_closed() -> None:
    state = native_pages_to_host_transport(
        _tensors(),
        _geometry(),
        page_order=(("page.root", 1, 2, ("branch.0", "branch.1")),),
        branch_tables=tuple(
            CanonicalBranchTable(
                logical_branch_id=branch.logical_branch_id,
                parent_logical_branch_id=branch.parent_logical_branch_id,
                token_ids=branch.token_ids[:3],
                token_history_sha256=token_history_sha256(branch.token_ids[:3]),
                computed_tokens=2,
                logical_page_ids=("page.root",),
            )
            for branch in _branches()
        ),
        identity=_identity(),
        pin_memory=False,
    )
    corrupt = state.payload.clone()
    corrupt[0] ^= 1
    with pytest.raises(ValueError, match="payload digest"):
        CanonicalKvTransportState(state.manifest, corrupt).verify()


def test_separately_validated_h2d_can_disable_redundant_transport_verify() -> None:
    class Payload:
        def __init__(self) -> None:
            self.copies = 0

        def to(self, *, device: str, non_blocking: bool) -> str:
            assert device == "cuda"
            assert non_blocking is False
            self.copies += 1
            return "device-payload"

    class State:
        def __init__(self) -> None:
            self.payload = Payload()
            self.verifications = 0

        def verify(self) -> None:
            self.verifications += 1

    state = State()
    assert copy_host_transport_to_canonical_device(state, verify=False) == "device-payload"
    assert state.verifications == 0
    assert state.payload.copies == 1


def test_destination_map_rejects_missing_or_duplicate_pages() -> None:
    state = native_pages_to_host_transport(
        _tensors(),
        _geometry(),
        page_order=(
            ("page.root", 1, 2, ("branch.0", "branch.1")),
            ("page.left", 2, 2, ("branch.0",)),
            ("page.right", 3, 1, ("branch.1",)),
        ),
        branch_tables=_branches(),
        identity=_identity(),
        pin_memory=False,
    )
    with pytest.raises(ValueError, match="cover each logical page"):
        restore_host_transport_to_native_pages(
            state,
            tuple(torch.zeros_like(tensor) for tensor in _tensors()),
            _geometry(),
            destination_block_indices={"page.root": 4},
            expected_identity=_identity(),
        )
    with pytest.raises(ValueError, match="unique and in range"):
        restore_host_transport_to_native_pages(
            state,
            tuple(torch.zeros_like(tensor) for tensor in _tensors()),
            _geometry(),
            destination_block_indices={
                "page.root": 4,
                "page.left": 4,
                "page.right": 5,
            },
            expected_identity=_identity(),
        )


def test_restore_supports_explicit_disjoint_staging_subsets() -> None:
    state = _transport_state()
    destination = tuple(torch.full_like(tensor, -1) for tensor in _tensors())
    restore_host_transport_to_native_pages(
        state,
        destination,
        _geometry(),
        destination_block_indices={"page.root": 4, "page.left": 5},
        expected_identity=_identity(),
        require_complete=False,
    )
    source = _tensors()
    assert torch.equal(destination[0][4], source[0][1])
    assert torch.equal(destination[0][5], source[0][2])
    assert torch.count_nonzero(destination[0][0] + 1) == 0


class _Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.is_null = False


class _Blocks:
    def __init__(self, blocks: list[_Block]) -> None:
        self.blocks = (blocks,)


class _ReadOnlyTokens:
    def __init__(self, values: tuple[int, ...]) -> None:
        self._values = values

    def __iter__(self) -> Iterator[int]:
        return iter(self._values)


class _Request:
    def __init__(
        self,
        request_id: str,
        all_token_ids: tuple[int, ...],
        *,
        read_only_tokens: bool = False,
    ) -> None:
        self.request_id = request_id
        self.all_token_ids: list[int] | _ReadOnlyTokens = (
            _ReadOnlyTokens(all_token_ids) if read_only_tokens else list(all_token_ids)
        )
        self.num_computed_tokens = 0


class _Queue:
    def __init__(self, requests: list[_Request]) -> None:
        self.requests = list(requests)

    def remove_requests(self, requests: list[_Request]) -> None:
        for request in requests:
            if request in self.requests:
                self.requests.remove(request)

    def add_request(self, request: _Request) -> None:
        self.requests.append(request)


class _Manager:
    def __init__(self) -> None:
        self.tables: dict[str, list[_Block]] = {}
        self.zero_queue: list[int] = []
        self.next_block = 10
        self.cached_root: _Block | None = None
        self.reset = False

    def take_new_block_ids(self) -> list[int]:
        result = self.zero_queue
        self.zero_queue = []
        return result

    def allocate_slots(
        self,
        request: _Request,
        *,
        num_new_tokens: int,
        num_new_computed_tokens: int,
        new_computed_blocks: _Blocks | None,
        num_external_computed_tokens: int,
        **_kwargs: object,
    ) -> _Blocks:
        assert num_new_tokens == 0
        prefix = [] if new_computed_blocks is None else list(new_computed_blocks.blocks[0])
        total = num_new_computed_tokens + num_external_computed_tokens
        required = (total + 1) // 2
        new = []
        while len(prefix) + len(new) < required:
            block = _Block(self.next_block)
            self.next_block += 1
            new.append(block)
            self.zero_queue.append(block.block_id)
        self.tables[request.request_id] = prefix + new
        return _Blocks(new)

    def get_blocks(self, request_id: str) -> _Blocks:
        return _Blocks(self.tables[request_id])

    def get_computed_blocks(self, _request: _Request) -> tuple[_Blocks, int]:
        assert self.cached_root is not None
        return _Blocks([self.cached_root]), 2

    def cache_blocks(self, request: _Request, _computed_tokens: int) -> None:
        if self.cached_root is None:
            self.cached_root = self.tables[request.request_id][0]

    def free(self, request: _Request) -> None:
        self.tables.pop(request.request_id, None)

    def reset_prefix_cache(self) -> bool:
        self.cached_root = None
        self.reset = True
        return True


class _Scheduler:
    def __init__(
        self,
        requests: list[_Request],
        manager: _Manager,
        *,
        needs_kv_cache_zeroing: bool = False,
    ) -> None:
        self.requests = {request.request_id: request for request in requests}
        self.waiting = _Queue(requests)
        self.skipped_waiting = _Queue([])
        self.running: list[_Request] = []
        self.scheduler_config = type("Config", (), {"async_scheduling": False})()
        self.kv_cache_manager = manager
        self.needs_kv_cache_zeroing = needs_kv_cache_zeroing

    def _enqueue_waiting_request(self, request: _Request) -> None:
        self.waiting.add_request(request)


def _transport_state() -> CanonicalKvTransportState:
    return native_pages_to_host_transport(
        _tensors(),
        _geometry(),
        page_order=(
            ("page.root", 1, 2, ("branch.0", "branch.1")),
            ("page.left", 2, 2, ("branch.0",)),
            ("page.right", 3, 1, ("branch.1",)),
        ),
        branch_tables=_branches(),
        identity=_identity(),
        pin_memory=False,
    )


def test_restore_stager_drains_zero_queue_and_admits_only_after_group_validation() -> None:
    requests = [
        _Request(
            "branch.0@restore-1",
            _branches()[0].token_ids,
            read_only_tokens=True,
        ),
        _Request("branch.1@restore-1", _branches()[1].token_ids),
    ]
    manager = _Manager()
    scheduler = _Scheduler(requests, manager)
    writes: list[dict[str, int]] = []
    validations: list[dict[str, int]] = []
    allocations: list[tuple[str, int, int]] = []
    result = Vllm0230RestoreStager(scheduler, manager).import_group(
        _transport_state(),
        runtime_request_ids={
            "branch.0": "branch.0@restore-1",
            "branch.1": "branch.1@restore-1",
        },
        expected_identity=_identity(),
        write_pages=lambda mapping: writes.append(dict(mapping)),
        validate_pages=lambda mapping: validations.append(dict(mapping)) or True,
        allocation_observer=lambda branch, start, end: allocations.append((branch, start, end)),
    )
    assert result.logical_page_destinations == {
        "page.root": 10,
        "page.left": 11,
        "page.right": 12,
    }
    assert result.zero_queue_drained_block_indices == (10, 11, 12)
    assert writes == [
        {"page.root": 10, "page.left": 11},
        {"page.right": 12},
    ]
    assert validations == [
        {"page.root": 10, "page.left": 11},
        {"page.right": 12},
    ]
    assert [request.request_id for request in scheduler.waiting.requests] == [
        "branch.0@restore-1",
        "branch.1@restore-1",
    ]
    assert [request.num_computed_tokens for request in requests] == [4, 3]
    assert [branch for branch, _start, _end in allocations] == ["branch.0", "branch.1"]
    assert all(end >= start for _branch, start, end in allocations)


def test_restore_stager_validation_failure_frees_pages_and_never_admits() -> None:
    requests = [
        _Request("branch.0@restore-1", _branches()[0].token_ids),
        _Request("branch.1@restore-1", _branches()[1].token_ids),
    ]
    manager = _Manager()
    scheduler = _Scheduler(requests, manager)
    with pytest.raises(ValueError, match="destination KV validation failed"):
        Vllm0230RestoreStager(scheduler, manager).import_group(
            _transport_state(),
            runtime_request_ids={
                "branch.0": "branch.0@restore-1",
                "branch.1": "branch.1@restore-1",
            },
            expected_identity=_identity(),
            write_pages=lambda _mapping: None,
            validate_pages=lambda _mapping: False,
        )
    assert scheduler.waiting.requests == []
    assert scheduler.requests == {}
    assert manager.tables == {}
    assert manager.reset


def test_restore_stager_rejects_cross_page_destination_alias_and_frees_group() -> None:
    class AliasingManager(_Manager):
        def get_computed_blocks(self, request: _Request) -> tuple[_Blocks, int]:
            # Reissue the first branch's private slot for the second branch.
            # A correct allocator never does this; the stager must nevertheless
            # reject the corrupt table before publishing the complete group.
            self.next_block = 11
            return super().get_computed_blocks(request)

    requests = [
        _Request("branch.0@restore-1", _branches()[0].token_ids),
        _Request("branch.1@restore-1", _branches()[1].token_ids),
    ]
    manager = AliasingManager()
    scheduler = _Scheduler(requests, manager)

    with pytest.raises(RuntimeError, match="distinct logical pages alias"):
        Vllm0230RestoreStager(scheduler, manager).import_group(
            _transport_state(),
            runtime_request_ids={
                "branch.0": "branch.0@restore-1",
                "branch.1": "branch.1@restore-1",
            },
            expected_identity=_identity(),
            write_pages=lambda _mapping: None,
            validate_pages=lambda _mapping: True,
        )

    assert scheduler.waiting.requests == []
    assert scheduler.requests == {}
    assert manager.tables == {}
    assert manager.reset


def test_restore_stager_ignores_stale_zero_queue_only_when_zeroing_is_disabled() -> None:
    requests = [
        _Request("branch.0@restore-1", _branches()[0].token_ids),
        _Request("branch.1@restore-1", _branches()[1].token_ids),
    ]
    manager = _Manager()
    manager.zero_queue = [99]
    scheduler = _Scheduler(requests, manager, needs_kv_cache_zeroing=False)

    result = Vllm0230RestoreStager(scheduler, manager).import_group(
        _transport_state(),
        runtime_request_ids={
            "branch.0": "branch.0@restore-1",
            "branch.1": "branch.1@restore-1",
        },
        expected_identity=_identity(),
        write_pages=lambda _mapping: None,
        validate_pages=lambda _mapping: True,
    )
    assert result.zero_queue_drained_block_indices == (10, 11, 12)

    strict_requests = [
        _Request("branch.0@restore-1", _branches()[0].token_ids),
        _Request("branch.1@restore-1", _branches()[1].token_ids),
    ]
    strict_manager = _Manager()
    strict_manager.zero_queue = [99]
    strict_scheduler = _Scheduler(
        strict_requests,
        strict_manager,
        needs_kv_cache_zeroing=True,
    )
    with pytest.raises(RuntimeError, match="unrelated vLLM pages"):
        Vllm0230RestoreStager(strict_scheduler, strict_manager).import_group(
            _transport_state(),
            runtime_request_ids={
                "branch.0": "branch.0@restore-1",
                "branch.1": "branch.1@restore-1",
            },
            expected_identity=_identity(),
            write_pages=lambda _mapping: None,
            validate_pages=lambda _mapping: True,
        )


def test_manifest_rejects_incomplete_page_coverage_and_false_ownership() -> None:
    state = _transport_state()
    data = state.manifest.model_dump(mode="python")
    data["branches"][0]["logical_page_ids"] = ("page.root",)
    with pytest.raises(ValueError, match="exactly cover"):
        CanonicalKvTransportManifest.model_validate(data, strict=True)

    data = state.manifest.model_dump(mode="python")
    data["pages"][1]["branch_ids"] = ("branch.1",)
    with pytest.raises(ValueError, match="ownership differs"):
        CanonicalKvTransportManifest.model_validate(data, strict=True)


def test_restore_rejects_runtime_identity_or_token_history_mismatch_before_allocation() -> None:
    state = _transport_state()
    destination = tuple(torch.zeros_like(tensor) for tensor in _tensors())
    wrong_identity = _identity() | {"policy_epoch": "wrong-policy"}
    with pytest.raises(ValueError, match="identity differs"):
        restore_host_transport_to_native_pages(
            state,
            destination,
            _geometry(),
            destination_block_indices={"page.root": 4, "page.left": 5, "page.right": 0},
            expected_identity=wrong_identity,
        )

    requests = [
        _Request("branch.0@restore-1", (10, 11, 999, 13)),
        _Request("branch.1@restore-1", _branches()[1].token_ids),
    ]
    manager = _Manager()
    scheduler = _Scheduler(requests, manager)
    with pytest.raises(RuntimeError, match="token history differs"):
        Vllm0230RestoreStager(scheduler, manager).import_group(
            state,
            runtime_request_ids={
                "branch.0": "branch.0@restore-1",
                "branch.1": "branch.1@restore-1",
            },
            expected_identity=_identity(),
            write_pages=lambda _mapping: None,
            validate_pages=lambda _mapping: True,
        )
    assert manager.tables == {}
    assert scheduler.waiting.requests == []


def test_restore_never_publishes_prefix_before_destination_validation() -> None:
    events: list[str] = []

    class RecordingManager(_Manager):
        def cache_blocks(self, request: _Request, computed_tokens: int) -> None:
            events.append(f"cache:{request.request_id}")
            super().cache_blocks(request, computed_tokens)

    requests = [
        _Request("branch.0@restore-1", _branches()[0].token_ids),
        _Request("branch.1@restore-1", _branches()[1].token_ids),
    ]
    manager = RecordingManager()
    scheduler = _Scheduler(requests, manager)
    Vllm0230RestoreStager(scheduler, manager).import_group(
        _transport_state(),
        runtime_request_ids={
            "branch.0": "branch.0@restore-1",
            "branch.1": "branch.1@restore-1",
        },
        expected_identity=_identity(),
        write_pages=lambda mapping: events.append(f"write:{','.join(sorted(mapping))}"),
        validate_pages=lambda mapping: (
            events.append(f"validate:{','.join(sorted(mapping))}") or True
        ),
    )
    first_cache = events.index("cache:branch.0@restore-1")
    assert events[:first_cache] == [
        "write:page.left,page.root",
        "validate:page.left,page.root",
    ]


def test_partial_admission_failure_removes_entire_restore_group() -> None:
    class FailingScheduler(_Scheduler):
        def __init__(self, requests: list[_Request], manager: _Manager) -> None:
            super().__init__(requests, manager)
            self.enqueue_count = 0

        def _enqueue_waiting_request(self, request: _Request) -> None:
            self.enqueue_count += 1
            if self.enqueue_count == 2:
                raise RuntimeError("injected enqueue failure")
            super()._enqueue_waiting_request(request)

    requests = [
        _Request("branch.0@restore-1", _branches()[0].token_ids),
        _Request("branch.1@restore-1", _branches()[1].token_ids),
    ]
    manager = _Manager()
    scheduler = FailingScheduler(requests, manager)
    with pytest.raises(RuntimeError, match="injected enqueue failure"):
        Vllm0230RestoreStager(scheduler, manager).import_group(
            _transport_state(),
            runtime_request_ids={
                "branch.0": "branch.0@restore-1",
                "branch.1": "branch.1@restore-1",
            },
            expected_identity=_identity(),
            write_pages=lambda _mapping: None,
            validate_pages=lambda _mapping: True,
        )
    assert scheduler.waiting.requests == []
    assert scheduler.requests == {}
    assert manager.tables == {}
    assert manager.reset
