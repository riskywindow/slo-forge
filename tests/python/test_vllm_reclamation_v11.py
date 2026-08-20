from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

import sloforge.continuum.adapters.vllm_reclamation_v11 as v11_adapter
from sloforge.continuum.adapters.vllm_allocator_epochs import (
    VllmAllocatorEpochProof,
    install_vllm_allocator_epoch_source,
)
from sloforge.continuum.adapters.vllm_reclamation import (
    CanonicalBranchTable,
    CanonicalKvTransportState,
    NativeKvGeometry,
    native_pages_to_host_transport,
    restore_host_transport_to_native_pages,
    token_history_sha256,
)
from sloforge.continuum.adapters.vllm_reclamation_v11 import (
    V11IntegrityError,
    V11PipelineConfig,
    V11RuntimeTeardownRequired,
    V11SubsetValidationEvidence,
    Vllm0230EngineStepBinding,
    Vllm0230StreamingRestoreStager,
    capture_native_to_transport_v11,
    destination_target_identity_v11,
    plan_transport_chunks,
    restore_transport_to_native_v11,
    scrub_native_pages_v11,
    verify_transport_streaming_v11,
    write_and_validate_native_subset_v11,
)
from sloforge.continuum.adapters.vllm_reclamation_v11_accounting import (
    V11StatePassRecorder,
    validate_authoritative_pass_records,
)

torch = pytest.importorskip("torch")


def _identity() -> dict[str, str]:
    return {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a" * 40,
        "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
        "tokenizer_revision": "a" * 40,
        "dtype": "bfloat16",
        "policy_epoch": "policy-v11",
    }


def _fixture(
    seed: int,
    *,
    layer_count: int,
) -> tuple[
    NativeKvGeometry,
    tuple[torch.Tensor, ...],
    tuple[tuple[str, int, int, tuple[str, ...]], ...],
    tuple[CanonicalBranchTable, ...],
    dict[str, int],
]:
    rng = random.Random(seed)
    block_size = 4
    num_blocks = 48
    block_dims = tuple(rng.randrange(0, 5) for _ in range(layer_count))
    geometry = NativeKvGeometry(
        layer_names=tuple(f"model.layers.{index}.attn" for index in range(layer_count)),
        num_blocks=num_blocks,
        block_size_tokens=block_size,
        kv_heads=2,
        head_size=3,
        element_size_bytes=2,
        block_dimensions=block_dims,
        source_axis_labels=("kv", "token", "head", "dim"),
    )
    tensors: list[torch.Tensor] = []
    semantic_shape = [2, block_size, geometry.kv_heads, geometry.head_size]
    for layer_index, block_dim in enumerate(block_dims):
        shape = list(semantic_shape)
        shape.insert(block_dim, num_blocks)
        generator = torch.Generator().manual_seed(seed * 101 + layer_index)
        tensors.append(
            torch.randint(-2048, 2048, shape, generator=generator, dtype=torch.int32).to(
                torch.bfloat16
            )
        )

    branch_ids = ("branch.0", "branch.1", "branch.2")
    source_blocks = rng.sample(range(num_blocks), 8)
    shared_page_ids = ("page.shared.0", "page.shared.1")
    page_order: list[tuple[str, int, int, tuple[str, ...]]] = [
        (shared_page_ids[0], source_blocks[0], block_size, branch_ids),
        (shared_page_ids[1], source_blocks[1], block_size, branch_ids),
    ]
    private_by_branch: dict[str, tuple[str, str]] = {}
    block_cursor = 2
    for branch_id in branch_ids:
        private = (f"page.{branch_id}.0", f"page.{branch_id}.1")
        private_by_branch[branch_id] = private
        page_order.extend(
            (
                (private[0], source_blocks[block_cursor], block_size, (branch_id,)),
                (private[1], source_blocks[block_cursor + 1], 1, (branch_id,)),
            )
        )
        block_cursor += 2

    common_tokens = tuple(range(100, 100 + block_size * 2))
    branches: list[CanonicalBranchTable] = []
    for branch_index, branch_id in enumerate(branch_ids):
        private_tokens = tuple(
            1000 + 100 * branch_index + offset for offset in range(block_size + 1)
        )
        tokens = common_tokens + private_tokens
        branches.append(
            CanonicalBranchTable(
                logical_branch_id=branch_id,
                parent_logical_branch_id="root",
                token_ids=tokens,
                token_history_sha256=token_history_sha256(tokens),
                computed_tokens=len(tokens),
                logical_page_ids=shared_page_ids + private_by_branch[branch_id],
            )
        )
    destination_blocks = rng.sample(
        [item for item in range(num_blocks) if item not in source_blocks], len(page_order)
    )
    destinations = {
        logical_id: destination
        for (logical_id, _source, _valid, _owners), destination in zip(
            page_order, destination_blocks, strict=True
        )
    }
    return geometry, tuple(tensors), tuple(page_order), tuple(branches), destinations


@pytest.mark.parametrize(("seed", "layer_count"), [(41, 1), (73, 2), (113, 4)])
def test_randomized_v11_export_is_bitwise_v10_and_restore_is_logically_equal(
    seed: int,
    layer_count: int,
) -> None:
    geometry, tensors, page_order, branches, destinations = _fixture(seed, layer_count=layer_count)
    maximum_chunk_bytes = geometry.physical_page_bytes * 3
    config = V11PipelineConfig(
        seed=seed,
        expected_device_type="cpu",
        maximum_chunk_bytes=maximum_chunk_bytes,
    )
    naive = native_pages_to_host_transport(
        tensors,
        geometry,
        page_order=page_order,
        branch_tables=branches,
        identity=_identity(),
        pin_memory=False,
    )
    optimized = capture_native_to_transport_v11(
        tensors,
        geometry,
        page_order=page_order,
        branch_tables=branches,
        identity=_identity(),
        config=config,
    )

    assert torch.equal(optimized.state.payload, naive.payload)
    assert optimized.state.manifest == naive.manifest
    assert optimized.integrity.payload_sha256 == naive.manifest.payload_sha256
    assert optimized.stats.maximum_chunk_bytes <= maximum_chunk_bytes
    assert optimized.stats.peak_host_temporary_bytes == 0
    assert (
        sum(item.payload_bytes for item in optimized.chunks) == naive.manifest.logical_state_bytes
    )

    naive_destination = tuple(torch.full_like(tensor, -71) for tensor in tensors)
    optimized_destination = tuple(torch.full_like(tensor, -113) for tensor in tensors)
    restore_host_transport_to_native_pages(
        naive,
        naive_destination,
        geometry,
        destination_block_indices=destinations,
        expected_identity=_identity(),
    )
    restored = restore_transport_to_native_v11(
        optimized.state,
        optimized_destination,
        geometry,
        destination_block_indices=destinations,
        expected_identity=_identity(),
        config=config,
    )
    assert restored.validation.passed
    assert restored.validation.full_byte_coverage
    assert restored.stats.zero_tail_bytes > 0
    for naive_tensor, optimized_tensor, block_dim in zip(
        naive_destination,
        optimized_destination,
        geometry.block_dimensions,
        strict=True,
    ):
        naive_pool = naive_tensor.movedim(block_dim, 0)
        optimized_pool = optimized_tensor.movedim(block_dim, 0)
        for destination in destinations.values():
            assert torch.equal(optimized_pool[destination], naive_pool[destination])
        untouched = set(range(geometry.num_blocks)) - set(destinations.values())
        assert all(
            torch.count_nonzero(optimized_pool[index] + 113).item() == 0 for index in untouched
        )


def test_chunk_planner_rejects_duplicate_missing_and_oversized_input() -> None:
    geometry, _tensors, page_order, _branches, _destinations = _fixture(41, layer_count=1)
    duplicate = (*page_order, page_order[0])
    with pytest.raises(ValueError, match="duplicate logical"):
        plan_transport_chunks(
            duplicate,
            geometry,
            maximum_chunk_bytes=geometry.physical_page_bytes * 2,
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        plan_transport_chunks((), geometry, maximum_chunk_bytes=geometry.physical_page_bytes)
    with pytest.raises(ValueError, match="exceeds"):
        plan_transport_chunks(page_order, geometry, maximum_chunk_bytes=1)


def test_streaming_integrity_fails_on_corruption_without_writing_destination() -> None:
    geometry, tensors, page_order, branches, destinations = _fixture(73, layer_count=2)
    config = V11PipelineConfig(
        seed=73,
        expected_device_type="cpu",
        maximum_chunk_bytes=geometry.physical_page_bytes * 2,
    )
    captured = capture_native_to_transport_v11(
        tensors,
        geometry,
        page_order=page_order,
        branch_tables=branches,
        identity=_identity(),
        config=config,
    )
    corrupt = captured.state.payload.clone()
    corrupt[len(corrupt) // 2] ^= 1
    corrupted_state = CanonicalKvTransportState(captured.state.manifest, corrupt)
    destination = tuple(torch.full_like(tensor, -9) for tensor in tensors)
    with pytest.raises(V11IntegrityError, match="digest mismatch"):
        restore_transport_to_native_v11(
            corrupted_state,
            destination,
            geometry,
            destination_block_indices=destinations,
            expected_identity=_identity(),
            config=config,
        )
    assert all(torch.count_nonzero(tensor + 9).item() == 0 for tensor in destination)


def test_streaming_integrity_rejects_missing_duplicated_and_reordered_ranges() -> None:
    geometry, tensors, page_order, branches, _destinations = _fixture(73, layer_count=2)
    config = V11PipelineConfig(
        seed=73,
        expected_device_type="cpu",
        maximum_chunk_bytes=geometry.physical_page_bytes * 2,
    )
    captured = capture_native_to_transport_v11(
        tensors,
        geometry,
        page_order=page_order,
        branch_tables=branches,
        identity=_identity(),
        config=config,
    )
    missing = CanonicalKvTransportState(
        captured.state.manifest, captured.state.payload[:-1].clone()
    )
    with pytest.raises(V11IntegrityError, match="shape or memory domain"):
        verify_transport_streaming_v11(missing, maximum_chunk_bytes=config.maximum_chunk_bytes)

    first, second = captured.state.manifest.pages[:2]
    assert first.payload_bytes == second.payload_bytes
    duplicated_payload = captured.state.payload.clone()
    duplicated_payload[
        second.payload_offset_bytes : second.payload_offset_bytes + second.payload_bytes
    ].copy_(
        duplicated_payload[
            first.payload_offset_bytes : first.payload_offset_bytes + first.payload_bytes
        ]
    )
    with pytest.raises(V11IntegrityError, match="page digest mismatch"):
        verify_transport_streaming_v11(
            CanonicalKvTransportState(captured.state.manifest, duplicated_payload),
            maximum_chunk_bytes=config.maximum_chunk_bytes,
        )

    reordered_payload = captured.state.payload.clone()
    first_bytes = reordered_payload[
        first.payload_offset_bytes : first.payload_offset_bytes + first.payload_bytes
    ].clone()
    second_bytes = reordered_payload[
        second.payload_offset_bytes : second.payload_offset_bytes + second.payload_bytes
    ].clone()
    reordered_payload[
        first.payload_offset_bytes : first.payload_offset_bytes + first.payload_bytes
    ].copy_(second_bytes)
    reordered_payload[
        second.payload_offset_bytes : second.payload_offset_bytes + second.payload_bytes
    ].copy_(first_bytes)
    with pytest.raises(V11IntegrityError, match="page digest mismatch"):
        verify_transport_streaming_v11(
            CanonicalKvTransportState(captured.state.manifest, reordered_payload),
            maximum_chunk_bytes=config.maximum_chunk_bytes,
        )


def test_wrong_identity_and_incomplete_destination_fail_closed_before_restore() -> None:
    geometry, tensors, page_order, branches, destinations = _fixture(113, layer_count=1)
    config = V11PipelineConfig(
        seed=113,
        expected_device_type="cpu",
        maximum_chunk_bytes=geometry.physical_page_bytes * 2,
    )
    captured = capture_native_to_transport_v11(
        tensors,
        geometry,
        page_order=page_order,
        branch_tables=branches,
        identity=_identity(),
        config=config,
    )
    destination = tuple(torch.full_like(tensor, -5) for tensor in tensors)
    wrong = _identity() | {"policy_epoch": "wrong"}
    with pytest.raises(ValueError, match="identity differs"):
        restore_transport_to_native_v11(
            captured.state,
            destination,
            geometry,
            destination_block_indices=destinations,
            expected_identity=wrong,
            config=config,
        )
    incomplete = dict(destinations)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="cover every logical page"):
        restore_transport_to_native_v11(
            captured.state,
            destination,
            geometry,
            destination_block_indices=incomplete,
            expected_identity=_identity(),
            config=config,
        )


def test_destination_corruption_is_detected_before_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sloforge.continuum.adapters.vllm_reclamation_v11 as module

    geometry, tensors, page_order, branches, destinations = _fixture(41, layer_count=1)
    config = V11PipelineConfig(
        seed=41,
        expected_device_type="cpu",
        maximum_chunk_bytes=geometry.physical_page_bytes * 3,
    )
    captured = capture_native_to_transport_v11(
        tensors,
        geometry,
        page_order=page_order,
        branch_tables=branches,
        identity=_identity(),
        config=config,
    )
    destination = tuple(torch.full_like(tensor, -3) for tensor in tensors)
    original_runs = module._consecutive_runs
    injected = False

    def corrupt_then_run(values: tuple[int, ...]):
        nonlocal injected
        if not injected:
            injected = True
            pool = destination[0].movedim(geometry.block_dimensions[0], 0)
            page = pool[values[0]]
            raw = page.contiguous().view(torch.int16).clone()
            raw.reshape(-1)[0] ^= 1
            page.copy_(raw.view(torch.bfloat16).view(page.shape))
        return original_runs(values)

    monkeypatch.setattr(module, "_consecutive_runs", corrupt_then_run)
    with pytest.raises(V11IntegrityError, match="exact comparison failed"):
        restore_transport_to_native_v11(
            captured.state,
            destination,
            geometry,
            destination_block_indices=destinations,
            expected_identity=_identity(),
            config=config,
        )
    for tensor, block_dim in zip(destination, geometry.block_dimensions, strict=True):
        restored_pages = tensor.movedim(block_dim, 0).index_select(
            0, torch.tensor(tuple(destinations.values()))
        )
        assert torch.count_nonzero(restored_pages.view(torch.uint8)).item() == 0


def test_verified_token_rejects_in_place_payload_mutation() -> None:
    geometry, tensors, page_order, branches, destinations = _fixture(73, layer_count=2)
    config = V11PipelineConfig(
        seed=73,
        expected_device_type="cpu",
        maximum_chunk_bytes=geometry.physical_page_bytes * 2,
    )
    captured = capture_native_to_transport_v11(
        tensors,
        geometry,
        page_order=page_order,
        branch_tables=branches,
        identity=_identity(),
        config=config,
    )
    verified = verify_transport_streaming_v11(
        captured.state, maximum_chunk_bytes=config.maximum_chunk_bytes
    )
    captured.state.payload[0] ^= 1
    destination = tuple(torch.full_like(tensor, -9) for tensor in tensors)
    with pytest.raises(V11IntegrityError, match="does not bind"):
        write_and_validate_native_subset_v11(
            captured.state,
            destination,
            geometry,
            destination_block_indices=destinations,
            expected_identity=_identity(),
            verified_transport=verified,
            config=config,
        )
    assert all(torch.count_nonzero(tensor + 9).item() == 0 for tensor in destination)


def test_consumed_chunk_rejects_numpy_alias_mutation_after_verification() -> None:
    geometry, tensors, page_order, branches, destinations = _fixture(113, layer_count=1)
    config = V11PipelineConfig(
        seed=113,
        expected_device_type="cpu",
        maximum_chunk_bytes=geometry.physical_page_bytes * 2,
    )
    captured = capture_native_to_transport_v11(
        tensors,
        geometry,
        page_order=page_order,
        branch_tables=branches,
        identity=_identity(),
        config=config,
    )
    verified = verify_transport_streaming_v11(
        captured.state, maximum_chunk_bytes=config.maximum_chunk_bytes
    )
    version_before = captured.state.payload._version
    alias = captured.state.payload.numpy()
    alias[0] ^= 1
    assert captured.state.payload._version == version_before
    destination = tuple(torch.full_like(tensor, -19) for tensor in tensors)
    with pytest.raises(V11IntegrityError, match="consumed canonical page digest mismatch"):
        write_and_validate_native_subset_v11(
            captured.state,
            destination,
            geometry,
            destination_block_indices=destinations,
            expected_identity=_identity(),
            verified_transport=verified,
            config=config,
        )
    for tensor, block_dim in zip(destination, geometry.block_dimensions, strict=True):
        pool = tensor.movedim(block_dim, 0)
        selected = pool.index_select(0, torch.tensor(tuple(destinations.values())))
        assert torch.count_nonzero(selected).item() == 0
        untouched = set(range(geometry.num_blocks)) - set(destinations.values())
        assert all(torch.count_nonzero(pool[index] + 19).item() == 0 for index in untouched)


@pytest.mark.parametrize("tamper", ["missing", "ambiguous", "wrong-pass", "zero-bound"])
def test_verified_transport_rejects_tampered_preflight_page_provenance(tamper: str) -> None:
    captured, _branches, source_geometry = _small_capture()
    config = V11PipelineConfig(seed=41, expected_device_type="cpu", maximum_chunk_bytes=128)
    verified = verify_transport_streaming_v11(captured.state, maximum_chunk_bytes=128)
    provenance = verified.preflight_pass_id_by_page
    if tamper == "missing":
        corrupted = replace(verified, preflight_pass_id_by_page=provenance[:-1])
    elif tamper == "ambiguous":
        corrupted = replace(verified, preflight_pass_id_by_page=provenance + provenance[:1])
    elif tamper == "wrong-pass":
        corrupted = replace(
            verified,
            preflight_pass_id_by_page=(
                (provenance[0][0], "restore:preflight:forged"),
                *provenance[1:],
            ),
        )
    else:
        corrupted = replace(verified, maximum_chunk_bytes=0)
    restore_geometry = source_geometry.model_copy(update={"num_blocks": 32})
    destination = (
        torch.full((32, 2, 2, 2, 2), -47, dtype=torch.bfloat16),
        torch.full((2, 32, 2, 2, 2), -47, dtype=torch.bfloat16),
    )
    with pytest.raises(V11IntegrityError, match="prevalidated transport token"):
        write_and_validate_native_subset_v11(
            captured.state,
            destination,
            restore_geometry,
            destination_block_indices={"page.left": 10},
            expected_identity=_identity(),
            verified_transport=corrupted,
            config=config,
        )
    assert all(torch.count_nonzero(tensor + 47).item() == 0 for tensor in destination)


def test_subset_starting_inside_preflight_chunk_depends_on_actual_covering_pass() -> None:
    captured, _branches, source_geometry = _small_capture()
    config = V11PipelineConfig(seed=41, expected_device_type="cpu", maximum_chunk_bytes=128)
    branch_group = captured.state_passes[0].branch_group
    recorder = V11StatePassRecorder(device="cpu", branch_group=branch_group)
    verified = verify_transport_streaming_v11(
        captured.state,
        maximum_chunk_bytes=128,
        pass_recorder=recorder,
    )
    restore_geometry = source_geometry.model_copy(update={"num_blocks": 32})
    destination = (
        torch.full((32, 2, 2, 2, 2), -53, dtype=torch.bfloat16),
        torch.full((2, 32, 2, 2, 2), -53, dtype=torch.bfloat16),
    )
    evidence, _stats = write_and_validate_native_subset_v11(
        captured.state,
        destination,
        restore_geometry,
        destination_block_indices={"page.left": 10},
        expected_identity=_identity(),
        verified_transport=verified,
        config=config,
        pass_recorder=recorder,
    )
    records = recorder.resolve()
    validate_authoritative_pass_records(
        records,
        expected_device="cpu",
        expected_branch_group=branch_group,
    )
    snapshot = next(record for record in records if record.operation == "owned_snapshot")
    assert snapshot.chunk_id.endswith("offset-000000000064")
    assert snapshot.synchronization_dependency == (
        "restore:preflight:chunk-0000-offset-000000000000:sha256",
    )
    assert evidence.passed


def test_destination_validation_compares_bfloat16_raw_bits() -> None:
    geometry, tensors, page_order, branches, destinations = _fixture(41, layer_count=1)
    block_dim = geometry.block_dimensions[0] % tensors[0].ndim
    source_page = tensors[0].movedim(block_dim, 0)[page_order[0][1]]
    bit_pattern = torch.tensor([32705, -32768, 0, 16256], dtype=torch.int16).repeat(
        (source_page.numel() + 3) // 4
    )[: source_page.numel()]
    source_page.copy_(bit_pattern.view(torch.bfloat16).view(source_page.shape))
    config = V11PipelineConfig(
        seed=41,
        expected_device_type="cpu",
        maximum_chunk_bytes=geometry.physical_page_bytes * 2,
    )
    captured = capture_native_to_transport_v11(
        tensors,
        geometry,
        page_order=page_order,
        branch_tables=branches,
        identity=_identity(),
        config=config,
    )
    destination = tuple(torch.full_like(tensor, -1) for tensor in tensors)
    restored = restore_transport_to_native_v11(
        captured.state,
        destination,
        geometry,
        destination_block_indices=destinations,
        expected_identity=_identity(),
        config=config,
    )
    assert restored.validation.destination_exact_match
    restored_page = destination[0].movedim(block_dim, 0)[destinations[page_order[0][0]]]
    assert torch.equal(
        restored_page.contiguous().view(torch.uint8),
        source_page.contiguous().view(torch.uint8),
    )


def test_all_block_dimensions_are_normalized_before_any_write() -> None:
    geometry, tensors, page_order, branches, destinations = _fixture(73, layer_count=2)
    normalized = tuple(value % 5 for value in geometry.block_dimensions)
    equivalent = geometry.model_copy(
        update={"block_dimensions": (normalized[0], normalized[1] + 5)}
    )
    config = V11PipelineConfig(
        seed=73,
        expected_device_type="cpu",
        maximum_chunk_bytes=geometry.physical_page_bytes * 2,
    )
    captured = capture_native_to_transport_v11(
        tensors,
        equivalent,
        page_order=page_order,
        branch_tables=branches,
        identity=_identity(),
        config=config,
    )
    destination = tuple(torch.full_like(tensor, -23) for tensor in tensors)
    restored = restore_transport_to_native_v11(
        captured.state,
        destination,
        equivalent,
        destination_block_indices=destinations,
        expected_identity=_identity(),
        config=config,
    )
    assert restored.validation.passed


class _Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.is_null = False
        self.ref_cnt = 0
        self.block_hash = None


class _BlockPool:
    def __init__(self, blocks: list[_Block]) -> None:
        self.blocks = blocks


class _Blocks:
    def __init__(self, blocks: list[_Block]) -> None:
        self.blocks = (blocks,)


class _Queue:
    def __init__(self, requests: list[object]) -> None:
        self.requests = list(requests)

    def remove_requests(self, requests: list[object]) -> None:
        for request in requests:
            if request in self.requests:
                self.requests.remove(request)

    def add_request(self, request: object) -> None:
        self.requests.append(request)


class _Request:
    def __init__(self, request_id: str, token_ids: tuple[int, ...]) -> None:
        self.request_id = request_id
        self.all_token_ids = token_ids
        self.num_computed_tokens = 0


class _Manager:
    def __init__(self) -> None:
        self.tables: dict[str, list[_Block]] = {}
        self.zero_queue: list[int] = []
        self.next_block = 10
        self.cached_root: _Block | None = None
        self.reset = False
        self.block_pool = _BlockPool([_Block(index) for index in range(64)])
        self.epoch_source = install_vllm_allocator_epoch_source(self)

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
        required = (num_new_computed_tokens + num_external_computed_tokens + 1) // 2
        new: list[_Block] = []
        while len(prefix) + len(new) < required:
            block = self.block_pool.blocks[self.next_block]
            self.next_block += 1
            block.ref_cnt += 1
            new.append(block)
            self.zero_queue.append(block.block_id)
        for block in prefix:
            block.ref_cnt += 1
        self.tables[request.request_id] = prefix + new
        self.epoch_source.issue_from_runtime_allocation(
            request,
            previous_block_ids=(),
            cached_block_ids=tuple(block.block_id for block in prefix),
            allocator_returned_block_ids=tuple(block.block_id for block in new),
        )
        return _Blocks(new)

    def get_blocks(self, request_id: str) -> _Blocks:
        return _Blocks(self.tables[request_id])

    def get_computed_blocks(self, _request: _Request) -> tuple[_Blocks, int]:
        assert self.cached_root is not None
        return _Blocks([self.cached_root]), 2

    def cache_blocks(self, request: _Request, _computed_tokens: int) -> None:
        if self.cached_root is None:
            self.cached_root = self.tables[request.request_id][0]
            self.cached_root.block_hash = "cached-root"

    def free(self, request: _Request) -> None:
        released = self.tables.pop(request.request_id, [])
        for block in released:
            block.ref_cnt -= 1
        self.epoch_source.observe_runtime_free(
            request,
            released_block_ids=tuple(block.block_id for block in released),
        )

    def reset_prefix_cache(self) -> bool:
        self.cached_root = None
        for block in self.block_pool.blocks:
            block.block_hash = None
        self.reset = True
        self.epoch_source.observe_prefix_cache_reset()
        return True


class _Scheduler:
    def __init__(self, requests: list[_Request], manager: _Manager) -> None:
        self.requests = {request.request_id: request for request in requests}
        self.waiting = _Queue(list(requests))
        self.skipped_waiting = _Queue([])
        self.running: list[_Request] = []
        self.scheduler_config = type("Config", (), {"async_scheduling": False})()
        self.kv_cache_manager = manager
        self.needs_kv_cache_zeroing = False

    def _enqueue_waiting_request(self, request: _Request) -> None:
        self.waiting.add_request(request)

    def add_request(self, _request: _Request) -> None:
        return None

    def schedule(self) -> object:
        return object()

    def update_from_output(self, _output: object) -> None:
        return None


class _EngineCore:
    def __init__(self, scheduler: _Scheduler) -> None:
        self.scheduler = scheduler


def _inproc_client_type() -> type:
    return type(
        "InprocClient",
        (),
        {"__module__": "vllm.v1.engine.core_client"},
    )


class _LlmEngine:
    def __init__(self, client: object) -> None:
        self.engine_core = client

    def add_request(self, _request: object) -> None:
        return None

    def step(self) -> list[object]:
        return []


def _engine_step_binding(
    scheduler: _Scheduler,
    manager: _Manager,
) -> Vllm0230EngineStepBinding:
    core = _EngineCore(scheduler)
    client = _inproc_client_type()()
    client.engine_core = core
    engine = _LlmEngine(client)
    view = SimpleNamespace(
        runtime_version="0.23.0",
        llm_engine=engine,
        engine_core_client=client,
        engine_core=core,
        scheduler=scheduler,
        manager=manager,
        vllm_config=SimpleNamespace(scheduler_config=scheduler.scheduler_config),
    )
    return Vllm0230EngineStepBinding(view, acquire_timeout_seconds=1.0)


@contextmanager
def _production_import_gate(
    scheduler: _Scheduler,
    manager: _Manager,
    *,
    gate_id: str,
) -> Iterator[Any]:
    binding = _engine_step_binding(scheduler, manager)
    try:
        with binding.critical_section("IMPORT_ADMISSION", gate_id=gate_id) as gate:
            yield gate
    finally:
        binding.close()


def _small_capture():
    first = torch.arange(6 * 2 * 2 * 2 * 2, dtype=torch.bfloat16).view(6, 2, 2, 2, 2)
    second = (1000 + torch.arange(6 * 2 * 2 * 2 * 2)).to(torch.bfloat16).view(2, 6, 2, 2, 2)
    geometry = NativeKvGeometry(
        layer_names=("layer.0", "layer.1"),
        num_blocks=6,
        block_size_tokens=2,
        kv_heads=2,
        head_size=2,
        element_size_bytes=2,
        block_dimensions=(0, 1),
        source_axis_labels=("kv", "token", "head", "dim"),
    )
    left = (10, 11, 12, 13)
    right = (10, 11, 20)
    branches = (
        CanonicalBranchTable(
            logical_branch_id="branch.0",
            parent_logical_branch_id="root",
            token_ids=left,
            token_history_sha256=token_history_sha256(left),
            computed_tokens=4,
            logical_page_ids=("page.root", "page.left"),
        ),
        CanonicalBranchTable(
            logical_branch_id="branch.1",
            parent_logical_branch_id="root",
            token_ids=right,
            token_history_sha256=token_history_sha256(right),
            computed_tokens=3,
            logical_page_ids=("page.root", "page.right"),
        ),
    )
    config = V11PipelineConfig(seed=41, expected_device_type="cpu", maximum_chunk_bytes=128)
    captured = capture_native_to_transport_v11(
        (first, second),
        geometry,
        page_order=(
            ("page.root", 1, 2, ("branch.0", "branch.1")),
            ("page.left", 2, 2, ("branch.0",)),
            ("page.right", 3, 1, ("branch.1",)),
        ),
        branch_tables=branches,
        identity=_identity(),
        config=config,
    )
    return captured, branches, geometry


def test_streaming_stager_requires_typed_fused_evidence_and_cleans_up() -> None:
    captured, branches, source_geometry = _small_capture()
    requests = [
        _Request("branch.0@v11", branches[0].token_ids),
        _Request("branch.1@v11", branches[1].token_ids),
    ]
    manager = _Manager()
    scheduler = _Scheduler(requests, manager)
    verified = verify_transport_streaming_v11(captured.state, maximum_chunk_bytes=128)
    calls: list[set[str]] = []
    proofs: list[V11SubsetValidationEvidence] = []
    restore_geometry = source_geometry.model_copy(update={"num_blocks": 32})
    destination = (
        torch.full((32, 2, 2, 2, 2), -17, dtype=torch.bfloat16),
        torch.full((2, 32, 2, 2, 2), -17, dtype=torch.bfloat16),
    )
    config = V11PipelineConfig(seed=41, expected_device_type="cpu", maximum_chunk_bytes=128)
    destination_target_sha256 = destination_target_identity_v11(
        destination, restore_geometry, config
    )

    def fused(
        mapping: dict[str, int], epoch_proof: VllmAllocatorEpochProof
    ) -> V11SubsetValidationEvidence:
        calls.append(set(mapping))
        evidence, _stats = write_and_validate_native_subset_v11(
            captured.state,
            destination,
            restore_geometry,
            destination_block_indices=mapping,
            expected_identity=_identity(),
            verified_transport=verified,
            config=config,
            allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
        )
        proofs.append(evidence)
        return evidence

    def scrub(mapping: dict[str, int], epoch_proof: VllmAllocatorEpochProof):
        return scrub_native_pages_v11(
            destination,
            restore_geometry,
            state=captured.state,
            verified_transport=verified,
            config=config,
            allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
            destination_block_indices=mapping,
        )

    with _production_import_gate(
        scheduler,
        manager,
        gate_id="fixture-engine-step-gate-41",
    ) as admission_gate:
        result = Vllm0230StreamingRestoreStager(scheduler, manager).import_group_v11(
            captured.state,
            verified_transport=verified,
            runtime_request_ids={"branch.0": "branch.0@v11", "branch.1": "branch.1@v11"},
            expected_identity=_identity(),
            destination_target_sha256=destination_target_sha256,
            admission_gate=admission_gate,
            write_and_validate_pages=fused,
            scrub_pages_on_failure=scrub,
        )
    assert calls == [{"page.root", "page.left"}, {"page.right"}]
    assert set(result.logical_page_destinations) == {"page.root", "page.left", "page.right"}
    assert [request.num_computed_tokens for request in requests] == [4, 3]

    failed_requests = [
        _Request("branch.0@failed", branches[0].token_ids),
        _Request("branch.1@failed", branches[1].token_ids),
    ]
    failed_manager = _Manager()
    failed_scheduler = _Scheduler(failed_requests, failed_manager)
    with (
        _production_import_gate(
            failed_scheduler,
            failed_manager,
            gate_id="fixture-engine-step-gate-failed-41",
        ) as failed_gate,
        pytest.raises(V11IntegrityError, match="evidence failed"),
    ):
        Vllm0230StreamingRestoreStager(failed_scheduler, failed_manager).import_group_v11(
            captured.state,
            verified_transport=verified,
            runtime_request_ids={
                "branch.0": "branch.0@failed",
                "branch.1": "branch.1@failed",
            },
            expected_identity=_identity(),
            destination_target_sha256=destination_target_sha256,
            admission_gate=failed_gate,
            write_and_validate_pages=lambda mapping, proof: replace(
                fused(mapping, proof),
                full_byte_coverage=False,
                passed=False,
            ),
            scrub_pages_on_failure=scrub,
        )
    assert failed_scheduler.requests == {}
    assert failed_manager.tables == {}
    assert failed_manager.reset

    stale_requests = [
        _Request("branch.0@stale", branches[0].token_ids),
        _Request("branch.1@stale", branches[1].token_ids),
    ]
    stale_manager = _Manager()
    stale_manager.next_block = 20
    stale_scheduler = _Scheduler(stale_requests, stale_manager)

    def stale_scrub(mapping: dict[str, int], epoch_proof: VllmAllocatorEpochProof):
        return scrub_native_pages_v11(
            destination,
            restore_geometry,
            state=captured.state,
            verified_transport=verified,
            config=config,
            allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
            destination_block_indices=mapping,
        )

    with (
        _production_import_gate(
            stale_scheduler,
            stale_manager,
            gate_id="fixture-engine-step-gate-stale-42",
        ) as stale_gate,
        pytest.raises(V11IntegrityError, match="evidence failed"),
    ):
        Vllm0230StreamingRestoreStager(stale_scheduler, stale_manager).import_group_v11(
            captured.state,
            verified_transport=verified,
            runtime_request_ids={
                "branch.0": "branch.0@stale",
                "branch.1": "branch.1@stale",
            },
            expected_identity=_identity(),
            destination_target_sha256=destination_target_sha256,
            admission_gate=stale_gate,
            write_and_validate_pages=lambda _mapping, _proof: proofs[0],
            scrub_pages_on_failure=stale_scrub,
        )
    assert stale_scheduler.requests == {}
    assert stale_manager.tables == {}


@pytest.mark.parametrize(("failing_subset", "expected_scrub_pages"), [(1, 2), (2, 3)])
def test_streaming_stager_owns_exactly_one_scrub_after_subset_write_failure(
    failing_subset: int,
    expected_scrub_pages: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, branches, source_geometry = _small_capture()
    requests = [
        _Request("branch.0@write-failure", branches[0].token_ids),
        _Request("branch.1@write-failure", branches[1].token_ids),
    ]
    manager = _Manager()
    scheduler = _Scheduler(requests, manager)
    restore_geometry = source_geometry.model_copy(update={"num_blocks": 32})
    destination = (
        torch.full((32, 2, 2, 2, 2), -43, dtype=torch.bfloat16),
        torch.full((2, 32, 2, 2, 2), -43, dtype=torch.bfloat16),
    )
    config = V11PipelineConfig(seed=41, expected_device_type="cpu", maximum_chunk_bytes=128)
    branch_group = captured.state_passes[0].branch_group
    recorder = V11StatePassRecorder(device="cpu", branch_group=branch_group)
    verified = verify_transport_streaming_v11(
        captured.state,
        maximum_chunk_bytes=128,
        pass_recorder=recorder,
    )
    target = destination_target_identity_v11(destination, restore_geometry, config)
    original_write_chunk = v11_adapter._write_validate_device_chunk
    active_subset = 0
    scrub_evidence: list[Any] = []

    def fail_selected_subset(*args: Any, **kwargs: Any) -> tuple[int, int]:
        if active_subset == failing_subset:
            raise RuntimeError(f"injected subset-{failing_subset} write failure")
        return original_write_chunk(*args, **kwargs)

    monkeypatch.setattr(v11_adapter, "_write_validate_device_chunk", fail_selected_subset)

    def write(
        mapping: dict[str, int], epoch_proof: VllmAllocatorEpochProof
    ) -> V11SubsetValidationEvidence:
        nonlocal active_subset
        active_subset += 1
        return write_and_validate_native_subset_v11(
            captured.state,
            destination,
            restore_geometry,
            destination_block_indices=mapping,
            expected_identity=_identity(),
            verified_transport=verified,
            config=config,
            allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
            pass_recorder=recorder,
            caller_guarantees_outer_scrub=True,
        )[0]

    def scrub(mapping: dict[str, int], epoch_proof: VllmAllocatorEpochProof) -> Any:
        evidence = scrub_native_pages_v11(
            destination,
            restore_geometry,
            state=captured.state,
            verified_transport=verified,
            config=config,
            allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
            destination_block_indices=mapping,
            pass_recorder=recorder,
        )
        scrub_evidence.append(evidence)
        return evidence

    with (
        _production_import_gate(
            scheduler,
            manager,
            gate_id=f"fixture-subset-{failing_subset}-write-failure",
        ) as gate,
        pytest.raises(RuntimeError, match=f"injected subset-{failing_subset}"),
    ):
        Vllm0230StreamingRestoreStager(scheduler, manager).import_group_v11(
            captured.state,
            verified_transport=verified,
            runtime_request_ids={
                "branch.0": "branch.0@write-failure",
                "branch.1": "branch.1@write-failure",
            },
            expected_identity=_identity(),
            destination_target_sha256=target,
            admission_gate=gate,
            write_and_validate_pages=write,
            scrub_pages_on_failure=scrub,
        )

    records = recorder.resolve()
    validate_authoritative_pass_records(
        records,
        expected_device="cpu",
        expected_branch_group=branch_group,
    )
    scrub_records = tuple(record for record in records if record.chunk_id.startswith("scrub-"))
    zero_records = tuple(
        record for record in scrub_records if record.operation == "native_page_zero"
    )
    assert len(scrub_evidence) == 1
    assert len(scrub_evidence[0].logical_page_ids) == expected_scrub_pages
    assert len(zero_records) == len(restore_geometry.layer_names)
    assert sum(record.physical_bytes_written for record in zero_records) == (
        scrub_evidence[0].physical_bytes_zeroed
    )
    assert len({record.pass_id for record in records}) == len(records)
    assert scheduler.requests == {}
    assert scheduler.waiting.requests == []
    assert scheduler.skipped_waiting.requests == []
    assert scheduler.running == []
    assert manager.tables == {}
    assert manager.reset
    assert all(request.num_computed_tokens == 0 for request in requests)


@pytest.mark.parametrize("failure_mode", ["scrub", "free", "reset", "final-enqueue"])
def test_stager_cleanup_failure_requires_runtime_teardown(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, branches, source_geometry = _small_capture()
    requests = [
        _Request("branch.0@teardown", branches[0].token_ids),
        _Request("branch.1@teardown", branches[1].token_ids),
    ]
    manager = _Manager()
    scheduler = _Scheduler(requests, manager)
    verified = verify_transport_streaming_v11(captured.state, maximum_chunk_bytes=128)
    restore_geometry = source_geometry.model_copy(update={"num_blocks": 32})
    destination = (
        torch.full((32, 2, 2, 2, 2), -29, dtype=torch.bfloat16),
        torch.full((2, 32, 2, 2, 2), -29, dtype=torch.bfloat16),
    )
    config = V11PipelineConfig(seed=41, expected_device_type="cpu", maximum_chunk_bytes=128)
    target = destination_target_identity_v11(destination, restore_geometry, config)

    def write(
        mapping: dict[str, int], epoch_proof: VllmAllocatorEpochProof
    ) -> V11SubsetValidationEvidence:
        evidence, _stats = write_and_validate_native_subset_v11(
            captured.state,
            destination,
            restore_geometry,
            destination_block_indices=mapping,
            expected_identity=_identity(),
            verified_transport=verified,
            config=config,
            allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
        )
        if failure_mode == "final-enqueue":
            return evidence
        return replace(evidence, passed=False)

    def scrub(mapping: dict[str, int], epoch_proof: VllmAllocatorEpochProof):
        if failure_mode == "scrub":
            raise RuntimeError("injected scrub failure")
        return scrub_native_pages_v11(
            destination,
            restore_geometry,
            state=captured.state,
            verified_transport=verified,
            config=config,
            allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
            destination_block_indices=mapping,
        )

    if failure_mode == "free":
        monkeypatch.setattr(
            manager,
            "free",
            lambda _request: (_ for _ in ()).throw(RuntimeError("injected free failure")),
        )
    if failure_mode == "reset":
        monkeypatch.setattr(manager, "reset_prefix_cache", lambda: False)
    if failure_mode == "final-enqueue":
        monkeypatch.setattr(
            scheduler.waiting,
            "add_request",
            lambda _request: (_ for _ in ()).throw(RuntimeError("injected enqueue failure")),
        )

    def invoke() -> None:
        with _production_import_gate(
            scheduler,
            manager,
            gate_id=f"gate-{failure_mode}",
        ) as gate:
            Vllm0230StreamingRestoreStager(scheduler, manager).import_group_v11(
                captured.state,
                verified_transport=verified,
                runtime_request_ids={
                    "branch.0": "branch.0@teardown",
                    "branch.1": "branch.1@teardown",
                },
                expected_identity=_identity(),
                destination_target_sha256=target,
                admission_gate=gate,
                write_and_validate_pages=write,
                scrub_pages_on_failure=scrub,
            )

    if failure_mode == "final-enqueue":
        with pytest.raises(RuntimeError, match="injected enqueue failure"):
            invoke()
        assert scheduler.requests == {}
        assert manager.tables == {}
        return
    with pytest.raises(V11RuntimeTeardownRequired) as raised:
        invoke()
    assert raised.value.affected_block_ids
    assert scheduler.requests
    if failure_mode in {"scrub", "free"}:
        assert manager.tables


def test_admission_gate_is_rechecked_between_every_group_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, branches, source_geometry = _small_capture()
    requests = [
        _Request("branch.0@gate", branches[0].token_ids),
        _Request("branch.1@gate", branches[1].token_ids),
    ]
    manager = _Manager()
    scheduler = _Scheduler(requests, manager)
    verified = verify_transport_streaming_v11(captured.state, maximum_chunk_bytes=128)
    geometry = source_geometry.model_copy(update={"num_blocks": 32})
    destination = (
        torch.full((32, 2, 2, 2, 2), -31, dtype=torch.bfloat16),
        torch.full((2, 32, 2, 2, 2), -31, dtype=torch.bfloat16),
    )
    config = V11PipelineConfig(seed=41, expected_device_type="cpu", maximum_chunk_bytes=128)

    def write(
        mapping: dict[str, int], epoch_proof: VllmAllocatorEpochProof
    ) -> V11SubsetValidationEvidence:
        return write_and_validate_native_subset_v11(
            captured.state,
            destination,
            geometry,
            destination_block_indices=mapping,
            expected_identity=_identity(),
            verified_transport=verified,
            config=config,
            allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
        )[0]

    def scrub(mapping: dict[str, int], epoch_proof: VllmAllocatorEpochProof):
        return scrub_native_pages_v11(
            destination,
            geometry,
            state=captured.state,
            verified_transport=verified,
            config=config,
            allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
            destination_block_indices=mapping,
        )

    original_add = scheduler.waiting.add_request
    admitted = 0

    def release_gate_after_first_add(request: object) -> None:
        nonlocal admitted
        original_add(request)
        admitted += 1
        # Displacing any installed production mutation wrapper invalidates the
        # witness. The stager must detect that before admitting the next branch.
        scheduler.schedule = lambda: object()  # type: ignore[method-assign]

    monkeypatch.setattr(scheduler.waiting, "add_request", release_gate_after_first_add)
    binding = _engine_step_binding(scheduler, manager)
    installed_schedule = scheduler.schedule
    try:
        with binding.critical_section(
            "IMPORT_ADMISSION",
            gate_id="gate-loss-fixture",
        ) as gate:
            with pytest.raises(V11RuntimeTeardownRequired, match="admission gate"):
                Vllm0230StreamingRestoreStager(scheduler, manager).import_group_v11(
                    captured.state,
                    verified_transport=verified,
                    runtime_request_ids={
                        "branch.0": "branch.0@gate",
                        "branch.1": "branch.1@gate",
                    },
                    expected_identity=_identity(),
                    destination_target_sha256=destination_target_identity_v11(
                        destination, geometry, config
                    ),
                    admission_gate=gate,
                    write_and_validate_pages=write,
                    scrub_pages_on_failure=scrub,
                )
            scheduler.schedule = installed_schedule  # type: ignore[method-assign]
    finally:
        binding.close()
    assert admitted == 1
    assert scheduler.requests == {}
    assert manager.tables == {}


def test_pipeline_config_never_silently_switches_memory_mode() -> None:
    with pytest.raises(ValueError, match="cannot silently"):
        V11PipelineConfig(seed=41, expected_device_type="cpu", pin_memory=True)
    with pytest.raises(ValueError, match="explicitly pinned"):
        V11PipelineConfig(
            seed=41,
            expected_device_type="cuda",
            expected_cuda_device_index=0,
            expected_gpu_uuid="GPU-fixture",
            pin_memory=False,
        )
    with pytest.raises(ValueError, match="exact nonnegative device index"):
        V11PipelineConfig(
            seed=41,
            expected_device_type="cuda",
            pin_memory=True,
            asynchronous_copy=True,
        )
    with pytest.raises(ValueError, match="explicit asynchronous"):
        V11PipelineConfig(
            seed=41,
            expected_device_type="cuda",
            expected_cuda_device_index=0,
            expected_gpu_uuid="GPU-fixture",
            pin_memory=True,
        )
    with pytest.raises(ValueError, match="at least two"):
        V11PipelineConfig(
            seed=41,
            expected_device_type="cuda",
            expected_cuda_device_index=0,
            expected_gpu_uuid="GPU-fixture",
            pin_memory=True,
            asynchronous_copy=True,
            buffer_count=1,
        )
