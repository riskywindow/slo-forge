from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from pydantic import ValidationError

from sloforge.continuum.adapters import ReferenceHeadMajorAdapter, ReferenceTokenMajorAdapter
from sloforge.continuum.conversion import (
    ConversionBackend,
    ConversionCompilationError,
    KVLayout,
    KVLayoutKind,
    KVShard,
    PhysicalKVState,
    StateIntegrityError,
    TransformationDAG,
    canonical_convert,
    compile_conversion,
    decode_logical,
    direct_convert,
    direct_convert_capture,
    make_random_state,
    measure_and_select_converter,
    quality_bounded_convert,
    stream_direct_conversion,
    verify_direct_against_canonical,
)


def _layouts(
    *,
    layers: int = 2,
    tokens: int = 11,
    heads: int = 8,
    dim: int = 4,
    source_page: int = 3,
    destination_page: int = 5,
    destination_dtype: str = "float32",
) -> tuple[KVLayout, KVLayout]:
    return (
        KVLayout(
            kind=KVLayoutKind.TOKEN_MAJOR_SEPARATE,
            tensor_parallel_degree=4,
            page_size_tokens=source_page,
            layer_count=layers,
            token_count=tokens,
            kv_head_count=heads,
            head_dim=dim,
            dtype="float32",
        ),
        KVLayout(
            kind=KVLayoutKind.HEAD_MAJOR_PACKED,
            tensor_parallel_degree=2,
            page_size_tokens=destination_page,
            layer_count=layers,
            token_count=tokens,
            kv_head_count=heads,
            head_dim=dim,
            dtype=destination_dtype,
        ),
    )


def _one_token_bound(source: KVLayout, destination: KVLayout) -> int:
    return (
        source.layer_count
        * source.kv_head_count
        * source.head_dim
        * 2
        * max(np.dtype(source.dtype).itemsize, np.dtype(destination.dtype).itemsize)
    )


@pytest.mark.parametrize("seed", range(24))
def test_direct_matches_canonical_across_pages_resharding_and_partial_pages(
    seed: int,
) -> None:
    # Deterministic randomized coverage remains available in minimal installations
    # where the optional Hypothesis dependency is intentionally absent.
    rng = np.random.default_rng(seed)
    layers = int(rng.integers(1, 4))
    tokens = int(rng.integers(0, 20))
    heads = (4, 8, 12)[int(rng.integers(0, 3))]
    dim = int(rng.integers(1, 8))
    source_page = int(rng.integers(1, 8))
    destination_page = int(rng.integers(1, 8))
    chunk_multiple = int(rng.integers(1, 6))
    source_layout, destination_layout = _layouts(
        layers=layers,
        tokens=tokens,
        heads=heads,
        dim=dim,
        source_page=source_page,
        destination_page=destination_page,
    )
    source = make_random_state(source_layout, seed=seed)
    memory_bound = _one_token_bound(source_layout, destination_layout) * chunk_multiple

    canonical = canonical_convert(source, destination_layout)
    direct = direct_convert(
        source,
        destination_layout,
        maximum_temporary_bytes=memory_bound,
    )
    canonical_key, canonical_value = decode_logical(canonical)
    direct_key, direct_value = decode_logical(direct)

    assert np.array_equal(canonical_key, direct_key)
    assert np.array_equal(canonical_value, direct_value)
    assert canonical.content_hash == direct.content_hash
    evidence = verify_direct_against_canonical(
        source,
        destination_layout,
        maximum_temporary_bytes=memory_bound,
    )
    assert evidence.exact
    assert evidence.maximum_absolute_error == 0.0


def test_compiled_ir_is_typed_acyclic_streaming_and_memory_bounded() -> None:
    source, destination = _layouts(tokens=13)
    bound = _one_token_bound(source, destination) * 2
    program = compile_conversion(
        source,
        destination,
        maximum_temporary_bytes=bound,
        measured_throughput_bytes_s=50_000_000.0,
    )

    assert program.direct_conversion
    assert program.prediction_basis == "measured_throughput"
    assert program.predicted_duration_ns > 0
    assert program.dag.topological_order()[0] == "read-reshard"
    assert program.dag.topological_order()[-1] == "validate"
    assert {operation.code.value for operation in program.dag.operations} >= {
        "reshard",
        "pack",
        "page_remap",
        "checksum",
        "validate",
    }
    assert all(
        chunk.estimated_temporary_bytes <= program.memory_plan.maximum_temporary_bytes
        for chunk in program.chunk_schedule.chunks
    )
    assert program.memory_plan.canonical_fallback_temporary_bytes > bound


def test_compiler_rejects_unbounded_or_semantically_different_conversion() -> None:
    source, destination = _layouts()
    with pytest.raises(ConversionCompilationError, match="cannot hold one logical token"):
        compile_conversion(
            source,
            destination,
            maximum_temporary_bytes=_one_token_bound(source, destination) - 1,
        )

    incompatible = KVLayout(
        kind=destination.kind,
        tensor_parallel_degree=destination.tensor_parallel_degree,
        page_size_tokens=destination.page_size_tokens,
        layer_count=destination.layer_count,
        token_count=destination.token_count + 1,
        kv_head_count=destination.kv_head_count,
        head_dim=destination.head_dim,
        dtype=destination.dtype,
    )
    with pytest.raises(ConversionCompilationError, match="cannot change logical"):
        compile_conversion(source, incompatible, maximum_temporary_bytes=100_000)


def test_streaming_chunks_cover_state_and_never_exceed_bound() -> None:
    source_layout, destination_layout = _layouts(tokens=17)
    source = make_random_state(source_layout, seed=42)
    bound = _one_token_bound(source_layout, destination_layout) * 3
    chunks = list(
        stream_direct_conversion(
            source,
            destination_layout,
            maximum_temporary_bytes=bound,
        )
    )

    assert chunks
    assert all(chunk.temporary_nbytes <= bound for chunk in chunks)
    for rank in range(destination_layout.tensor_parallel_degree):
        rank_chunks = [chunk for chunk in chunks if chunk.destination_rank == rank]
        assert rank_chunks[0].token_start == 0
        assert rank_chunks[-1].token_end == destination_layout.token_count
        assert all(left.token_end == right.token_start for left, right in pairwise(rank_chunks))


def test_non_contiguous_source_shards_are_supported() -> None:
    source_layout, destination_layout = _layouts(tokens=7)
    contiguous = make_random_state(source_layout, seed=9)
    shards = tuple(
        KVShard(
            rank=shard.rank,
            head_start=shard.head_start,
            head_end=shard.head_end,
            key=shard.key[..., ::-1] if shard.key is not None else None,
            value=shard.value[..., ::-1] if shard.value is not None else None,
        )
        for shard in contiguous.shards
    )
    non_contiguous = PhysicalKVState(layout=source_layout, shards=shards)
    assert shards[0].key is not None and not shards[0].key.flags.c_contiguous

    evidence = verify_direct_against_canonical(
        non_contiguous,
        destination_layout,
        maximum_temporary_bytes=_one_token_bound(source_layout, destination_layout) * 2,
    )
    assert evidence.exact


def test_tampered_source_is_rejected_before_conversion() -> None:
    source_layout, destination_layout = _layouts()
    source = make_random_state(source_layout, seed=3)
    shard = source.shards[0]
    assert shard.key is not None
    shard.key[0, 0, 0, 0] += 1.0

    with pytest.raises(StateIntegrityError, match="checksum mismatch"):
        direct_convert(
            source,
            destination_layout,
            maximum_temporary_bytes=_one_token_bound(source_layout, destination_layout),
        )


def test_dtype_conversion_is_verified_against_canonical_reference() -> None:
    source_layout, destination_layout = _layouts(destination_dtype="float16")
    source = make_random_state(source_layout, seed=21)
    evidence = verify_direct_against_canonical(
        source,
        destination_layout,
        maximum_temporary_bytes=_one_token_bound(source_layout, destination_layout) * 2,
    )
    assert evidence.exact
    assert evidence.maximum_absolute_error == 0.0


def test_backend_selection_contains_raw_measured_provenance() -> None:
    source_layout, destination_layout = _layouts(tokens=8)
    source = make_random_state(source_layout, seed=17)
    selection = measure_and_select_converter(
        source,
        destination_layout,
        maximum_temporary_bytes=_one_token_bound(source_layout, destination_layout) * 2,
        repetitions=2,
        seed=99,
    )

    assert selection.selected_backend in {
        ConversionBackend.CANONICAL_CPU,
        ConversionBackend.DIRECT_CPU,
    }
    assert len(selection.measurements) == 4
    assert {measurement.backend for measurement in selection.measurements} == {
        ConversionBackend.CANONICAL_CPU,
        ConversionBackend.DIRECT_CPU,
    }
    assert {measurement.seed for measurement in selection.measurements} == {99}
    assert selection.verification.exact


def test_transformation_dag_rejects_cycle() -> None:
    source, destination = _layouts(tokens=3)
    program = compile_conversion(
        source,
        destination,
        maximum_temporary_bytes=_one_token_bound(source, destination),
    )
    operations = list(program.dag.operations)
    operations[0] = operations[0].model_copy(update={"depends_on": ("validate",)})
    with pytest.raises(ValidationError, match="cycle"):
        TransformationDAG(operations=tuple(operations))


def test_direct_converter_operates_on_live_reference_capture_bytes() -> None:
    source = ReferenceTokenMajorAdapter(page_size_tokens=3)
    destination = ReferenceHeadMajorAdapter(page_size_tokens=5)
    source.create_session(
        session_id="live-conversion",
        request_id="request-conversion",
        tenant_id="tenant-a",
        input_token_ids=(2, 3, 5),
        seed=71,
    )
    for event in source.stream_tokens("live-conversion", count=9):
        source.acknowledge_gateway(
            "live-conversion",
            token_index=event.token_index,
            owner_epoch=event.owner_epoch,
        )
    captured = source.capture_consistent("live-conversion")
    converted, evidence = direct_convert_capture(
        captured,
        destination=destination,
        maximum_temporary_bytes=512,
    )

    assert evidence.canonical_attention_match
    assert evidence.compared_attention_bytes > 0
    assert converted.layout == destination.config.layout
    assert converted.runtime == destination.identity
    destination.prepare_destination_session(
        converted,
        destination_session_id="live-conversion",
        proposed_owner_epoch=2,
    )
    destination.import_captured_state("live-conversion", converted)
    validation = destination.validate_imported_state("live-conversion")
    assert validation.dry_run_next_token == source.dry_run_next_token("live-conversion")


def test_quality_bounded_lossy_dtype_conversion_measures_and_enforces_budget() -> None:
    source_layout, destination_layout = _layouts(tokens=13, destination_dtype="float16")
    source = make_random_state(source_layout, seed=8202)
    converted, evidence = quality_bounded_convert(
        source,
        destination_layout,
        maximum_temporary_bytes=_one_token_bound(source_layout, destination_layout) * 3,
        maximum_absolute_error_budget=0.01,
    )
    converted.verify_integrity()
    assert evidence.exactness_class.value == "quality_bounded"
    assert evidence.source_dtype == "float32"
    assert evidence.destination_dtype == "float16"
    assert 0 < evidence.observed_quality_loss <= evidence.quality_budget
    assert evidence.contract_satisfied
    assert evidence.reference_verification.exact

    with pytest.raises(ValueError, match="exceeds"):
        quality_bounded_convert(
            source,
            destination_layout,
            maximum_temporary_bytes=_one_token_bound(source_layout, destination_layout) * 3,
            maximum_absolute_error_budget=1e-12,
        )


def test_quality_bounded_converter_rejects_non_lossy_or_invalid_contracts() -> None:
    source_layout, exact_destination = _layouts(tokens=4)
    source = make_random_state(source_layout, seed=8203)
    with pytest.raises(ValueError, match="lower-precision"):
        quality_bounded_convert(
            source,
            exact_destination,
            maximum_temporary_bytes=_one_token_bound(source_layout, exact_destination),
            maximum_absolute_error_budget=0.01,
        )
    _, lossy_destination = _layouts(tokens=4, destination_dtype="float16")
    with pytest.raises(ValueError, match="positive finite"):
        quality_bounded_convert(
            source,
            lossy_destination,
            maximum_temporary_bytes=_one_token_bound(source_layout, lossy_destination),
            maximum_absolute_error_budget=0.0,
        )
