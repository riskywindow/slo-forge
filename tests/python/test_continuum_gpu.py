from __future__ import annotations

from dataclasses import replace

import pytest

from sloforge.continuum.adapters.external import IntegrationStatus
from sloforge.continuum.adapters.pytorch import probe_pytorch
from sloforge.continuum.conversion import (
    KVLayout,
    KVLayoutKind,
    canonical_convert,
    make_random_state,
    pytorch_convert,
)


@pytest.mark.continuum_gpu
def test_pytorch_cuda_conversion_matches_independent_canonical_converter() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("a CUDA-enabled PyTorch device is unavailable")
    probe = probe_pytorch()
    assert probe.status is IntegrationStatus.READY, probe.detail
    source_layout = KVLayout(
        kind=KVLayoutKind.TOKEN_MAJOR_SEPARATE,
        tensor_parallel_degree=4,
        page_size_tokens=3,
        layer_count=2,
        token_count=11,
        kv_head_count=8,
        head_dim=16,
    )
    destination_layout = replace(
        source_layout,
        kind=KVLayoutKind.HEAD_MAJOR_PACKED,
        tensor_parallel_degree=2,
        page_size_tokens=5,
    )
    source = make_random_state(source_layout, seed=820317)
    converted, evidence = pytorch_convert(
        source,
        destination_layout,
        maximum_temporary_bytes=64 * 1024,
        device="cuda",
        probe=probe,
    )
    canonical = canonical_convert(source, destination_layout)
    assert evidence.gpu_exercised
    assert evidence.exercised_device == "cuda"
    assert evidence.canonical_match
    assert evidence.numeric_contract_satisfied
    assert converted.content_hash == canonical.content_hash
