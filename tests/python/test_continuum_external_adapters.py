from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest

from sloforge.continuum.adapters.external import (
    ADAPTER_VERSION,
    AdapterProbe,
    CapabilityName,
    IntegrationStatus,
    OptionalRuntimeNotInstalledError,
    RuntimePackageView,
    SemanticVersion,
    UnsupportedRuntimeVersionError,
)
from sloforge.continuum.adapters.genesis import GenesisRuntimeBinding
from sloforge.continuum.adapters.pytorch import (
    PYTORCH_REQUIREMENTS,
    PyTorchRuntimeBinding,
    probe_pytorch,
)
from sloforge.continuum.adapters.sdk import UnsupportedCapabilityError
from sloforge.continuum.adapters.sglang import (
    SGLANG_REQUIREMENTS,
    SglangPdConfiguration,
    SglangRuntimeBinding,
    probe_sglang,
)
from sloforge.continuum.adapters.vllm import (
    VLLM_REQUIREMENTS,
    VllmRuntimeBinding,
    probe_vllm,
)
from sloforge.genesis.frontend import inspect_reference_package
from sloforge.genesis.runtime import generate_baseline_runtime

ROOT = Path(__file__).resolve().parents[2]
HYBRID = ROOT / "models" / "reference_tasks" / "hybrid_decoder"


def _view(runtime: str, version: str, symbols: tuple[str, ...]) -> RuntimePackageView:
    return RuntimePackageView(
        distribution_name=runtime,
        import_name=runtime,
        version=version,
        available_symbols=frozenset(symbols),
        source="static_fixture",
    )


def test_version_gates_and_public_api_requirements_fail_closed() -> None:
    assert SemanticVersion.parse("0.23.0.post1") == SemanticVersion(0, 23, 0)

    unsupported = probe_vllm(_view("vllm", "0.24.0", VLLM_REQUIREMENTS))
    assert unsupported.status is IntegrationStatus.VERSION_UNSUPPORTED
    with pytest.raises(UnsupportedRuntimeVersionError):
        unsupported.require_ready()

    incomplete = probe_sglang(_view("sglang", "0.5.12", SGLANG_REQUIREMENTS[:-1]))
    assert incomplete.status is IntegrationStatus.API_INCOMPATIBLE
    assert incomplete.missing_requirements == (SGLANG_REQUIREMENTS[-1],)


def test_optional_runtime_not_installed_is_typed() -> None:
    probe = AdapterProbe(
        runtime_name="pytorch",
        runtime_version=None,
        adapter_version=ADAPTER_VERSION,
        status=IntegrationStatus.PACKAGE_NOT_INSTALLED,
        capabilities=frozenset(),
        missing_requirements=("install pytorch",),
        evidence=(),
        exercised=False,
    )
    with pytest.raises(OptionalRuntimeNotInstalledError) as captured:
        PyTorchRuntimeBinding(probe).capture_cpu_rng_state()
    assert captured.value.code == "optional_runtime_not_installed"


def test_static_vllm_and_sglang_bindings_do_not_claim_portable_state() -> None:
    vllm_probe = probe_vllm(_view("vllm", "0.23.0", VLLM_REQUIREMENTS))
    assert vllm_probe.status is IntegrationStatus.READY
    assert not vllm_probe.exercised
    with pytest.raises(UnsupportedCapabilityError, match="runtime-native cache movement"):
        VllmRuntimeBinding(vllm_probe).require_portable_execution_state_export()

    sglang_probe = probe_sglang(_view("sglang", "0.5.12", SGLANG_REQUIREMENTS))
    binding = SglangRuntimeBinding(sglang_probe)
    arguments = binding.build_pd_launch_arguments(
        SglangPdConfiguration(
            role="decode",
            transfer_backend="nixl",
            page_size_tokens=64,
            tensor_parallel_degree=2,
            pipeline_parallel_degree=1,
            random_seed=73129,
        )
    )
    assert arguments == (
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "nixl",
        "--page-size",
        "64",
        "--tp-size",
        "2",
        "--pp-size",
        "1",
        "--random-seed",
        "73129",
    )
    with pytest.raises(UnsupportedCapabilityError, match="runtime-native transfer"):
        binding.require_portable_execution_state_export()


def test_pytorch_cpu_state_when_installed_or_typed_absence() -> None:
    probe = probe_pytorch()
    try:
        installed_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        assert probe.status is IntegrationStatus.PACKAGE_NOT_INSTALLED
        with pytest.raises(OptionalRuntimeNotInstalledError):
            PyTorchRuntimeBinding(probe).capture_cpu_rng_state()
        return

    assert probe.runtime_version == installed_version
    if probe.status is IntegrationStatus.READY:
        import torch

        binding = PyTorchRuntimeBinding(probe)
        original = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 1)
        captured = binding.capture_cpu_tensor(original)
        restored = binding.import_cpu_tensor(captured)
        assert torch.equal(original.contiguous(), restored)
        rng = binding.capture_cpu_rng_state()
        first = torch.rand(3)
        binding.restore_cpu_rng_state(rng)
        second = torch.rand(3)
        assert torch.equal(first, second)
    else:
        assert probe.status in {
            IntegrationStatus.VERSION_UNSUPPORTED,
            IntegrationStatus.API_INCOMPATIBLE,
        }


def test_genesis_generated_runtime_is_loaded_and_exercised(tmp_path: Path) -> None:
    inspection = inspect_reference_package(HYBRID)
    bundle = generate_baseline_runtime(
        HYBRID,
        inspection,
        tmp_path / "generated-runtime",
        seed=73129,
    )
    binding = GenesisRuntimeBinding.from_config(bundle.output_directory / "runtime_config.json")

    assert binding.descriptor.runtime_id == bundle.runtime_id
    assert binding.descriptor.package_hash == bundle.package_hash
    result = binding.run_cpu_smoke(
        request_id="continuum-genesis-smoke",
        text="portable state",
        maximum_new_tokens=4,
        seed=42,
        timeout_seconds=5.0,
    )
    assert len(result.token_ids) == 4
    assert result.terminal_kind == "completed"
    assert result.health_before == "ready"
    assert result.health_after == "stopped"
    with pytest.raises(UnsupportedCapabilityError, match="does not publish"):
        binding.require_portable_execution_state_export()


def test_external_adapter_manifests_are_machine_readable() -> None:
    for runtime in ("pytorch", "genesis", "vllm", "sglang"):
        path = ROOT / "adapters" / "continuum" / runtime / "compatibility.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["adapter_schema"] == "sloforge.continuum.external-adapter/v1"
        assert document["adapter_version"] == ADAPTER_VERSION
        assert document["runtime"] == runtime
        assert document["source_lock"]


def test_pytorch_static_public_contract_is_version_scoped() -> None:
    probe = probe_pytorch(_view("torch", "2.13.0", PYTORCH_REQUIREMENTS))
    assert probe.status is IntegrationStatus.READY
    assert CapabilityName.CPU_TENSOR_STATE in probe.capabilities
    assert not probe.exercised
