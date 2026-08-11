from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click import unstyle
from typer.testing import CliRunner

from sloforge.cli.main import app
from sloforge.fabric.ir import (
    RankPlacement,
    WorkerRole,
    canonical_hash,
    load_physical_execution_plan,
    load_topology_graph,
    save_physical_execution_plan,
)
from sloforge.ir import ArtifactDigest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "fabric"
runner = CliRunner()


def _matching_plan(tmp_path: Path, *, aggregated: bool = False) -> tuple[Path, Path]:
    topology_path = FIXTURES / "topology-graph-v1.json"
    topology = load_topology_graph(topology_path)
    plan = load_physical_execution_plan(FIXTURES / "physical-execution-plan-v1.json")
    updates: dict[str, object] = {
        "topology_fingerprint": ArtifactDigest(value=canonical_hash(topology))
    }
    if aggregated:
        bindings = tuple(
            binding.model_copy(update={"worker_role": WorkerRole.AGGREGATED})
            for binding in plan.rank_placement.bindings
        )
        updates.update(
            {
                "parallelism": plan.parallelism.model_copy(
                    update={"prefill_decode_disaggregated": False}
                ),
                "rank_placement": RankPlacement(bindings=bindings),
                "kv_transfer": None,
            }
        )
    plan_path = tmp_path / "physical-plan.json"
    save_physical_execution_plan(plan_path, plan.model_copy(update=updates))
    return plan_path, topology_path


def _arguments(
    *, plan: Path, topology: Path, output: Path, target: str, runtime: str = "native"
) -> list[str]:
    versions = {"native": "0.1.0", "vllm": "0.26.0", "sglang": "0.5.15"}
    return [
        "fabric",
        "export",
        "--plan",
        str(plan),
        "--topology",
        str(topology),
        "--target",
        target,
        "--output",
        str(output),
        "--model-id",
        "Qwen/Qwen3-0.6B",
        "--model-revision",
        "main",
        "--image",
        "ghcr.io/sloforge/runtime:0.1.0",
        "--runtime",
        runtime,
        "--runtime-version",
        versions[runtime],
    ]


def _invoke(arguments: list[str]) -> str:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output or repr(result.exception)
    return result.output


def test_fabric_export_cli_emits_fail_closed_local_artifacts(tmp_path: Path) -> None:
    plan, topology = _matching_plan(tmp_path)
    output = tmp_path / "local"
    result = json.loads(
        _invoke(_arguments(plan=plan, topology=topology, output=output, target="local"))
    )

    launch = json.loads((output / "launch-plan.json").read_text(encoding="utf-8"))
    assert launch["binding_policy"] == "fail_on_mismatch"
    assert result["target"] == "local"
    assert result["deployed"] is False
    assert result == json.loads((output / "export-result.json").read_text(encoding="utf-8"))


def test_fabric_export_cli_emits_kubernetes_physical_constraints(tmp_path: Path) -> None:
    plan, topology = _matching_plan(tmp_path)
    output = tmp_path / "kubernetes"
    _invoke(_arguments(plan=plan, topology=topology, output=output, target="kubernetes"))

    manifest = yaml.safe_load((output / "physical-plan.yaml").read_text(encoding="utf-8"))
    deployment = next(item for item in manifest["items"] if item["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    assert pod["affinity"]["nodeAffinity"]
    assert pod["topologySpreadConstraints"]
    assert deployment["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0


def test_fabric_export_cli_selects_validated_sglang_runtime(tmp_path: Path) -> None:
    plan, topology = _matching_plan(tmp_path, aggregated=True)
    output = tmp_path / "sglang-local"
    _invoke(
        _arguments(
            plan=plan,
            topology=topology,
            output=output,
            target="local",
            runtime="sglang",
        )
    )

    launch = json.loads((output / "launch-plan.json").read_text(encoding="utf-8"))
    command = launch["groups"][0]["command"]
    assert command[:3] == ["python", "-m", "sglang.launch_server"]
    assert command[command.index("--ep-size") + 1] == "2"


def test_fabric_export_cli_rejects_invalid_target_without_output(tmp_path: Path) -> None:
    plan, topology = _matching_plan(tmp_path)
    output = tmp_path / "invalid-target"
    result = runner.invoke(
        app,
        _arguments(plan=plan, topology=topology, output=output, target="nominal-cloud"),
    )

    assert result.exit_code == 2
    assert "Invalid value for '--target'" in unstyle(result.output)
    assert not output.exists()


def test_fabric_export_cli_rejects_mismatched_context_without_output(tmp_path: Path) -> None:
    topology = FIXTURES / "topology-graph-v1.json"
    plan = FIXTURES / "physical-execution-plan-v1.json"
    output = tmp_path / "invalid-context"
    result = runner.invoke(
        app,
        _arguments(plan=plan, topology=topology, output=output, target="local"),
    )

    assert result.exit_code == 2
    assert "invalid physical adapter context" in result.output
    assert "topology fingerprint" in result.output
    assert not output.exists()


def test_fabric_export_cli_rejects_preexisting_output(tmp_path: Path) -> None:
    plan, topology = _matching_plan(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "owned-by-user"
    marker.write_text("preserve", encoding="utf-8")

    result = runner.invoke(
        app,
        _arguments(plan=plan, topology=topology, output=output, target="local"),
    )

    assert result.exit_code == 2
    assert "must not already exist" in result.output
    assert marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("target", ["modal", "truss"])
def test_fabric_export_cli_requires_explicit_advisory_cloud_opt_in(
    target: str, tmp_path: Path
) -> None:
    plan, topology = _matching_plan(tmp_path)
    output = tmp_path / target
    result = runner.invoke(
        app,
        _arguments(plan=plan, topology=topology, output=output, target=target),
    )

    assert result.exit_code == 2
    assert "physical export rejected" in result.output
    assert "rank/GPU/NIC placement" in result.output
    assert not output.exists()
