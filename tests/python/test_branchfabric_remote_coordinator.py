from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
TOOL = ROOT / "tools" / "branchfabric-remote-coordinator.py"


def _environment(base: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "SLOFORGE_COORDINATOR_TEST_MODE": "1",
            "SLOFORGE_COORDINATOR_TEST_BASE": str(base),
        }
    )
    return environment


def _run(base: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=_environment(base),
    )


def _bootstrap(base: Path) -> Path:
    result = _run(base, "bootstrap")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("REMOTE_ROOT=")
    assert lines[1].startswith("BOOTSTRAP_MANIFEST=")
    return Path(lines[0].split("=", 1)[1])


def _load_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location("branchfabric_remote_coordinator", TOOL)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_bootstrap_reports_root_first_and_confines_all_remote_state(tmp_path: Path) -> None:
    remote_root = _bootstrap(tmp_path)

    assert remote_root.parent == tmp_path
    assert remote_root.name.startswith("sloforge-real-cow-")
    assert len(remote_root.name.removeprefix("sloforge-real-cow-")) == 8
    expected = {
        "repo",
        "env",
        "cache",
        "hf",
        "torch",
        "triton",
        "tmp",
        "model",
        "runtime",
        "traces",
        "results",
        "logs",
        "pids",
        "environment",
    }
    assert expected <= {path.name for path in remote_root.iterdir() if path.is_dir()}

    marker = json.loads((remote_root / ".sloforge-experiment-root.json").read_text())
    assert marker["remote_root"] == str(remote_root)
    assert marker["uid"] == os.getuid()
    environment = json.loads((remote_root / "environment/remote-env.json").read_text())
    assert environment["XDG_CACHE_HOME"] == str(remote_root / "cache")
    assert environment["HF_HOME"] == str(remote_root / "hf")
    assert environment["HUGGINGFACE_HUB_CACHE"] == str(remote_root / "hf/hub")
    assert environment["TRANSFORMERS_CACHE"] == str(remote_root / "hf/transformers")
    assert environment["TORCH_HOME"] == str(remote_root / "torch")
    assert environment["TRITON_CACHE_DIR"] == str(remote_root / "triton")
    assert environment["TMPDIR"] == str(remote_root / "tmp")
    assert environment["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert environment["DO_NOT_TRACK"] == "1"
    assert environment["WANDB_MODE"] == "disabled"
    assert environment["WANDB_DISABLED"] == "true"

    capture = json.loads((remote_root / "environment/bootstrap.json").read_text())
    assert capture["captures"]["hostname"]["command"] == ["hostname"]
    assert capture["captures"]["user"]["command"] == ["id", "-un"]
    assert capture["captures"]["disk"]["command"][0:2] == ["df", "-Pk"]
    assert capture["captures"]["nvidia_smi"]["skipped"] is True
    assert "SLOFORGE_COORDINATOR_TEST_MODE" in capture["captures"]["nvidia_smi"]["reason"]
    assert json.loads((remote_root / "pids/processes.json").read_text())["entries"] == []
    assert remote_root.exists()  # No tested subcommand removes the fixture root.


def test_environment_audit_and_artifact_inventory_are_non_destructive(tmp_path: Path) -> None:
    remote_root = _bootstrap(tmp_path)
    (remote_root / "traces/state.jsonl").write_text('{"event":1}\n')
    (remote_root / "results/metrics.json").write_text('{"metric":2}\n')

    capture = _run(tmp_path, "capture-environment", str(remote_root), "before-smoke")
    assert capture.returncode == 0, capture.stderr
    assert (remote_root / "environment/before-smoke.json").is_file()

    inventory = _run(
        tmp_path,
        "inventory-artifacts",
        str(remote_root),
        "environment",
        "traces",
        "results/metrics.json",
    )
    assert inventory.returncode == 0, inventory.stderr
    inventory_document = json.loads((remote_root / "results/artifact-inventory.json").read_text())
    paths = {record["path"] for record in inventory_document["files"]}
    assert "traces/state.jsonl" in paths
    assert "results/metrics.json" in paths
    assert all(len(record["sha256"]) == 64 for record in inventory_document["files"])

    audit = _run(tmp_path, "audit", str(remote_root))
    assert audit.returncode == 0, audit.stderr
    audit_document = json.loads(audit.stdout)
    assert audit_document["exact_path_validated"] is True
    assert audit_document["live_registered_processes"] == []
    assert audit_document["live_unregistered_owned_processes"] == []
    assert audit_document["owned_containers"] == []
    assert remote_root.exists()
    assert (remote_root / "traces/state.jsonl").is_file()


def test_process_registration_fixture_records_exact_identity_without_launching_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_root = _bootstrap(tmp_path)
    module = _load_module()
    monkeypatch.setenv("SLOFORGE_COORDINATOR_TEST_MODE", "1")
    monkeypatch.setenv("SLOFORGE_COORDINATOR_TEST_BASE", str(tmp_path))
    fake_process = {
        "pid": 4242,
        "ppid": 4000,
        "process_group": 4242,
        "uid": os.getuid(),
        "command": "python experiment.py",
        "command_sha256": "a" * 64,
        "purpose": None,
        "start_time_utc": "2026-08-10T00:00:00+00:00",
        "start_time_ticks": 12345,
        "remote_root_environment": str(remote_root),
    }
    monkeypatch.setattr(module, "read_process", lambda _pid: fake_process.copy())

    result = module.command_register_process(
        SimpleNamespace(remote_root=str(remote_root), pid=4242, purpose="runtime server")
    )
    assert result == 0
    registry = json.loads((remote_root / "pids/processes.json").read_text())
    assert registry["entries"] == [
        {
            **fake_process,
            "purpose": "runtime server",
        }
    ]


@pytest.mark.parametrize(
    ("subcommand", "extra"),
    [
        ("shutdown-processes", ()),
        ("shutdown-containers", ()),
        ("cleanup-external", ()),
        ("finalize", ()),
    ],
)
def test_destructive_subcommands_are_disabled_in_local_fixture_mode(
    tmp_path: Path, subcommand: str, extra: tuple[str, ...]
) -> None:
    remote_root = _bootstrap(tmp_path)
    marker_before = (remote_root / ".sloforge-experiment-root.json").read_bytes()
    result = _run(tmp_path, subcommand, str(remote_root), *extra)
    assert result.returncode == 2
    assert "disabled in local test mode" in result.stderr
    assert remote_root.exists()
    assert (remote_root / ".sloforge-experiment-root.json").read_bytes() == marker_before


def test_gpu_selection_is_read_only_and_fails_closed_in_fixture_mode(tmp_path: Path) -> None:
    remote_root = _bootstrap(tmp_path)
    result = _run(tmp_path, "verify-idle-gpu", str(remote_root))
    assert result.returncode == 2
    assert "cannot select an idle GPU" in result.stderr
    selection = json.loads((remote_root / "environment/selected-idle-gpu.json").read_text())
    assert selection["status"] == "unavailable"
    assert selection["selected_gpu"] is None
    assert remote_root.exists()


def test_inventory_rejects_path_escape_without_touching_external_file(tmp_path: Path) -> None:
    remote_root = _bootstrap(tmp_path)
    external = tmp_path / "external.txt"
    external.write_text("preserve me")
    result = _run(
        tmp_path,
        "inventory-artifacts",
        str(remote_root),
        "../external.txt",
    )
    assert result.returncode == 2
    assert "remain relative" in result.stderr
    assert external.read_text() == "preserve me"
    assert remote_root.exists()
