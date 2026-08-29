from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CONTROLLER_PATH = _ROOT / "experiments/branchfabric/gpu_cow_controller.py"
_CONTROLLER_SPEC = importlib.util.spec_from_file_location(
    "sloforge_gpu_cow_controller_lifecycle_tests",
    _CONTROLLER_PATH,
)
assert _CONTROLLER_SPEC is not None and _CONTROLLER_SPEC.loader is not None
_CONTROLLER = importlib.util.module_from_spec(_CONTROLLER_SPEC)
_CONTROLLER_SPEC.loader.exec_module(_CONTROLLER)


def _gpu_sample(memory_used_mib: int, *, processes: list[dict[str, object]] | None = None):
    return {
        "observed_at_utc": "2026-08-11T00:00:00+00:00",
        "observed_at_monotonic_ns": 1,
        "gpu": {
            "index": 0,
            "uuid": "GPU-local-lifecycle-test",
            "name": "NVIDIA A100 80GB PCIe",
            "driver_version": "test-driver",
            "memory_total_mib": 81_920,
            "memory_used_mib": memory_used_mib,
            "utilization_percent": 0,
        },
        "compute_processes": [] if processes is None else processes,
        "commands": {},
    }


def test_controller_fresh_import_is_cuda_clean_and_standard_library_only() -> None:
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imported_roots.add((node.module or "").partition(".")[0])
    assert imported_roots <= sys.stdlib_module_names

    dynamic_import_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "__import__")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
        )
    ]
    assert dynamic_import_calls == []

    probe = textwrap.dedent(
        """
        import importlib.util
        import json
        import sys

        path = sys.argv[1]
        spec = importlib.util.spec_from_file_location("fresh_gpu_cow_controller", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(json.dumps(module.forbidden_import_audit(stage="fresh-import")))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(_CONTROLLER_PATH)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    audit = json.loads(completed.stdout)
    assert audit["stage"] == "fresh-import"
    assert audit["cuda_clean"] is True
    assert audit["loaded_forbidden_modules"] == []


def test_forbidden_import_audit_matches_module_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = {
        "torch": object(),
        "torch.cuda": object(),
        "vllm.worker": object(),
        "triton": object(),
        "cuda.bindings": object(),
        "numba.cuda": object(),
    }
    for name, value in sentinels.items():
        monkeypatch.setitem(sys.modules, name, value)

    audit = _CONTROLLER.forbidden_import_audit(stage="unit-test")

    assert audit["cuda_clean"] is False
    assert set(sentinels) <= set(audit["loaded_forbidden_modules"])
    assert audit["source"] == "sys.modules"


def test_nvidia_smi_parsers_accept_exact_query_shapes() -> None:
    inventory = _CONTROLLER.parse_gpu_inventory(
        "0, GPU-1234, NVIDIA A100 80GB PCIe, 580.95.05, 81920, 14213, 37\n"
    )
    assert inventory == {
        "index": 0,
        "uuid": "GPU-1234",
        "name": "NVIDIA A100 80GB PCIe",
        "driver_version": "580.95.05",
        "memory_total_mib": 81_920,
        "memory_used_mib": 14_213,
        "utilization_percent": 37,
    }
    assert _CONTROLLER.parse_compute_apps("No running processes found\n") == []
    assert _CONTROLLER.parse_compute_apps("GPU-1234, 9182, /usr/bin/python3, 14196\n") == [
        {
            "gpu_uuid": "GPU-1234",
            "host_pid": 9182,
            "process_name": "/usr/bin/python3",
            "used_gpu_memory_mib": 14_196,
        }
    ]


@pytest.mark.parametrize(
    ("parser", "stdout", "message"),
    [
        (_CONTROLLER.parse_gpu_inventory, "", "exactly one visible GPU"),
        (
            _CONTROLLER.parse_gpu_inventory,
            "0, GPU-a, A100, driver, 81920, 4, 0\n1, GPU-b, A100, driver, 81920, 4, 0\n",
            "exactly one visible GPU",
        ),
        (
            _CONTROLLER.parse_gpu_inventory,
            "zero, GPU-a, A100, driver, 81920, 4, 0\n",
            "non-integer field",
        ),
        (
            _CONTROLLER.parse_compute_apps,
            "GPU-a, not-a-pid, python, 1024\n",
            "row was invalid",
        ),
        (
            _CONTROLLER.parse_compute_apps,
            "GPU-a, 42, python\n",
            "unexpected shape",
        ),
    ],
)
def test_nvidia_smi_parsers_fail_closed(parser, stdout: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parser(stdout)


def test_cleanup_evaluation_requires_three_uninterrupted_clean_samples() -> None:
    dirty_process = [
        {
            "gpu_uuid": "GPU-local-lifecycle-test",
            "host_pid": 99,
            "process_name": "python",
            "used_gpu_memory_mib": 128,
        }
    ]
    samples = [
        _gpu_sample(1028),
        _gpu_sample(1028),
        _gpu_sample(1029),
        _gpu_sample(8, processes=dirty_process),
        _gpu_sample(8),
        _gpu_sample(8),
    ]
    failed = _CONTROLLER.evaluate_cleanup_samples(
        samples,
        threshold_mib=1028,
        required_consecutive=3,
    )
    assert failed["passed"] is False
    assert failed["winning_sample_start_index"] is None
    assert failed["winning_sample_end_index"] is None
    assert failed["maximum_consecutive_qualifying_samples"] == 2

    passed = _CONTROLLER.evaluate_cleanup_samples(
        [*samples, _gpu_sample(8)],
        threshold_mib=1028,
        required_consecutive=3,
    )
    assert passed["passed"] is True
    assert passed["winning_sample_start_index"] == 4
    assert passed["winning_sample_end_index"] == 6


def test_controller_uses_fresh_process_group_and_serializes_cleanup_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in list(sys.modules):
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _CONTROLLER.FORBIDDEN_GPU_MODULE_PREFIXES
        ):
            monkeypatch.delitem(sys.modules, name)
    worker = tmp_path / "fixture_worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import os
            import subprocess
            import sys
            import time

            parser = argparse.ArgumentParser()
            parser.add_argument("--config")
            parser.add_argument("--work-root")
            parser.add_argument("--model-snapshot")
            parser.add_argument("--gpu-uuid")
            parser.add_argument("--event-log")
            parser.add_argument("--child-manifest")
            args = parser.parse_args()

            grandchild = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(0.15)"]
            )
            time.sleep(0.05)
            grandchild.wait(timeout=2)
            manifest = {
                "schema_version": "fixture-child-manifest/v1",
                "status": "succeeded",
                "pid": os.getpid(),
                "pgid": os.getpgid(0),
                "gpu_uuid": args.gpu_uuid,
                "semantic_invariants": {
                    "shared_root": True,
                    "private_suffix": True,
                },
                "final_runtime_assigned_kv_bytes": 0,
                "summary": {"fixture": True},
            }
            with open(args.child_manifest, "x", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\\n")
                handle.flush()
                os.fsync(handle.fileno())
            print("fixture worker complete", flush=True)
            """
        ),
        encoding="utf-8",
    )
    model_snapshot = tmp_path / "model"
    model_snapshot.mkdir()
    sample_calls = 0

    def fake_sampler(expected_uuid: str | None):
        nonlocal sample_calls
        sample_calls += 1
        assert expected_uuid in {None, "GPU-local-lifecycle-test"}
        baseline_values = [12, 4, 8]
        memory = baseline_values[sample_calls - 1] if sample_calls <= 3 else 8
        return _gpu_sample(memory)

    work_root = tmp_path / "attempt"
    result = _CONTROLLER.run_controller(
        config_payload={
            "attempt_id": "local-lifecycle-v1",
            "initialization_timeout_seconds": 1,
            "maximum_wall_seconds": 2,
            "cleanup_timeout_seconds": 1,
        },
        work_root=work_root,
        worker_path=worker,
        model_snapshot=model_snapshot,
        gpu_sampler=fake_sampler,
        postflight_timeout_s=3.0,
        baseline_interval_s=0.0,
        monitor_interval_s=0.01,
    )

    assert result["status"] == "succeeded"
    assert result["cleanup_gate"]["GPU_RUNTIME_LIFECYCLE_CLEAN"] is True
    assert result["cleanup_gate"]["forced_gpu_process_kill_required"] is False
    assert result["controller_manifest"]["parent_cuda_clean"] is True
    assert result["controller_manifest"]["child_pid"] == result["controller_manifest"]["child_pgid"]
    assert result["child_manifest"]["pid"] == result["controller_manifest"]["child_pid"]
    assert result["child_manifest"]["pgid"] == result["controller_manifest"]["child_pgid"]
    assert result["child_manifest"]["gpu_uuid"] == "GPU-local-lifecycle-test"

    lifecycle = work_root / "lifecycle"
    required_artifacts = {
        "controller-manifest.json",
        "child-manifest.json",
        "process-tree-before.json",
        "process-tree-during.json",
        "process-tree-after.json",
        "hbm-baseline.json",
        "hbm-postflight.json",
        "cleanup-gate.json",
        "forbidden-import-audit.json",
    }
    assert required_artifacts <= {path.name for path in lifecycle.iterdir()}
    baseline = json.loads((lifecycle / "hbm-baseline.json").read_text())
    assert baseline["baseline_samples_mib"] == [12, 4, 8]
    assert baseline["baseline_median_mib"] == 8
    assert baseline["cleanup_threshold_mib"] == 1032
    cleanup = json.loads((lifecycle / "cleanup-gate.json").read_text())
    assert cleanup["parent_cuda_clean"] is True
    assert cleanup["stable_sample_evaluation"]["required_consecutive_samples"] == 3
    assert cleanup["stable_sample_evaluation"]["winning_sample_end_index"] == 2
    assert cleanup["no_owned_descendant_remained"] is True
    import_audit = json.loads((lifecycle / "forbidden-import-audit.json").read_text())
    assert import_audit["cuda_clean"] is True
    assert [audit["stage"] for audit in import_audit["audits"]] == [
        "controller_entry_before_baseline",
        "immediately_before_child_launch",
        "after_child_exit_and_hbm_postflight",
    ]
    during = json.loads((lifecycle / "process-tree-during.json").read_text())
    assert during["child_pid"] == during["child_pgid"]
    assert during["tracked_owned_processes"]
    assert json.loads((lifecycle / "process-tree-after.json").read_text())["processes"]
    assert (work_root / "logs/child.stdout.log").read_text() == "fixture worker complete\n"
    for path in lifecycle.glob("*.json"):
        assert path.read_bytes().endswith(b"\n")
