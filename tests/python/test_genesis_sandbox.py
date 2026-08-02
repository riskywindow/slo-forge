from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from sloforge.genesis.sandbox import (
    IsolationStatus,
    SandboxBackend,
    SandboxLimits,
    SandboxRequest,
    SandboxTermination,
    detect_capabilities,
    execute_sandboxed,
)
from sloforge.genesis.sandbox.executor import _macos_firmlink_alias, _process_group_rss_bytes


def test_macos_firmlink_aliases_are_symmetric() -> None:
    assert _macos_firmlink_alias(Path("/tmp/genesis")) == Path("/private/tmp/genesis")
    assert _macos_firmlink_alias(Path("/private/tmp/genesis")) == Path("/tmp/genesis")
    assert _macos_firmlink_alias(Path("/var/folders/genesis")) == Path(
        "/private/var/folders/genesis"
    )
    assert _macos_firmlink_alias(Path("/private/var/folders/genesis")) == Path(
        "/var/folders/genesis"
    )
    assert _macos_firmlink_alias(Path("/Users/genesis")) is None


def _request(
    source: Path,
    output: Path,
    script: Path,
    *,
    wall_time: float = 3.0,
    output_bytes: int = 4096,
) -> SandboxRequest:
    return SandboxRequest(
        argv=(sys.executable, str(script)),
        working_directory=source,
        read_only_paths=(source,),
        artifact_output_directory=output,
        seed=73129,
        limits=SandboxLimits(
            wall_time_seconds=wall_time,
            cpu_time_seconds=2,
            memory_bytes=2 * 1024 * 1024 * 1024,
            process_count=8,
            output_bytes=output_bytes,
            artifact_bytes=1024 * 1024,
            artifact_entries=128,
            open_files=32,
        ),
    )


def test_sandbox_capabilities_are_explicit() -> None:
    capabilities = detect_capabilities()
    assert capabilities.environment_sanitization is IsolationStatus.ENFORCED
    assert capabilities.output_limit is IsolationStatus.ENFORCED
    assert capabilities.limitations


def test_parent_resource_watchdog_observes_its_process_group() -> None:
    import os

    observed = _process_group_rss_bytes(os.getpgrp())
    assert observed is None or observed > 0


def test_sandbox_sanitizes_environment_and_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    script = source / "environment.py"
    script.write_text(
        "import os\n"
        "print(os.environ.get('AWS_SECRET_ACCESS_KEY', 'absent'))\n"
        "print(os.environ['PYTHONHASHSEED'])\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    result = execute_sandboxed(_request(source, output, script))
    if result.capabilities.network_isolation is IsolationStatus.UNAVAILABLE:
        assert result.termination is SandboxTermination.POLICY_UNAVAILABLE
        return
    assert result.termination is SandboxTermination.SUCCESS, result.stderr
    assert result.stdout.splitlines() == ["absent", "73129"]
    assert "AWS_SECRET_ACCESS_KEY" not in result.sanitized_environment_names


def test_relative_sandbox_request_gets_canonical_home_and_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("source")
    source.mkdir()
    script = source / "paths.py"
    script.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "print(Path(os.environ['HOME']))\n"
        "print(Path(os.environ['TMPDIR']))\n",
        encoding="utf-8",
    )
    result = execute_sandboxed(_request(source, Path("output"), script.resolve()))
    if result.capabilities.network_isolation is IsolationStatus.UNAVAILABLE:
        assert result.termination is SandboxTermination.POLICY_UNAVAILABLE
        return
    assert result.termination is SandboxTermination.SUCCESS, result.stderr
    assert result.stdout.splitlines() == [
        str((tmp_path / "output/home").resolve()),
        str((tmp_path / "output/tmp").resolve()),
    ]


def test_sandbox_denies_network_and_undeclared_reads(tmp_path: Path) -> None:
    capabilities = detect_capabilities()
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    secret = tmp_path / "credential.txt"
    secret.write_text("secret", encoding="utf-8")
    script = source / "adversary.py"
    script.write_text(
        "import pathlib, socket\n"
        "try:\n"
        f"    pathlib.Path({str(secret)!r}).read_text()\n"
        "except OSError:\n"
        "    print('read-blocked')\n"
        "else:\n"
        "    print('read-leaked')\n"
        "sock = socket.socket()\n"
        "try:\n"
        "    sock.connect(('127.0.0.1', 9))\n"
        "except OSError:\n"
        "    print('network-blocked')\n"
        "else:\n"
        "    print('network-open')\n",
        encoding="utf-8",
    )
    result = execute_sandboxed(_request(source, output, script))
    if capabilities.network_isolation is IsolationStatus.UNAVAILABLE:
        assert result.termination is SandboxTermination.POLICY_UNAVAILABLE
        assert "not executed" in result.stderr
        return
    assert result.termination is SandboxTermination.SUCCESS, result.stderr
    assert result.stdout.splitlines() == ["read-blocked", "network-blocked"]


def test_macos_sandbox_denies_undeclared_system_reads(tmp_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sloforge-undeclared-") as unrelated:
        secret = Path(unrelated) / "secret.txt"
        secret.write_text("secret", encoding="utf-8")
        source = tmp_path / "source"
        source.mkdir()
        script = source / "system_read.py"
        script.write_text(
            "import pathlib\n"
            "try:\n"
            f"    pathlib.Path({str(secret)!r}).read_text()\n"
            "except OSError:\n"
            "    print('read-blocked')\n"
            "else:\n"
            "    print('read-leaked')\n",
            encoding="utf-8",
        )
        result = execute_sandboxed(_request(source, tmp_path / "output", script))
        if result.capabilities.backend is not SandboxBackend.MACOS_SANDBOX_EXEC:
            return
        assert result.termination is SandboxTermination.SUCCESS, result.stderr
        assert result.stdout.strip() == "read-blocked"


def test_sandbox_refuses_nonempty_artifact_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = source / "ok.py"
    script.write_text("print('must-not-run')\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "trusted-evidence.json"
    marker.write_text("preserve", encoding="utf-8")
    result = execute_sandboxed(_request(source, output, script))
    if result.capabilities.network_isolation is IsolationStatus.UNAVAILABLE:
        assert result.termination is SandboxTermination.POLICY_UNAVAILABLE
        return
    assert result.termination is SandboxTermination.SETUP_ERROR
    assert "must be empty" in result.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert "must-not-run" not in result.stdout


def test_sandbox_kills_timed_out_process_group(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    script = source / "forever.py"
    script.write_text("while True:\n    1 + 1\n", encoding="utf-8")
    result = execute_sandboxed(_request(source, output, script, wall_time=0.2))
    if result.capabilities.network_isolation is IsolationStatus.UNAVAILABLE:
        assert result.termination is SandboxTermination.POLICY_UNAVAILABLE
        return
    assert result.termination is SandboxTermination.TIMEOUT
    assert result.process_group_cleaned


def test_sandbox_stops_output_flood(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    script = source / "flood.py"
    script.write_text("import os\nos.write(1, b'x' * 1000000)\n", encoding="utf-8")
    result = execute_sandboxed(_request(source, output, script, output_bytes=128))
    if result.capabilities.network_isolation is IsolationStatus.UNAVAILABLE:
        assert result.termination is SandboxTermination.POLICY_UNAVAILABLE
        return
    assert result.termination is SandboxTermination.OUTPUT_LIMIT
    assert len(result.stdout.encode()) <= 128
    assert result.output_bytes > 128


def test_sandbox_stops_artifact_entry_flood_while_process_is_running(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    script = source / "artifact_flood.py"
    script.write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['HOME']).parent\n"
        "for index in range(256):\n"
        "    (root / f'artifact-{index}').write_text('x')\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    result = execute_sandboxed(_request(source, output, script, wall_time=3.0))
    if result.capabilities.network_isolation is IsolationStatus.UNAVAILABLE:
        assert result.termination is SandboxTermination.POLICY_UNAVAILABLE
        return
    assert result.termination is SandboxTermination.SANDBOX_VIOLATION
    assert "entry limit" in result.stderr
    assert result.duration_seconds < 2.5
    assert result.process_group_cleaned


def test_sandbox_rejects_credential_environment_name(tmp_path: Path) -> None:
    from sloforge.genesis.sandbox import EnvironmentVariable

    source = tmp_path / "source"
    source.mkdir()
    script = source / "ok.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    request = _request(source, tmp_path / "output", script).model_copy(
        update={"environment": (EnvironmentVariable(name="API_TOKEN", value="secret"),)}
    )
    result = execute_sandboxed(request)
    if result.capabilities.network_isolation is IsolationStatus.UNAVAILABLE:
        assert result.termination is SandboxTermination.POLICY_UNAVAILABLE
    else:
        assert result.termination is SandboxTermination.SETUP_ERROR
        assert "forbidden" in result.stderr


def test_sandbox_writes_only_to_artifact_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    script = source / "writes.py"
    script.write_text(
        "import os, pathlib\n"
        "try:\n"
        "    pathlib.Path('forbidden').write_text('bad')\n"
        "except OSError:\n"
        "    print('source-write-blocked')\n"
        "target = pathlib.Path(os.environ['HOME']).parent / 'result.txt'\n"
        "target.write_text('accepted artifact')\n"
        "print('artifact-written')\n",
        encoding="utf-8",
    )
    result = execute_sandboxed(_request(source, output, script))
    if result.capabilities.network_isolation is IsolationStatus.UNAVAILABLE:
        assert result.termination is SandboxTermination.POLICY_UNAVAILABLE
        return
    assert result.termination is SandboxTermination.SUCCESS, result.stderr
    assert result.stdout.splitlines() == ["source-write-blocked", "artifact-written"]
    assert (output / "result.txt").read_text(encoding="utf-8") == "accepted artifact"
    assert not (source / "forbidden").exists()


def test_macos_sandbox_denies_generated_child_process(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    script = source / "spawn.py"
    script.write_text(
        "import subprocess\n"
        "try:\n"
        "    subprocess.run(['/usr/bin/true'], check=False)\n"
        "except OSError:\n"
        "    print('spawn-blocked')\n"
        "else:\n"
        "    print('spawn-allowed')\n",
        encoding="utf-8",
    )
    result = execute_sandboxed(_request(source, output, script))
    if result.capabilities.backend is not SandboxBackend.MACOS_SANDBOX_EXEC:
        assert result.capabilities.process_limit in {
            IsolationStatus.BEST_EFFORT,
            IsolationStatus.ENFORCED,
        }
        return
    assert result.termination is SandboxTermination.SUCCESS, result.stderr
    assert result.stdout.strip() == "spawn-blocked"


def test_sandbox_rejects_symlink_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    script = source / "symlink.py"
    script.write_text(
        "import os, pathlib\n"
        "target = pathlib.Path(os.environ['HOME']).parent / 'link'\n"
        "target.symlink_to('/etc/passwd')\n",
        encoding="utf-8",
    )
    result = execute_sandboxed(_request(source, output, script))
    if result.capabilities.network_isolation is IsolationStatus.UNAVAILABLE:
        assert result.termination is SandboxTermination.POLICY_UNAVAILABLE
        return
    assert result.termination is SandboxTermination.SANDBOX_VIOLATION
    assert "symlink" in result.stderr
