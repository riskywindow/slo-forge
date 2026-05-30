"""Create a tiny local Git history with a deterministic performance regression."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "GIT_AUTHOR_NAME": "SLOForge Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@sloforge.invalid",
            "GIT_COMMITTER_NAME": "SLOForge Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@sloforge.invalid",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    return completed.stdout.strip()


def create_regression_fixture(directory: Path) -> tuple[str, str, str]:
    """Return (good, first_bad, bad) revisions for an isolated linear history."""

    directory.mkdir(parents=True, exist_ok=False)
    _git(directory, "init", "--quiet")
    benchmark = directory / "benchmark.py"
    benchmark.write_text(
        """import json
import os
import random

trial = int(os.environ["SLOFORGE_FORGECI_TRIAL"])
seed = int(os.environ["SLOFORGE_FORGECI_SEED"])
factor = float(open("factor.txt", encoding="utf-8").read())
rng = random.Random(seed * 1009 + trial)
latency = factor * (100.0 + rng.uniform(-0.4, 0.4))
throughput = 10000.0 / latency
print(json.dumps({"p99_ttft_ms": latency, "throughput_rps": throughput}))
""",
        encoding="utf-8",
    )
    (directory / "factor.txt").write_text("1.0\n", encoding="utf-8")
    (directory / "change.txt").write_text("initial\n", encoding="utf-8")
    _git(directory, "add", "benchmark.py", "factor.txt", "change.txt")
    _git(directory, "commit", "--quiet", "-m", "fixture baseline")
    good = _git(directory, "rev-parse", "HEAD")

    (directory / "change.txt").write_text("unrelated-one\n", encoding="utf-8")
    _git(directory, "add", "change.txt")
    _git(directory, "commit", "--quiet", "-m", "unrelated pre-regression change")

    (directory / "factor.txt").write_text("1.12\n", encoding="utf-8")
    _git(directory, "add", "factor.txt")
    _git(directory, "commit", "--quiet", "-m", "introduce deterministic latency regression")
    first_bad = _git(directory, "rev-parse", "HEAD")

    (directory / "change.txt").write_text("unrelated-after\n", encoding="utf-8")
    _git(directory, "add", "change.txt")
    _git(directory, "commit", "--quiet", "-m", "unrelated post-regression change")
    bad = _git(directory, "rev-parse", "HEAD")
    return good, first_bad, bad
