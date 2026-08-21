from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "tools/branchfabric-experiment-004-local-loadgen-gate.py"


def test_local_loadgen_gate_is_machine_readable_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    paths = (tmp_path / "gate-a.json", tmp_path / "gate-b.json")
    stdout_payloads = []
    for path in paths:
        completed = subprocess.run(
            [sys.executable, str(GATE), "--seed", "41", "--output", str(path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        assert completed.returncode == 0, completed.stderr
        stdout_payloads.append(json.loads(completed.stdout))
        assert json.loads(path.read_text()) == stdout_payloads[-1]

    assert stdout_payloads[0] == stdout_payloads[1]
    payload = stdout_payloads[0]
    assert payload["fixture_kind"] == (
        "deterministic-cpu-methodology-validation-not-gpu-measurement"
    )
    assert payload["seed"] == 41
    assert payload["passed"] is True
    assert payload["status"] == "passed"
    assert payload["assertions"] == {
        "completion_rate_independently_recomputed": True,
        "configured_actual_arrival_rate_match": True,
        "fixture_operating_points_sustainable": True,
        "gpu0_only_route_exact": True,
        "load_reduced_before_restore": True,
        "no_client_timer_burst": True,
        "offered_rate_independently_recomputed": True,
        "output_length_exactly_64": True,
        "queue_depth_independently_recomputed": True,
        "queue_flow_conservation_independently_recomputed": True,
        "routing_cutovers_exact": True,
        "timestamps_monotonic": True,
        "ttft_independently_recomputed": True,
        "two_gpu_route_exact": True,
        "warmup_carry_in_accounted_without_cohort_contamination": True,
    }
    assert payload["provenance"]["fixture_input_sha256"] == (
        "81441ed0354ee68ef0e4eaa48bcc5ec781b554d01cb38a9a604cb0575424fc55"
    )
    assert payload["derived_metrics"]["gpu0_only"]["offered_rate_rps"] == 20.0
    assert payload["derived_metrics"]["gpu0_only"]["completion_rate_rps"] == 20.0
    assert payload["derived_metrics"]["two_gpu"]["offered_rate_rps"] == 30.0
    assert payload["derived_metrics"]["two_gpu"]["completion_rate_rps"] == 30.0
