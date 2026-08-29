from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.pop("SLOFORGE_MODAL_PREFLIGHT_TOKEN", None)
sys.path.insert(0, str(Path("experiments/branchfabric").resolve()))

from build_experiment_003_campaign import build_campaign
from modal_metadata_characterization import (
    CAMPAIGN_RESERVATION_GPU_HOURS,
    GPU_HOUR_HARD_LIMIT,
    GPU_HOUR_TARGET,
    SCHEMA,
    _attach_function_call,
    _budget,
    _load_gpu_ledger,
    _record_terminal_campaign,
    _reserve_campaign,
    validate_campaign,
)


def _trial(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": SCHEMA,
        "attempt_id": "exp003-seed41-independent",
        "campaign_id": "exp003-primary-campaign",
        "pair_id": "seed-41",
        "pair_order_position": 1,
        "pair_order_rule": "independent_then_shared",
        "pair_order_seed": None,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "tokenizer_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "runtime": "vllm",
        "runtime_version": "0.23.0",
        "fanout": 8,
        "prefix_length": 16_384,
        "suffix_length": 256,
        "seed": 41,
        "baseline_mode": "independent",
        "implementation": "baseline",
        "tracing_level": "minimal",
        "maximum_wall_seconds": 240,
        "initialization_timeout_seconds": 300,
        "cleanup_timeout_seconds": 60,
        "gpu_memory_utilization": 0.8,
        "gpu_sample_interval_seconds": 1.0,
    }
    row.update(changes)
    return row


def _campaign(trials: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "sloforge.branchfabric.modal-campaign-003/v1",
        "campaign_id": "exp003-primary-campaign",
        "maximum_campaign_seconds": 1800,
        "trials": trials,
    }


def test_campaign_is_strict_bounded_and_deterministically_cloned() -> None:
    source = _campaign([_trial()])
    parsed = validate_campaign(source)
    assert parsed == json.loads(json.dumps(source))
    assert parsed is not source
    assert GPU_HOUR_TARGET < GPU_HOUR_HARD_LIMIT


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"fanout": 64}, "fanout"),
        ({"prefix_length": 8192}, "pinned"),
        ({"suffix_length": 16}, "pinned"),
        ({"runtime_version": "0.24.0"}, "pinned"),
        ({"implementation": "fpga"}, "implementation"),
    ],
)
def test_campaign_rejects_unbounded_or_unpinned_trials(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_campaign(_campaign([_trial(**change)]))


def test_campaign_requires_unique_attempts_and_one_to_twenty_four_children() -> None:
    trial = _trial()
    with pytest.raises(ValueError, match="unique"):
        validate_campaign(_campaign([trial, trial]))
    with pytest.raises(ValueError, match=r"1\.\.24"):
        validate_campaign(_campaign([]))


def test_budget_gate_refuses_modal_when_operator_variable_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLOFORGE_GPU_BUDGET_USD", raising=False)
    with pytest.raises(RuntimeError, match="SLOFORGE_GPU_BUDGET_USD"):
        _budget()


def test_source_requests_one_explicit_a100_and_serial_children() -> None:
    source = Path("experiments/branchfabric/modal_metadata_characterization.py").read_text()
    assert 'gpu="A100-80GB"' in source
    assert "max_containers=1" in source
    assert "single_use_containers=True" in source
    assert "for position, trial in enumerate" in source
    assert source.count("_run_campaign_function.spawn(") == 1
    assert "gpu_call.get()" in source


def test_generated_campaign_preserves_required_pair_order_and_trace_controls() -> None:
    payload = build_campaign("baseline", campaign_id="exp003-primary-campaign")
    parsed = validate_campaign(payload)
    paired = [row for row in parsed["trials"] if row["pair_id"] is not None]
    assert [(row["seed"], row["baseline_mode"]) for row in paired] == [
        (41, "independent"),
        (41, "shared_root"),
        (73, "shared_root"),
        (73, "independent"),
        (113, "independent"),
        (113, "shared_root"),
    ]
    assert {row["fanout"] for row in parsed["trials"]} == {1, 8, 16, 32}
    trace_controls = [
        row for row in parsed["trials"] if row["pair_order_rule"] == "instrumentation_control"
    ]
    assert [row["tracing_level"] for row in trace_controls] == [
        "disabled",
        "minimal",
        "full",
    ]
    optimized = validate_campaign(
        build_campaign("optimized", campaign_id="exp003-optimized-campaign")
    )
    assert [
        row["tracing_level"]
        for row in optimized["trials"]
        if row["pair_order_rule"] == "instrumentation_control"
    ] == ["disabled", "minimal", "full"]


def test_campaign_rejects_path_escape_and_unbounded_child_timeout() -> None:
    with pytest.raises(ValueError, match="campaign_id"):
        validate_campaign(
            {
                **_campaign([_trial(campaign_id="../escape")]),
                "campaign_id": "../escape",
            }
        )
    with pytest.raises(ValueError, match="timeout bounds"):
        validate_campaign(_campaign([_trial(maximum_wall_seconds=1000)]))


def test_ledger_reservation_is_durable_attached_and_terminally_reconciled(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "gpu-hours.json"
    ledger, reservation_id = _reserve_campaign(
        ledger_path,
        campaign_id="exp003-baseline-campaign",
        budget_usd=100.0,
    )
    reservation = ledger["modal_in_flight_reservations"][0]
    assert reservation["maximum_gpu_hours"] == CAMPAIGN_RESERVATION_GPU_HOURS == 2400 / 3600
    assert reservation["function_call_id"] is None
    with pytest.raises(RuntimeError, match="unresolved Modal reservation"):
        _reserve_campaign(
            ledger_path,
            campaign_id="exp003-overlap-campaign",
            budget_usd=100.0,
        )

    _attach_function_call(ledger_path, reservation_id=reservation_id, call_id="fc-baseline")
    attached = _load_gpu_ledger(ledger_path)
    assert attached["modal_in_flight_reservations"][0]["function_call_id"] == "fc-baseline"
    finished = _record_terminal_campaign(
        ledger_path,
        reservation_id=reservation_id,
        campaign_id="exp003-baseline-campaign",
        call_id="fc-baseline",
        gpu_seconds=1800.0,
        local_call_elapsed_seconds=1801.0,
        status="succeeded",
    )
    assert finished["consumed_gpu_hours"] == 0.5
    assert finished["modal_in_flight_reservations"] == []
    assert (
        finished["gpu_active_intervals"][0]["cost_usd"]
        > (finished["gpu_active_intervals"][0]["gpu_cost_usd"])
    )

    # A successful 0.5 h campaign leaves enough room for the bounded 2400 s
    # optimized campaign under the 1.25 h hard limit.
    optimized, _ = _reserve_campaign(
        ledger_path,
        campaign_id="exp003-optimized-campaign",
        budget_usd=100.0,
    )
    assert optimized["consumed_gpu_hours"] + CAMPAIGN_RESERVATION_GPU_HOURS < 1.25


def test_ledger_rejects_tampering_and_cumulative_hard_limit(tmp_path: Path) -> None:
    ledger_path = tmp_path / "gpu-hours.json"
    ledger, reservation_id = _reserve_campaign(
        ledger_path,
        campaign_id="exp003-baseline-campaign",
        budget_usd=100.0,
    )
    del ledger
    _attach_function_call(ledger_path, reservation_id=reservation_id, call_id="fc-baseline")
    _record_terminal_campaign(
        ledger_path,
        reservation_id=reservation_id,
        campaign_id="exp003-baseline-campaign",
        call_id="fc-baseline",
        gpu_seconds=2100.0,
        local_call_elapsed_seconds=2101.0,
        status="succeeded",
    )
    _, second_reservation_id = _reserve_campaign(
        ledger_path,
        campaign_id="exp003-second-campaign",
        budget_usd=100.0,
    )
    _attach_function_call(
        ledger_path,
        reservation_id=second_reservation_id,
        call_id="fc-second",
    )
    _record_terminal_campaign(
        ledger_path,
        reservation_id=second_reservation_id,
        campaign_id="exp003-second-campaign",
        call_id="fc-second",
        gpu_seconds=600.0,
        local_call_elapsed_seconds=601.0,
        status="succeeded",
    )
    with pytest.raises(RuntimeError, match="hard limit"):
        _reserve_campaign(
            ledger_path,
            campaign_id="exp003-optimized-campaign",
            budget_usd=100.0,
        )

    payload = json.loads(ledger_path.read_text())
    payload["consumed_gpu_hours"] = 0.1
    ledger_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="disagree with raw intervals"):
        _load_gpu_ledger(ledger_path)
