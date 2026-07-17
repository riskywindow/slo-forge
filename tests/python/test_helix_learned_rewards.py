from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from sloforge.helix.rewards.learned import LearnedRewardSpec, LearnedRewardWorker


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, LearnedRewardSpec, LearnedRewardWorker]:
    root = tmp_path / "authority"
    root.mkdir(parents=True)
    runner = root / "runner.py"
    runner.write_text(
        "import json\n"
        "print(json.dumps({'score': 0.75, 'confidence': 0.8, "
        "'components': [{'name': 'judge', 'score': 0.75, 'confidence': 0.8}]}))\n",
        encoding="utf-8",
    )
    model = root / "reward-model.bin"
    model.write_bytes(b"learned-reward-model-v1")
    calibration = root / "calibration.json"
    calibration.write_text('{"method":"temperature","value":1.0}', encoding="utf-8")
    (root / "trajectory.json").write_text('{"trajectory_id":"trajectory-1"}', encoding="utf-8")
    spec = LearnedRewardSpec(
        reward_source_id="judge-v1",
        source_version="1.0.0",
        reward_policy_epoch_id="reward-policy:1",
        runner="runner.py",
        model_artifact="reward-model.bin",
        calibration_artifact="calibration.json",
        known_limitations=("fixture scorer; not a quality gate by itself",),
    )
    worker = LearnedRewardWorker(
        trusted_runner_digests=frozenset({_digest(runner)}),
        trusted_model_digests=frozenset({_digest(model)}),
        trusted_calibration_digests=frozenset({_digest(calibration)}),
    )
    return root, spec, worker


def test_learned_reward_is_sandboxed_and_provenance_complete(tmp_path: Path) -> None:
    root, spec, worker = _fixture(tmp_path)
    evidence = worker.evaluate(
        reward_id="reward-1",
        trajectory_id="trajectory-1",
        behavior_policy_epoch_id="behavior-policy:4",
        authority_root=root,
        input_artifact="trajectory.json",
        spec=spec,
        evidence_directory=tmp_path / "evidence",
        seed=41,
    )
    assert evidence.score == 0.75
    assert evidence.behavior_policy_epoch_id == "behavior-policy:4"
    assert evidence.reward_policy_epoch_id == "reward-policy:1"
    assert evidence.authority_verified
    assert not evidence.deterministic


def test_learned_reward_rejects_tampered_model_and_duplicates(tmp_path: Path) -> None:
    root, spec, worker = _fixture(tmp_path)
    (root / "reward-model.bin").write_bytes(b"tampered")
    with pytest.raises(PermissionError, match="model"):
        worker.evaluate(
            reward_id="reward-tampered",
            trajectory_id="trajectory-1",
            behavior_policy_epoch_id="behavior-policy:4",
            authority_root=root,
            input_artifact="trajectory.json",
            spec=spec,
            evidence_directory=tmp_path / "tampered-evidence",
            seed=41,
        )

    root, spec, worker = _fixture(tmp_path / "duplicate")
    worker.evaluate(
        reward_id="reward-duplicate",
        trajectory_id="trajectory-1",
        behavior_policy_epoch_id="behavior-policy:4",
        authority_root=root,
        input_artifact="trajectory.json",
        spec=spec,
        evidence_directory=tmp_path / "first-evidence",
        seed=41,
    )
    with pytest.raises(ValueError, match="duplicate"):
        worker.evaluate(
            reward_id="reward-duplicate",
            trajectory_id="trajectory-1",
            behavior_policy_epoch_id="behavior-policy:4",
            authority_root=root,
            input_artifact="trajectory.json",
            spec=spec,
            evidence_directory=tmp_path / "second-evidence",
            seed=41,
        )
