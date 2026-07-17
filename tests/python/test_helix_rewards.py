from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sloforge.genesis.sandbox import (
    IsolationStatus,
    SandboxResult,
    SandboxTermination,
    detect_capabilities,
)
from sloforge.helix.rewards import (
    DeterministicRewardWorker,
    HiddenCase,
    VerifierCommand,
    compute_evaluator_digest,
)


def _repository(root: Path, *, correct: bool) -> Path:
    root.mkdir()
    expression = (
        "max(0, merchandise - credit) + shipping"
        if correct
        else "max(0, merchandise + shipping - credit)"
    )
    (root / "pricing.py").write_text(
        "def apply_credit(merchandise: int, shipping: int, credit: int) -> int:\n"
        f"    return {expression}\n",
        encoding="utf-8",
    )
    (root / "visible_test.py").write_text(
        "from pricing import apply_credit\n"
        "assert apply_credit(1000, 0, 250) == 750\n"
        "print('visible-pass')\n",
        encoding="utf-8",
    )
    (root / "runner.py").write_text(
        "import sys\n"
        "from pricing import apply_credit\n"
        "print(apply_credit(*(int(item) for item in sys.argv[1:])))\n",
        encoding="utf-8",
    )
    return root


def _verify(source: Path, evidence: Path, *, reward_id: str):
    return DeterministicRewardWorker().verify(
        reward_id=reward_id,
        trajectory_id=f"trajectory-{reward_id}",
        policy_epoch_id="policy-champion-1",
        source=source,
        evidence_directory=evidence,
        commands=(
            VerifierCommand(
                verifier_id="visible-tests",
                argv=("{python}", str(source / "visible_test.py")),
                source_version="visible-v1",
            ),
        ),
        hidden_cases=(
            HiddenCase(
                case_id="shipping-not-discounted",
                runner="runner.py",
                arguments=("1000", "300", "1200"),
                expected_stdout="300",
            ),
        ),
        seed=41,
    )


@pytest.mark.skipif(
    detect_capabilities().network_isolation is IsolationStatus.UNAVAILABLE,
    reason="strict OS sandbox unavailable",
)
def test_actual_hidden_black_box_reward_rejects_plausible_wrong_patch(tmp_path: Path) -> None:
    wrong = _verify(
        _repository(tmp_path / "wrong", correct=False),
        tmp_path / "evidence-wrong",
        reward_id="wrong",
    )
    correct = _verify(
        _repository(tmp_path / "correct", correct=True),
        tmp_path / "evidence-correct",
        reward_id="correct",
    )
    assert [item.passed for item in wrong.components] == [True, False]
    assert [item.passed for item in correct.components] == [True, True]
    assert correct.total_score > wrong.total_score
    assert wrong.immutable_source and correct.immutable_source
    assert not wrong.hidden_expected_values_exposed
    assert "300" not in wrong.verifier_spec_hash


@pytest.mark.skipif(
    detect_capabilities().network_isolation is IsolationStatus.UNAVAILABLE,
    reason="strict OS sandbox unavailable",
)
def test_reward_worker_detects_duplicate_submission_and_escaping_symlink(tmp_path: Path) -> None:
    source = _repository(tmp_path / "source", correct=True)
    worker = DeterministicRewardWorker()
    kwargs = {
        "reward_id": "reward-1",
        "trajectory_id": "trajectory-1",
        "policy_epoch_id": "policy-1",
        "source": source,
        "evidence_directory": tmp_path / "evidence",
        "commands": (
            VerifierCommand(
                verifier_id="visible",
                argv=(sys.executable, str(source / "visible_test.py")),
                source_version="v1",
            ),
        ),
        "hidden_cases": (),
        "seed": 47,
    }
    worker.verify(**kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        worker.verify(**kwargs)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (source / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="escaping symlink"):
        DeterministicRewardWorker().verify(
            **{**kwargs, "reward_id": "reward-2", "trajectory_id": "trajectory-2"}
        )


def test_reward_authority_is_content_bound_and_fail_closed(tmp_path: Path) -> None:
    source = _repository(tmp_path / "source", correct=True)
    commands = (
        VerifierCommand(
            verifier_id="visible",
            argv=("{python}", str(source / "visible_test.py")),
            source_version="v1",
        ),
    )
    evaluator = compute_evaluator_digest(source=source, commands=commands, hidden_cases=())
    worker = DeterministicRewardWorker(trusted_evaluator_digests=())
    with pytest.raises(PermissionError, match="trusted authority"):
        worker.verify(
            reward_id="reward-1",
            trajectory_id="trajectory-1",
            policy_epoch_id="policy-1",
            source=source,
            evidence_directory=tmp_path / "evidence",
            commands=commands,
            hidden_cases=(),
            seed=1,
        )
    (source / "visible_test.py").write_text("print('changed')\n")
    assert compute_evaluator_digest(source=source, commands=commands, hidden_cases=()) != evaluator


def test_reward_excerpts_are_normalized_redacted_and_evidenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    verifier = source / "verify.py"
    verifier.write_text("print('unused')\n")
    command = VerifierCommand(
        verifier_id="visible",
        argv=("{python}", str(verifier)),
        source_version="v1",
    )
    evaluator = compute_evaluator_digest(source=source, commands=(command,), hidden_cases=())
    capabilities = detect_capabilities()
    result = SandboxResult(
        termination=SandboxTermination.SUCCESS,
        return_code=0,
        stdout="\x1b[31mfixture-secret person@example.test\x1b[0m\x00",
        stderr="fixture-secret",
        duration_seconds=0.01,
        output_bytes=64,
        capabilities=capabilities,
        sanitized_environment_names=("LANG",),
        process_group_cleaned=True,
        artifact_output_directory=tmp_path / "evidence",
    )
    worker = DeterministicRewardWorker(trusted_evaluator_digests={evaluator})
    monkeypatch.setattr(worker, "_execute", lambda **_kwargs: result)
    reward = worker.verify(
        reward_id="reward-1",
        trajectory_id="trajectory-1",
        policy_epoch_id="policy-1",
        source=source,
        evidence_directory=tmp_path / "evidence",
        commands=(command,),
        hidden_cases=(),
        seed=1,
        secret_values=("fixture-secret",),
    )
    component = reward.components[0]
    assert reward.evaluator_trusted and reward.evaluator_sha256 == evaluator
    assert "fixture-secret" not in component.stdout_excerpt + component.stderr_excerpt
    assert "person@example.test" not in component.stdout_excerpt
    assert "\x1b" not in component.stdout_excerpt and "\x00" not in component.stdout_excerpt
    assert component.output_redaction_verified
    assert component.stdout_security_evidence_sha256
