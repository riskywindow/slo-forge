from __future__ import annotations

from dataclasses import dataclass

from sloforge.genesis.evaluation import derive_evaluation_rates


@dataclass(frozen=True)
class _Run:
    accepted_candidate_id: str
    accepted_genome_hash: str
    runtime_differential_passed: bool
    capsule_local_evolution_eligible: bool
    capsule_external_production_eligible: bool
    evolution_promoted: bool


def test_rates_require_accepted_runtime_and_keep_local_external_scopes_separate() -> None:
    accepted_local = _Run(
        accepted_candidate_id="candidate-a",
        accepted_genome_hash="a" * 64,
        runtime_differential_passed=True,
        capsule_local_evolution_eligible=True,
        capsule_external_production_eligible=False,
        evolution_promoted=True,
    )
    externally_flagged_but_runtime_failed = _Run(
        accepted_candidate_id="candidate-b",
        accepted_genome_hash="b" * 64,
        runtime_differential_passed=False,
        capsule_local_evolution_eligible=True,
        capsule_external_production_eligible=True,
        evolution_promoted=True,
    )

    rates = derive_evaluation_rates((accepted_local, externally_flagged_but_runtime_failed))

    assert rates.accepted_runtime_success_rate == 0.5
    assert rates.local_capsule_acceptance_rate == 0.5
    assert rates.external_production_eligibility_rate == 0.0
    assert rates.local_evolution_promotion_rate == 0.5
