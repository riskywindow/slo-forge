"""Deterministic local Genesis synthesis over the initialized baseline genome."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sloforge.genesis.frontend import load_reference_package
from sloforge.genesis.ir import (
    ArtifactDigest,
    BudgetUsage,
    Candidate,
    CandidateFailureState,
    CandidateSuccessState,
    EvidenceReference,
    Extensions,
    InferenceGenome,
    LifecycleEvent,
    SearchBudget,
    canonical_hash,
    load_inference_genome,
    write_canonical,
)
from sloforge.genesis.policy_dsl import check_policy, compile_policy, format_policy, parse_policy
from sloforge.genesis.sandbox import SandboxLimits, SandboxRequest, execute_sandboxed
from sloforge.genesis.search import CandidateDesign

from .fixture import cancellation_fixture_candidates, run_cancellation_cegis
from .models import CegisRunResult


class LocalSynthesisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    seed: int
    baseline_genome_hash: str
    candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    suppressed_candidate_ids: tuple[str, ...]
    accepted_candidate_id: str | None
    accepted_genome_hash: str | None
    cross_layer_accepted: bool
    counterexample_ids: tuple[str, ...]
    constraint_ids: tuple[str, ...]
    runtime_differential_passed: bool
    sandbox_termination: str
    cegis_result_path: str


_UNSAFE_POLICY = """\
policy deadline_batch
input queue_length int 0 32
input slo_slack_ms int 0 1000
output int 0 4
limit 64
return (clamp (if (lt slo_slack_ms 20) 1 (min queue_length 4)) 0 4)
"""

_CORRECTED_POLICY = """\
policy deadline_cancel_batch
input queue_length int 0 32
input slo_slack_ms int 0 1000
input cancellation_pending bool false true
output int 0 4
limit 64
return (clamp (if cancellation_pending 0 (if (lt slo_slack_ms 20) 1 (min queue_length 4))) 0 4)
"""


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite synthesis artifact: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _policy(design: CandidateDesign) -> tuple[str, bytes]:
    safe = any(
        mutation.parameter("cancel_check_before_emit") == "true" for mutation in design.mutations
    )
    source = _CORRECTED_POLICY if safe else _UNSAFE_POLICY
    program = parse_policy(source)
    check_policy(program)
    bytecode = compile_policy(program)
    payload = json.dumps(
        dataclasses.asdict(bytecode),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return format_policy(program), payload


def _mutate_genome(baseline: InferenceGenome, design: CandidateDesign) -> InferenceGenome:
    payload = baseline.model_dump(mode="json")
    safe = any(
        mutation.parameter("cancel_check_before_emit") == "true" for mutation in design.mutations
    )
    payload["genome_id"] = f"{baseline.genome_id}-{design.candidate_id[-12:]}"
    payload["request"]["queue_discipline"] = "earliest_deadline"
    payload["request"]["cancellation_behavior"] = "immediate" if safe else "safe_point"
    payload["serving"]["decode_scheduling"] = "slo_slack"
    for region in ("request", "serving"):
        payload[region]["node"]["extensions"]["sloforge.dev/synthesized-policy"] = {
            "cancel_check_before_emit": safe,
            "candidate_id": design.candidate_id,
            "policy": "deadline_cancel_batch" if safe else "deadline_batch",
        }
    payload["extensions"]["sloforge.dev/synthesis-candidate"] = {
        "candidate_id": design.candidate_id,
        "parent_genome_hash": canonical_hash(baseline),
        "transformation_ids": [item.transformation_id for item in design.mutations],
    }
    return InferenceGenome.model_validate_json(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False), strict=True
    )


def _evidence_reference(path: Path, *, evidence_id: str, claim_id: str) -> EvidenceReference:
    digest = hashlib.sha256(path.read_bytes().rstrip(b"\n")).hexdigest()
    return EvidenceReference(
        evidence_id=evidence_id,
        artifact_uri=str(path.resolve()),
        digest=ArtifactDigest(value=digest),
        claim_ids=(claim_id,),
    )


def _candidate_ir(
    design: CandidateDesign,
    genome: InferenceGenome,
    cegis: CegisRunResult,
    *,
    budget_usd: float,
    runtime_differential_passed: bool,
    counterexample_directory: Path,
) -> Candidate:
    events = [
        LifecycleEvent(
            sequence=0,
            from_state=None,
            to_state=CandidateSuccessState.PROPOSED,
            reason=f"deterministic local {design.proposal_engine} proposal",
        )
    ]

    def transition(state: CandidateSuccessState | CandidateFailureState, reason: str) -> None:
        events.append(
            LifecycleEvent(
                sequence=len(events),
                from_state=events[-1].to_state,
                to_state=state,
                reason=reason,
            )
        )

    if design.candidate_id in cegis.suppressed_candidate_ids:
        transition(
            CandidateFailureState.SUPERSEDED,
            "learned family constraint suppressed repeated unsafe cancellation policy",
        )
    else:
        transition(CandidateSuccessState.STATICALLY_VALID, "typed genome and policy DSL validated")
        transition(CandidateSuccessState.COMPILED, "policy compiled to bounded bytecode")
        if design.candidate_id in cegis.rejected_candidate_ids:
            transition(
                CandidateFailureState.SEMANTIC_REJECTED,
                "independent cancellation verifier produced a minimized counterexample",
            )
        elif runtime_differential_passed:
            transition(
                CandidateSuccessState.REFERENCE_TESTED,
                "generated runtime differential harness passed in the sandbox",
            )
            transition(
                CandidateSuccessState.PROPERTY_TESTED,
                "bounded cancellation protocol verifier accepted the corrected policy",
            )
        else:
            transition(
                CandidateFailureState.SANDBOX_VIOLATION,
                "generated runtime differential harness did not pass strict sandbox execution",
            )
    counterexample_refs = tuple(
        _evidence_reference(
            path,
            evidence_id=f"counterexample:{path.stem}",
            claim_id="cancellation-safety",
        )
        for path in sorted(counterexample_directory.glob("*.json"))
        if design.candidate_id in path.read_text(encoding="utf-8")
    )
    if counterexample_refs:
        final = events[-1]
        events[-1] = final.model_copy(update={"evidence": counterexample_refs})
    state = events[-1].to_state
    return Candidate(
        candidate_id=design.candidate_id,
        seed=design.seed,
        genome_hash=ArtifactDigest(value=canonical_hash(genome)),
        parent_candidate_ids=design.parent_candidate_ids,
        transformation_ids=tuple(item.transformation_id for item in design.mutations),
        state=state,
        lifecycle=tuple(events),
        budget=SearchBudget(
            wall_time_seconds=300.0,
            cpu_time_seconds=300.0,
            gpu_time_seconds=0.0,
            cloud_cost_usd=budget_usd,
            external_synthesis_cost_usd=0.0,
            candidate_count=3,
            compilation_count=3,
            benchmark_count=0,
            verifier_time_seconds=120.0,
        ),
        usage=BudgetUsage(candidate_count=1, compilation_count=1),
        extensions=Extensions(
            root={
                "sloforge.dev/local-synthesis": {
                    "cross_layer": design.cross_layer,
                    "proposal_engine": design.proposal_engine,
                }
            }
        ),
    )


def _run_differential_harness(run_directory: Path, *, seed: int) -> tuple[bool, str]:
    runtime_directory = run_directory / "generated_runtime"
    config = json.loads((runtime_directory / "runtime_config.json").read_text(encoding="utf-8"))
    package = load_reference_package(Path(config["reference_package_root"]))
    samples = package.resolve(package.manifest.quality_contract.final_evaluation_corpus)
    sandbox_output = run_directory / "synthesis/runtime-differential-sandbox"
    repository_python = Path(__file__).resolve().parents[3]
    result = execute_sandboxed(
        SandboxRequest(
            argv=(
                sys.executable,
                "correctness_harness.py",
                "--samples",
                str(samples),
                "--seed",
                str(seed),
                "--timeout-seconds",
                "3",
            ),
            working_directory=runtime_directory,
            read_only_paths=(runtime_directory, package.root, repository_python),
            artifact_output_directory=sandbox_output,
            seed=seed,
            limits=SandboxLimits(
                wall_time_seconds=15.0,
                cpu_time_seconds=10,
                memory_bytes=2 * 1024 * 1024 * 1024,
                process_count=1,
                output_bytes=64 * 1024,
                artifact_bytes=1024 * 1024,
                artifact_entries=16,
                open_files=64,
            ),
        )
    )
    passed = result.succeeded and '"passed": true' in result.stdout
    return passed, result.termination.value


def synthesize_local_run(
    run_directory: Path, *, seed: int, budget_usd: float = 0.0
) -> LocalSynthesisResult:
    """Run the deterministic local CEGIS fixture and persist every candidate."""

    if seed < 0 or budget_usd < 0:
        raise ValueError("seed and budget_usd must be non-negative")
    baseline = load_inference_genome(run_directory / "inference_genome.json")
    synthesis_directory = run_directory / "synthesis"
    cegis_directory = synthesis_directory / "cegis"
    cegis = run_cancellation_cegis(cegis_directory, seed=seed)
    write_canonical(cegis, synthesis_directory / "cegis_result.json")
    designs = cancellation_fixture_candidates(seed)
    runtime_passed, sandbox_termination = _run_differential_harness(run_directory, seed=seed)
    accepted_hash: str | None = None
    accepted_id: str | None = None
    for design in designs:
        candidate_directory = run_directory / "candidates" / design.candidate_id
        genome = _mutate_genome(baseline, design)
        source, bytecode = _policy(design)
        _atomic_write(candidate_directory / "policy.slo", source.encode())
        _atomic_write(candidate_directory / "policy.bytecode.json", bytecode)
        write_canonical(design, candidate_directory / "candidate_design.json")
        write_canonical(genome, candidate_directory / "inference_genome.json")
        candidate = _candidate_ir(
            design,
            genome,
            cegis,
            budget_usd=budget_usd,
            runtime_differential_passed=(
                runtime_passed and design.candidate_id == cegis.accepted_candidate_id
            ),
            counterexample_directory=cegis_directory / "counterexamples",
        )
        write_canonical(candidate, candidate_directory / "candidate.json")
        if candidate.state is CandidateSuccessState.PROPERTY_TESTED:
            accepted_id = candidate.candidate_id
            accepted_hash = candidate.genome_hash.value
    result = LocalSynthesisResult(
        seed=seed,
        baseline_genome_hash=canonical_hash(baseline),
        candidate_ids=tuple(item.candidate_id for item in designs),
        rejected_candidate_ids=cegis.rejected_candidate_ids,
        suppressed_candidate_ids=cegis.suppressed_candidate_ids,
        accepted_candidate_id=accepted_id,
        accepted_genome_hash=accepted_hash,
        cross_layer_accepted=bool(
            accepted_id
            and next(item for item in designs if item.candidate_id == accepted_id).cross_layer
        ),
        counterexample_ids=cegis.counterexample_ids,
        constraint_ids=cegis.constraint_ids,
        runtime_differential_passed=runtime_passed,
        sandbox_termination=sandbox_termination,
        cegis_result_path=str((synthesis_directory / "cegis_result.json").resolve()),
    )
    write_canonical(result, synthesis_directory / "result.json")
    return result


__all__ = ["LocalSynthesisResult", "synthesize_local_run"]
