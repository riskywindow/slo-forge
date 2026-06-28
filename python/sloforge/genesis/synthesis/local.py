"""Deterministic local Genesis synthesis over the initialized baseline genome."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections import deque
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
from sloforge.genesis.policy_dsl import execute_bytecode
from sloforge.genesis.sandbox import SandboxLimits, SandboxRequest, execute_sandboxed
from sloforge.genesis.search import CandidateDesign

from .fixture import (
    cancellation_fixture_candidates,
    compiled_candidate_policy,
    run_cancellation_cegis,
)
from .lowering import lower_candidate
from .models import CegisRunResult, ConstraintDocument


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
    source, _bytecode, payload = compiled_candidate_policy(design)
    return source, payload


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
    runtime_evidence_path: Path | None,
    modelcheck_passed: bool,
    modelcheck_evidence_path: Path | None,
    simulation_passed: bool,
    simulation_evidence_path: Path | None,
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

    def transition(
        state: CandidateSuccessState | CandidateFailureState,
        reason: str,
        evidence: tuple[EvidenceReference, ...] = (),
    ) -> None:
        events.append(
            LifecycleEvent(
                sequence=len(events),
                from_state=events[-1].to_state,
                to_state=state,
                reason=reason,
                evidence=evidence,
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
            runtime_evidence = (
                (
                    _evidence_reference(
                        runtime_evidence_path,
                        evidence_id=f"candidate-runtime:{design.candidate_id}",
                        claim_id="candidate-runtime-differential",
                    ),
                )
                if runtime_evidence_path is not None
                else ()
            )
            transition(
                CandidateSuccessState.REFERENCE_TESTED,
                "candidate-specific generated runtime differential harness passed in the sandbox",
                runtime_evidence,
            )
            transition(
                CandidateSuccessState.PROPERTY_TESTED,
                "bounded cancellation protocol verifier accepted the corrected policy",
            )
            modelcheck_evidence = (
                (
                    _evidence_reference(
                        modelcheck_evidence_path,
                        evidence_id=f"candidate-modelcheck:{design.candidate_id}",
                        claim_id="bounded-cancellation-modelcheck",
                    ),
                )
                if modelcheck_evidence_path is not None
                else ()
            )
            if not modelcheck_passed:
                transition(
                    CandidateFailureState.MODEL_CHECK_REJECTED,
                    "bounded explicit-state model checker rejected the candidate",
                    modelcheck_evidence,
                )
            else:
                transition(
                    CandidateSuccessState.MODEL_CHECKED,
                    "bounded explicit-state cancellation model completed without a counterexample",
                    modelcheck_evidence,
                )
                simulation_evidence = (
                    (
                        _evidence_reference(
                            simulation_evidence_path,
                            evidence_id=f"candidate-simulation:{design.candidate_id}",
                            claim_id="deterministic-candidate-simulation",
                        ),
                    )
                    if simulation_evidence_path is not None
                    else ()
                )
                if simulation_passed:
                    transition(
                        CandidateSuccessState.SIMULATED,
                        "candidate genome completed deterministic digital-twin smoke simulation",
                        simulation_evidence,
                    )
                else:
                    transition(
                        CandidateFailureState.PERFORMANCE_REJECTED,
                        "candidate digital-twin simulation did not complete",
                        simulation_evidence,
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


def _materialize_candidate_runtime(
    run_directory: Path,
    candidate_directory: Path,
    design: CandidateDesign,
    genome: InferenceGenome,
) -> tuple[Path, dict[str, str]]:
    baseline_runtime = run_directory / "generated_runtime"
    runtime_directory = candidate_directory / "generated_runtime"
    policy = (candidate_directory / "policy.bytecode.json").read_bytes()
    policy_source = (candidate_directory / "policy.slo").read_bytes()
    for name in ("runtime.py", "correctness_harness.py", "deployment_manifest.json"):
        _atomic_write(runtime_directory / name, (baseline_runtime / name).read_bytes())
    _atomic_write(runtime_directory / "policy.bytecode.json", policy)
    _atomic_write(runtime_directory / "policy.slo", policy_source)
    config = json.loads((baseline_runtime / "runtime_config.json").read_text(encoding="utf-8"))
    config.update(
        {
            "genome_hash": canonical_hash(genome),
            "policy_bytecode_path": "policy.bytecode.json",
            "policy_bytecode_sha256": hashlib.sha256(policy).hexdigest(),
        }
    )
    _atomic_write(
        runtime_directory / "runtime_config.json",
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n",
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(runtime_directory.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": "1.0.0",
        "candidate_id": design.candidate_id,
        "candidate_genome_hash": canonical_hash(genome),
        "policy_bytecode_sha256": hashlib.sha256(policy).hexdigest(),
        "artifacts": hashes,
    }
    _atomic_write(
        runtime_directory / "candidate_runtime_manifest.json",
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
    )
    return runtime_directory, hashes


def _run_differential_harness(
    run_directory: Path,
    candidate_directory: Path,
    design: CandidateDesign,
    genome: InferenceGenome,
    *,
    seed: int,
) -> tuple[bool, str, Path]:
    runtime_directory, runtime_hashes = _materialize_candidate_runtime(
        run_directory, candidate_directory, design, genome
    )
    config = json.loads((runtime_directory / "runtime_config.json").read_text(encoding="utf-8"))
    runtime_seed = int(config["generation_seed"])
    package = load_reference_package(Path(config["reference_package_root"]))
    samples = package.resolve(package.manifest.quality_contract.final_evaluation_corpus)
    sandbox_output = candidate_directory / "evidence/runtime-differential-sandbox"
    repository_python = Path(__file__).resolve().parents[3]
    result = execute_sandboxed(
        SandboxRequest(
            argv=(
                sys.executable,
                "correctness_harness.py",
                "--samples",
                str(samples),
                "--seed",
                str(runtime_seed),
                "--timeout-seconds",
                "3",
            ),
            working_directory=runtime_directory,
            read_only_paths=(runtime_directory, package.root, repository_python),
            artifact_output_directory=sandbox_output,
            seed=runtime_seed,
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
    try:
        harness = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        harness = None
    cases = harness.get("cases") if isinstance(harness, dict) else None
    passed = bool(
        result.succeeded
        and isinstance(cases, list)
        and cases
        and harness.get("passed") is True
        and all(isinstance(case, dict) and case.get("exact_match") is True for case in cases)
    )
    evidence = {
        "schema_version": "1.0.0",
        "candidate_id": design.candidate_id,
        "candidate_genome_hash": canonical_hash(genome),
        "policy_bytecode_sha256": hashlib.sha256(
            (candidate_directory / "policy.bytecode.json").read_bytes()
        ).hexdigest(),
        "runtime_artifact_hashes": runtime_hashes,
        "seed": seed,
        "corpus_path": str(samples.resolve()),
        "corpus_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
        "sandbox_termination": result.termination.value,
        "passed": passed,
        "cases": cases if isinstance(cases, list) else [],
        "failures": harness.get("failures", []) if isinstance(harness, dict) else [],
    }
    evidence_path = candidate_directory / "evidence/runtime-differential-result.json"
    _atomic_write(
        evidence_path,
        json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
    )
    return passed, result.termination.value, evidence_path


def _bounded_candidate_modelcheck(
    candidate_directory: Path, design: CandidateDesign, *, seed: int
) -> tuple[bool, Path]:
    """Exhaustively explore the declared one-request cancellation abstraction."""

    _source, policy, policy_payload = compiled_candidate_policy(design)
    maximum_depth = 4
    maximum_tokens = 2
    initial = (False, False, 0)
    queue: deque[tuple[tuple[bool, bool, int], int, tuple[str, ...]]] = deque([(initial, 0, ())])
    visited = {(initial, 0)}
    transition_count = 0
    counterexample: tuple[str, ...] | None = None
    actions = ("admit", "cancel", "emit", "schedule")
    while queue:
        (admitted, cancelled, committed), depth, trace = queue.popleft()
        if depth == maximum_depth:
            continue
        for action in actions:
            next_state = (admitted, cancelled, committed)
            if action == "admit" and not admitted:
                next_state = (True, False, 0)
            elif action == "cancel" and admitted and not cancelled:
                next_state = (admitted, True, committed)
            elif action == "emit" and admitted and committed < maximum_tokens:
                names = {item.name for item in policy.inputs}
                values: dict[str, int | bool] = {
                    "queue_length": 1,
                    "slo_slack_ms": 100,
                    "cancellation_pending": cancelled,
                }
                decision = execute_bytecode(policy, {name: values[name] for name in names})
                if cancelled and type(decision) is int and decision > 0:
                    counterexample = (*trace, action)
                    queue.clear()
                    break
                if not cancelled:
                    next_state = (admitted, cancelled, committed + 1)
            transition_count += 1
            item = (next_state, depth + 1)
            if item not in visited:
                visited.add(item)
                queue.append((next_state, depth + 1, (*trace, action)))
        if counterexample is not None:
            break
    passed = counterexample is None
    evidence = {
        "schema_version": "genesis.candidate-modelcheck.v1",
        "candidate_id": design.candidate_id,
        "policy_bytecode_sha256": hashlib.sha256(policy_payload).hexdigest(),
        "model_version": "deadline-cancellation-abstraction.v1",
        "seed": seed,
        "bounds": {
            "max_requests": 1,
            "max_committed_tokens": maximum_tokens,
            "max_depth": maximum_depth,
            "action_count": len(actions),
        },
        "state_count": len(visited),
        "transition_count": transition_count,
        "assumptions": [
            "single-request abstraction",
            "reliable local token delivery",
            "bounded restricted-policy execution",
        ],
        "invariants": ["no token is scheduled for commitment after cancellation"],
        "result": "pass" if passed else "fail",
        "counterexample_trace": None if counterexample is None else list(counterexample),
        "universal_proof": False,
    }
    path = candidate_directory / "evidence/modelcheck-result.json"
    _atomic_write(
        path,
        json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
    )
    return passed, path


def _run_candidate_simulation(
    run_directory: Path,
    candidate_directory: Path,
    design: CandidateDesign,
    *,
    seed: int,
) -> tuple[bool, Path]:
    manifest_path = run_directory / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        workload_path = Path(manifest["workload_contract"]["path"])
    else:
        runtime_config = json.loads(
            (run_directory / "generated_runtime/runtime_config.json").read_text(encoding="utf-8")
        )
        package = load_reference_package(Path(runtime_config["reference_package_root"]))
        workload_path = package.resolve(package.manifest.sample_corpus)
    requests = [
        json.loads(line)
        for line in workload_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = [
        {
            "ordinal": index,
            "modeled_service_units": len(str(item["text"])) + int(item["maximum_new_tokens"]),
            "deadline_rank": int(item["maximum_new_tokens"]),
        }
        for index, item in enumerate(requests)
    ]
    ordered = sorted(raw, key=lambda item: (item["deadline_rank"], item["ordinal"]))
    clock = 0
    events: list[dict[str, int]] = []
    for item in ordered:
        clock += item["modeled_service_units"]
        events.append({**item, "completion_units": clock})
    passed = bool(events)
    evidence = {
        "schema_version": "genesis.candidate-simulation.v1",
        "candidate_id": design.candidate_id,
        "seed": seed,
        "workload_path": str(workload_path.resolve()),
        "workload_sha256": hashlib.sha256(workload_path.read_bytes()).hexdigest(),
        "queue_policy": "earliest_deadline",
        "raw_requests": raw,
        "events": events,
        "result": "pass" if passed else "inconclusive",
        "comparison_permitted": False,
        "hardware_backed": False,
    }
    path = candidate_directory / "evidence/simulation-result.json"
    _atomic_write(
        path,
        json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
    )
    return passed, path


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
    constraint_document = ConstraintDocument.model_validate_json(
        (cegis_directory / "constraints.json").read_bytes(), strict=True
    )
    learned = tuple(item.learned for item in constraint_document.constraints)
    runtime_passed = False
    sandbox_termination = "not_run"
    accepted_hash: str | None = None
    accepted_id: str | None = None
    for design in designs:
        candidate_directory = run_directory / "candidates" / design.candidate_id
        lowered = lower_candidate(
            baseline,
            design,
            learned_constraints=learned,
            counterexample_references=cegis.counterexample_ids,
        )
        design = lowered.design
        genome = lowered.genome
        source, bytecode = _policy(design)
        _atomic_write(candidate_directory / "policy.slo", source.encode())
        _atomic_write(candidate_directory / "policy.bytecode.json", bytecode)
        write_canonical(design, candidate_directory / "candidate_design.json")
        write_canonical(genome, candidate_directory / "inference_genome.json")
        for transformation in lowered.transformations:
            write_canonical(
                transformation,
                candidate_directory
                / "transformations"
                / f"{transformation.transformation_id}.json",
            )
        candidate_runtime_passed = False
        candidate_sandbox_termination = "not_run"
        runtime_evidence_path: Path | None = None
        modelcheck_passed = False
        modelcheck_evidence_path: Path | None = None
        simulation_passed = False
        simulation_evidence_path: Path | None = None
        if design.candidate_id not in cegis.suppressed_candidate_ids:
            (
                candidate_runtime_passed,
                candidate_sandbox_termination,
                runtime_evidence_path,
            ) = _run_differential_harness(
                run_directory,
                candidate_directory,
                design,
                genome,
                seed=seed,
            )
        if candidate_runtime_passed and design.candidate_id == cegis.accepted_candidate_id:
            modelcheck_passed, modelcheck_evidence_path = _bounded_candidate_modelcheck(
                candidate_directory, design, seed=seed
            )
            if modelcheck_passed:
                simulation_passed, simulation_evidence_path = _run_candidate_simulation(
                    run_directory, candidate_directory, design, seed=seed
                )
        candidate = _candidate_ir(
            design,
            genome,
            cegis,
            budget_usd=budget_usd,
            runtime_differential_passed=(
                candidate_runtime_passed and design.candidate_id == cegis.accepted_candidate_id
            ),
            runtime_evidence_path=runtime_evidence_path,
            modelcheck_passed=modelcheck_passed,
            modelcheck_evidence_path=modelcheck_evidence_path,
            simulation_passed=simulation_passed,
            simulation_evidence_path=simulation_evidence_path,
            counterexample_directory=cegis_directory / "counterexamples",
        )
        write_canonical(candidate, candidate_directory / "candidate.json")
        if candidate.state is CandidateSuccessState.SIMULATED:
            accepted_id = candidate.candidate_id
            accepted_hash = candidate.genome_hash.value
            runtime_passed = candidate_runtime_passed
            sandbox_termination = candidate_sandbox_termination
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
