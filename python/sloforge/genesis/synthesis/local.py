"""Deterministic local Genesis synthesis over the initialized baseline genome."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections import deque
from itertools import product
from pathlib import Path
from typing import Any, Literal

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
    property_passed: bool,
    property_evidence_path: Path | None,
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
            property_evidence = (
                (
                    _evidence_reference(
                        property_evidence_path,
                        evidence_id=f"candidate-property:{design.candidate_id}",
                        claim_id="bounded-policy-properties",
                    ),
                )
                if property_evidence_path is not None
                else ()
            )
            if not property_passed:
                transition(
                    CandidateFailureState.SEMANTIC_REJECTED,
                    "bounded policy property enumeration found a counterexample",
                    property_evidence,
                )
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
            transition(
                CandidateSuccessState.PROPERTY_TESTED,
                "all 66,066 declared integer/boolean policy states satisfied cancellation and output-bound properties",
                property_evidence,
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
    genome_hash = canonical_hash(genome)
    if design.genome_hash.value != genome_hash:
        raise ValueError("candidate design is not bound to the lowered genome")
    synthesis_record = genome.extensions.root.get("sloforge.dev/synthesis-candidate")
    expected_transformations = [item.transformation_id for item in design.mutations]
    if (
        not isinstance(synthesis_record, dict)
        or synthesis_record.get("candidate_id") != design.candidate_id
        or synthesis_record.get("transformation_ids") != expected_transformations
    ):
        raise ValueError("lowered genome does not bind the candidate mutation sequence")

    batching = tuple(
        mutation
        for mutation in design.mutations
        if mutation.family.value == "batching_transformation"
    )
    if len(batching) != 1:
        raise ValueError("candidate runtime requires exactly one lowered batching policy")
    safe = batching[0].parameter("cancel_check_before_emit") == "true"
    expected_policy = "deadline_cancel_batch" if safe else "deadline_batch"
    policy_record = genome.request.node.extensions.root.get("sloforge.dev/synthesized-policy")
    serving_policy_record = genome.serving.node.extensions.root.get(
        "sloforge.dev/synthesized-policy"
    )
    if (
        genome.request.queue_discipline.value != "earliest_deadline"
        or genome.request.cancellation_behavior.value
        != ("immediate" if safe else "safe_point")
        or genome.serving.decode_scheduling.value != "slo_slack"
        or policy_record != serving_policy_record
        or not isinstance(policy_record, dict)
        or policy_record.get("candidate_id") != design.candidate_id
        or policy_record.get("policy") != expected_policy
        or policy_record.get("cancel_check_before_emit") is not safe
    ):
        raise ValueError("lowered Request/Serving genome is inconsistent with policy execution")

    baseline_runtime = run_directory / "generated_runtime"
    runtime_directory = candidate_directory / "generated_runtime"
    policy = (candidate_directory / "policy.bytecode.json").read_bytes()
    policy_source = (candidate_directory / "policy.slo").read_bytes()
    for name in ("runtime.py", "correctness_harness.py"):
        _atomic_write(runtime_directory / name, (baseline_runtime / name).read_bytes())
    _atomic_write(runtime_directory / "policy.bytecode.json", policy)
    _atomic_write(runtime_directory / "policy.slo", policy_source)
    config = json.loads((baseline_runtime / "runtime_config.json").read_text(encoding="utf-8"))
    layouts = {state.layout.value for state in genome.state.states}
    if len(layouts) != 1:
        raise ValueError("generated runtime requires one state allocator layout")
    layout = next(iter(layouts))
    maximum_bytes_per_request = max(
        1, sum(state.maximum_bytes_per_request for state in genome.state.states)
    )
    page_bytes = 64
    layout_record = genome.state.node.extensions.root.get("sloforge.dev/synthesized-state-layout")
    if layout == "paged":
        if not isinstance(layout_record, dict):
            raise ValueError("paged StateGenome is missing its typed allocator page size")
        page_value = layout_record.get("page_bytes")
        if (
            type(page_value) is not int
            or layout_record.get("candidate_id") != design.candidate_id
            or layout_record.get("layout") != "paged"
            or any(
                state.node.extensions.root.get("sloforge.dev/synthesized-state-layout")
                != layout_record
                for state in genome.state.states
            )
        ):
            raise ValueError("paged StateGenome is missing its typed allocator page size")
        page_bytes = page_value
        reserved_per_request = (
            (maximum_bytes_per_request + page_bytes - 1) // page_bytes
        ) * page_bytes
    elif layout == "contiguous":
        reserved_per_request = maximum_bytes_per_request
    else:
        raise ValueError(f"generated runtime does not support state layout {layout!r}")
    queue_depth = int(config["limits"]["maximum_queue_depth"])
    state_allocator = {
        "layout": layout,
        "page_bytes": page_bytes,
        "maximum_bytes_per_request": maximum_bytes_per_request,
        "maximum_total_bytes": reserved_per_request * queue_depth,
    }
    policy_digest = hashlib.sha256(policy).hexdigest()
    runtime_identity = hashlib.sha256(
        json.dumps(
            {
                "base_runtime_id": config["runtime_id"],
                "candidate_id": design.candidate_id,
                "genome_hash": genome_hash,
                "policy_bytecode_sha256": policy_digest,
                "state_allocator": state_allocator,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    config.update(
        {
            "runtime_id": runtime_identity,
            "genome_hash": genome_hash,
            "policy_bytecode_path": "policy.bytecode.json",
            "policy_bytecode_sha256": policy_digest,
            "state_allocator": state_allocator,
        }
    )
    _atomic_write(
        runtime_directory / "runtime_config.json",
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n",
    )
    deployment = json.loads(
        (baseline_runtime / "deployment_manifest.json").read_text(encoding="utf-8")
    )
    deployment.update(
        {
            "runtime_id": runtime_identity,
            "candidate_id": design.candidate_id,
            "genome_hash": genome_hash,
            "policy_bytecode_sha256": policy_digest,
            "state_allocator": state_allocator,
        }
    )
    _atomic_write(
        runtime_directory / "deployment_manifest.json",
        json.dumps(deployment, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n",
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(runtime_directory.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": "1.0.0",
        "candidate_id": design.candidate_id,
        "candidate_genome_hash": genome_hash,
        "policy_bytecode_sha256": policy_digest,
        "state_allocator": state_allocator,
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
    final_evaluation: bool,
) -> tuple[bool, str, Path]:
    runtime_directory, runtime_hashes = _materialize_candidate_runtime(
        run_directory, candidate_directory, design, genome
    )
    config = json.loads((runtime_directory / "runtime_config.json").read_text(encoding="utf-8"))
    runtime_seed = int(config["generation_seed"])
    package = load_reference_package(Path(config["reference_package_root"]))
    samples = package.resolve(
        package.manifest.quality_contract.final_evaluation_corpus
        if final_evaluation
        else package.manifest.quality_contract.search_corpus
    )
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
        "state_allocator": config["state_allocator"],
        "runtime_artifact_hashes": runtime_hashes,
        "seed": seed,
        "candidate_seed": design.seed,
        "runtime_seed": runtime_seed,
        "corpus_role": "final_evaluation" if final_evaluation else "search",
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


def bounded_candidate_modelcheck_document(
    design: CandidateDesign, *, seed: int
) -> dict[str, object]:
    """Independently derive the complete bounded cancellation-check document."""

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
    return {
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


def bounded_candidate_policy_property_document(
    design: CandidateDesign, *, seed: int
) -> dict[str, object]:
    """Exhaustively check executable policy properties over its declared domain."""

    _source, policy, policy_payload = compiled_candidate_policy(design)
    domains: list[tuple[int | bool, ...]] = []
    serialized_domains: dict[str, dict[str, object]] = {}
    for specification in policy.inputs:
        if specification.scalar_type.value == "bool":
            domain: tuple[int | bool, ...] = tuple(
                value
                for value in (False, True)
                if specification.lower <= value <= specification.upper
            )
        elif specification.scalar_type.value == "int":
            domain = tuple(range(int(specification.lower), int(specification.upper) + 1))
        else:
            raise ValueError("bounded property enumeration does not support floating inputs")
        domains.append(domain)
        serialized_domains[specification.name] = {
            "type": specification.scalar_type.value,
            "minimum": specification.lower,
            "maximum": specification.upper,
            "cardinality": len(domain),
        }
    maximum_states = 100_000
    state_count = 1
    for domain in domains:
        state_count *= len(domain)
    if state_count > maximum_states:
        raise ValueError(
            f"policy property domain contains {state_count} states, limit is {maximum_states}"
        )
    checked = 0
    input_names = {specification.name for specification in policy.inputs}
    counterexample: dict[str, object] | None = (
        {
            "assignment": {},
            "observed_output": None,
            "violations": ["required boolean cancellation_pending input is absent"],
        }
        if "cancellation_pending" not in input_names
        else None
    )
    if counterexample is None:
        for values in product(*domains):
            assignment = {
                specification.name: value
                for specification, value in zip(policy.inputs, values, strict=True)
            }
            decision = execute_bytecode(policy, assignment)
            checked += 1
            violations: list[str] = []
            if type(decision) is not int or not 0 <= decision <= 4:
                violations.append("output is outside the declared integer bound [0,4]")
            if assignment.get("cancellation_pending") is True and decision != 0:
                violations.append("cancelled request remains schedulable")
            if violations:
                counterexample = {
                    "assignment": assignment,
                    "observed_output": decision,
                    "violations": violations,
                }
                break
    return {
        "schema_version": "genesis.candidate-policy-property.v1",
        "candidate_id": design.candidate_id,
        "policy_bytecode_sha256": hashlib.sha256(policy_payload).hexdigest(),
        "seed": seed,
        "domains": serialized_domains,
        "maximum_states": maximum_states,
        "states_checked": checked,
        "properties": [
            "typed boolean cancellation_pending input is present",
            "output is an integer in the declared interval [0,4]",
            "cancellation_pending implies output equals zero",
        ],
        "result": "pass" if counterexample is None else "fail",
        "counterexample": counterexample,
        "universal_proof": False,
    }


def _bounded_candidate_policy_properties(
    candidate_directory: Path, design: CandidateDesign, *, seed: int
) -> tuple[bool, Path]:
    evidence = bounded_candidate_policy_property_document(design, seed=seed)
    path = candidate_directory / "evidence/property-result.json"
    _atomic_write(
        path,
        json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
    )
    return evidence["result"] == "pass", path


def _bounded_candidate_modelcheck(
    candidate_directory: Path, design: CandidateDesign, *, seed: int
) -> tuple[bool, Path]:
    """Exhaustively explore and persist the declared cancellation abstraction."""

    evidence = bounded_candidate_modelcheck_document(design, seed=seed)
    path = candidate_directory / "evidence/modelcheck-result.json"
    _atomic_write(
        path,
        json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
    )
    return evidence["result"] == "pass", path


def _run_candidate_simulation(
    run_directory: Path,
    candidate_directory: Path,
    design: CandidateDesign,
    *,
    seed: int,
) -> tuple[bool, Path]:
    """Exercise the compiled policy in a candidate-bound, non-comparative service model."""

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
    _source, policy, policy_payload = compiled_candidate_policy(design)
    policy_digest = hashlib.sha256(policy_payload).hexdigest()
    runtime_manifest_path = (
        candidate_directory / "generated_runtime/candidate_runtime_manifest.json"
    )
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    if (
        runtime_manifest.get("candidate_id") != design.candidate_id
        or runtime_manifest.get("candidate_genome_hash") != design.genome_hash.value
        or runtime_manifest.get("policy_bytecode_sha256") != policy_digest
    ):
        raise ValueError("candidate runtime manifest is not bound to simulation inputs")
    raw: list[dict[str, Any]] = []
    for index, item in enumerate(requests):
        prompt = item.get("text", item.get("prompt_tokens", ""))
        prompt_units = len(prompt) if isinstance(prompt, (str, list)) else 0
        output_units = item.get("maximum_new_tokens", item.get("output_tokens", 1))
        if not isinstance(output_units, int) or isinstance(output_units, bool):
            raise ValueError("simulation workload output bound must be an integer")
        deadline = item.get("deadline_ms")
        if deadline is not None and not isinstance(deadline, (int, float)):
            raise ValueError("simulation workload deadline must be numeric")
        slack_ms = 1000 if deadline is None else max(0, min(1000, int(deadline)))
        available: dict[str, int | bool] = {
            "queue_length": min(32, len(requests) - index),
            "slo_slack_ms": slack_ms,
            "cancellation_pending": False,
        }
        names = {specification.name for specification in policy.inputs}
        decision = execute_bytecode(policy, {name: available[name] for name in names})
        if type(decision) is not int or decision <= 0:
            raise ValueError("compiled candidate policy produced an invalid live-request decision")
        raw.append(
            {
                "ordinal": index,
                "modeled_service_units": prompt_units + output_units,
                "deadline_ms": deadline,
                "policy_batch_limit": decision,
            }
        )
    deadline_declared = all(item["deadline_ms"] is not None for item in raw)
    ordered = (
        sorted(raw, key=lambda item: (float(item["deadline_ms"]), int(item["ordinal"])))
        if deadline_declared and policy.name == "deadline_cancel_batch"
        else raw
    )
    clock = 0
    events: list[dict[str, int]] = []
    for item in ordered:
        clock += item["modeled_service_units"]
        events.append({**item, "completion_units": clock})
    passed = bool(events)
    evidence = {
        "schema_version": "genesis.candidate-simulation.v1",
        "candidate_id": design.candidate_id,
        "candidate_genome_hash": design.genome_hash.value,
        "policy_bytecode_sha256": policy_digest,
        "runtime_manifest_sha256": hashlib.sha256(runtime_manifest_path.read_bytes()).hexdigest(),
        "seed": seed,
        "workload_path": str(workload_path.resolve()),
        "workload_sha256": hashlib.sha256(workload_path.read_bytes()).hexdigest(),
        "queue_policy": policy.name,
        "deadline_order_exercised": deadline_declared,
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
        property_passed = False
        property_evidence_path: Path | None = None
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
                final_evaluation=design.candidate_id == cegis.accepted_candidate_id,
            )
        if candidate_runtime_passed and design.candidate_id == cegis.accepted_candidate_id:
            property_passed, property_evidence_path = _bounded_candidate_policy_properties(
                candidate_directory,
                design,
                seed=seed,
            )
            if property_passed:
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
            property_passed=property_passed,
            property_evidence_path=property_evidence_path,
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


__all__ = [
    "LocalSynthesisResult",
    "bounded_candidate_modelcheck_document",
    "bounded_candidate_policy_property_document",
    "synthesize_local_run",
]
