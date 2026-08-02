"""Trusted compilation from zero-day inspection evidence to a baseline genome."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from sloforge.genesis.frontend import InspectionResult, load_reference_package
from sloforge.genesis.frontend.models import DiagnosticSeverity, StateFieldContract, TensorDomain
from sloforge.genesis.ir import (
    AdmissionControl,
    ArtifactDigest,
    CancellationBehavior,
    CollectiveKind,
    CollectiveStep,
    ConsistencyModel,
    DecodeScheduling,
    DistributedGenome,
    EvidenceReference,
    Extensions,
    FallbackBehavior,
    GenomeNodeMetadata,
    HardwarePrecondition,
    HotSwapCategory,
    InferenceGenome,
    KernelBackend,
    KernelGenome,
    KernelSpec,
    LaunchConfiguration,
    ParallelismSpec,
    Placement,
    Precision,
    PrefillPolicy,
    ProofObligation,
    QueueDiscipline,
    RecoveryGenome,
    RecoveryTransition,
    RequestGenome,
    ResourceRequirements,
    RetentionPolicy,
    RoutingPolicy,
    SemanticCategory,
    SemanticContract,
    ServingGenome,
    ServingTopology,
    ShapeDomain,
    SoftwarePrecondition,
    SoftwareRequirement,
    StateGenome,
    StateKind,
    StateLayout,
    StateOwnership,
    StateSpec,
    StreamingSemantics,
    SymbolicDimension,
    TensorGenome,
    TensorOperator,
    TensorValue,
    TransitionPoint,
    Uncertainty,
    VerificationLevel,
    WorkflowEdge,
    WorkflowGenome,
    WorkflowStep,
    WorkflowStepKind,
    canonical_hash,
    write_canonical,
)
from sloforge.genesis.runtime import GeneratedRuntimeBundle, generate_baseline_runtime


class GenomeCompilationError(ValueError):
    """The declared reference package cannot be represented without guessing."""


@dataclass(frozen=True)
class InitializedGenesisRun:
    output_directory: Path
    genome: InferenceGenome
    genome_hash: str
    runtime: GeneratedRuntimeBundle


_PRECISIONS = {member.value: member for member in Precision}
_PRECISION_ALIASES = {
    "double": Precision.FLOAT64,
    "float": Precision.FLOAT32,
    "half": Precision.FLOAT16,
    "long": Precision.INT64,
    "torch.bool": Precision.BOOL,
    "torch.bfloat16": Precision.BFLOAT16,
    "torch.float16": Precision.FLOAT16,
    "torch.float32": Precision.FLOAT32,
    "torch.float64": Precision.FLOAT64,
    "torch.int16": Precision.INT16,
    "torch.int32": Precision.INT32,
    "torch.int64": Precision.INT64,
    "torch.int8": Precision.INT8,
    "torch.uint8": Precision.UINT8,
}
_PRECISION_BYTES = {
    Precision.BOOL: 1,
    Precision.FLOAT64: 8,
    Precision.FLOAT32: 4,
    Precision.BFLOAT16: 2,
    Precision.FLOAT16: 2,
    Precision.FP8: 1,
    Precision.INT64: 8,
    Precision.INT32: 4,
    Precision.INT16: 2,
    Precision.INT8: 1,
    Precision.UINT8: 1,
    Precision.INT4: 1,
}
WorkflowSpec: TypeAlias = tuple[
    str,
    Literal["model", "tool", "verification", "branch", "loop"],
    tuple[str, ...],
    int | None,
    int | None,
]


def _precision(dtype: str) -> Precision:
    normalized = dtype.strip().lower()
    precision = _PRECISIONS.get(normalized) or _PRECISION_ALIASES.get(normalized)
    if precision is None:
        raise GenomeCompilationError(
            f"dtype {dtype!r} is not representable by InferenceGenome v1; "
            "declare a typed extension before synthesis"
        )
    return precision


def _scope(inspection: InspectionResult) -> tuple[str, ...]:
    bounds = [
        f"{dimension.name}={dimension.minimum}..{dimension.maximum} step {dimension.multiple_of}"
        for dimension in inspection.graph.symbolic_dimensions
    ]
    bounds.extend(
        f"{scalar.name}:{scalar.kind}" for scalar in inspection.supported_input_domain.scalars
    )
    return tuple(bounds) or ("declared reference-package input domain",)


def _metadata(
    stable_id: str,
    inspection: InspectionResult,
    *,
    resource: ResourceRequirements,
    hot_swap: HotSwapCategory = HotSwapCategory.REQUEST_BOUNDARY,
    state_invariants: tuple[str, ...] = (),
    diagnostic_obligations: bool = False,
    extensions: Extensions | None = None,
) -> GenomeNodeMetadata:
    scope = _scope(inspection)
    obligations = [
        ProofObligation(
            obligation_id=f"{stable_id}.reference-differential",
            property="outputs and committed streaming tokens match the reference implementation",
            minimum_level=VerificationLevel.DIFFERENTIAL,
            scope="; ".join(scope),
            assumptions=("reference package hashes match the inspection",),
        )
    ]
    if diagnostic_obligations:
        obligations.extend(
            ProofObligation(
                obligation_id=f"{stable_id}.{diagnostic.diagnostic_id}",
                property=diagnostic.proof_obligation or diagnostic.message,
                minimum_level=VerificationLevel.PROPERTY,
                scope="; ".join(scope),
                assumptions=(diagnostic.message,),
            )
            for diagnostic in inspection.diagnostics
            if diagnostic.severity == DiagnosticSeverity.OBLIGATION
        )
    return GenomeNodeMetadata(
        stable_id=stable_id,
        semantic_contract=SemanticContract(
            contract_id=f"contract.{stable_id}",
            category=SemanticCategory.EXACT,
            input_domain=scope,
            output_guarantees=("reference-equivalent behavior within the declared scope",),
            state_invariants=state_invariants,
            numerical_contract="preserve the reference dtype and operation order",
            deterministic=inspection.semantic_contract.deterministic_for_seed,
        ),
        resource_requirements=resource,
        legal_rewrite_rules=("genesis.rules/baseline-exact-v1",),
        proof_obligations=tuple(obligations),
        hardware_preconditions=(HardwarePrecondition(architecture="cpu"),),
        software_preconditions=(
            SoftwarePrecondition(
                requirements=(SoftwareRequirement(package="python", version_range=">=3.11,<3.14"),)
            ),
        ),
        uncertainty=Uncertainty(
            method="static inspection; measurements not yet available",
            confidence=0.0,
            lower=0.0,
            upper=0.0,
        ),
        hot_swap_category=hot_swap,
        extensions=extensions or Extensions(root={}),
    )


def _state_bytes(field: StateFieldContract) -> int:
    elements = 1
    for dimension in field.shape:
        elements *= dimension.maximum
    return elements * _PRECISION_BYTES[_precision(field.dtype)]


def _state_kind(field: StateFieldContract) -> StateKind:
    return {
        "kv": StateKind.KV,
        "recurrent": StateKind.RECURRENT,
        "convolutional": StateKind.CONVOLUTIONAL,
        "speculative": StateKind.SPECULATIVE,
        "custom": StateKind.CUSTOM,
        "workflow": StateKind.WORKFLOW,
    }[field.kind]


def _strides(domain: TensorDomain) -> tuple[str, ...]:
    if domain.allowed_strides:
        return tuple(str(value) for value in domain.allowed_strides[0])
    dimensions = domain.dimensions
    result: list[str] = []
    for index in range(len(dimensions)):
        suffix = [dimension.name for dimension in dimensions[index + 1 :]]
        result.append("*".join(suffix) if suffix else "1")
    return tuple(result)


def compile_inference_genome(
    package_path: Path,
    inspection: InspectionResult,
    *,
    seed: int,
    maximum_queue_depth: int = 32,
    workload_contract_hash: str | None = None,
    hardware_contract_hash: str | None = None,
) -> InferenceGenome:
    """Compile a conservative, complete genome without executing model source."""

    if seed < 0:
        raise GenomeCompilationError("seed must be non-negative")
    if maximum_queue_depth <= 0:
        raise GenomeCompilationError("maximum_queue_depth must be positive")
    package = load_reference_package(package_path)
    if package.package_hash != inspection.package_hash:
        raise GenomeCompilationError("inspection no longer matches the reference package")
    unsupported = [
        diagnostic.message
        for diagnostic in inspection.diagnostics
        if diagnostic.severity == DiagnosticSeverity.UNSUPPORTED
    ]
    if unsupported:
        raise GenomeCompilationError(
            "unsupported reference behavior must be resolved before compilation: "
            + "; ".join(unsupported)
        )

    state_bytes = sum(_state_bytes(field) for field in package.manifest.state_contract.fields)
    resource = ResourceRequirements(
        peak_host_bytes=state_bytes * maximum_queue_depth,
        queue_entries=maximum_queue_depth,
        worker_processes=1,
    )
    state_contract_id = f"state-contract.{inspection.package_hash[:16]}"

    manifest_workflow = package.manifest.workflow
    workflow_specs: tuple[WorkflowSpec, ...]
    if manifest_workflow is None or not manifest_workflow.steps:
        workflow_specs = (("model", "model", (), None, None),)
    else:
        workflow_specs = tuple(
            (
                step.step_id,
                step.kind,
                step.dependencies,
                step.deadline_ms,
                step.expected_latency_ms,
            )
            for step in manifest_workflow.steps
        )
    workflow_kind = {
        "model": WorkflowStepKind.MODEL_INVOCATION,
        "tool": WorkflowStepKind.TOOL_CALL,
        "verification": WorkflowStepKind.VERIFICATION,
        "branch": WorkflowStepKind.BRANCH,
        "loop": WorkflowStepKind.LOOP,
    }
    workflow_steps = tuple(
        WorkflowStep(
            node=_metadata(f"workflow.step.{step_id}", inspection, resource=resource),
            kind=workflow_kind[kind],
            target=package.manifest.package_id if kind == "model" else kind,
            branch_probability=1.0,
            expected_latency_ms=float(latency or 0),
            deadline_ms=float(deadline) if deadline is not None else None,
            maximum_iterations=1 if kind == "loop" else 0,
            cancellation_behavior=CancellationBehavior.SAFE_POINT,
        )
        for step_id, kind, _dependencies, deadline, latency in workflow_specs
    )
    workflow_edges = tuple(
        WorkflowEdge(
            node=_metadata(f"workflow.edge.{dependency}.{step_id}", inspection, resource=resource),
            source_id=f"workflow.step.{dependency}",
            target_id=f"workflow.step.{step_id}",
            condition="dependency-complete",
            probability=1.0,
        )
        for step_id, _kind, dependencies, _deadline, _latency in workflow_specs
        for dependency in dependencies
    )
    workflow_deadlines: list[float] = []
    for _step_id, _kind, _dependencies, deadline, _latency in workflow_specs:
        if deadline is not None:
            workflow_deadlines.append(float(deadline))
    workflow = WorkflowGenome(
        node=_metadata("workflow", inspection, resource=resource),
        steps=workflow_steps,
        edges=workflow_edges,
        entry_step_id=workflow_steps[0].node.stable_id,
        workflow_deadline_ms=max(workflow_deadlines) if workflow_deadlines else None,
    )

    request = RequestGenome(
        node=_metadata(
            "request",
            inspection,
            resource=resource,
            state_invariants=("bounded queue", "committed tokens are emitted once"),
        ),
        admission_control=AdmissionControl.BOUNDED_FIFO,
        maximum_queue_depth=maximum_queue_depth,
        default_priority=0,
        default_deadline_ms=max(workflow_deadlines) if workflow_deadlines else None,
        batching_eligible=bool(inspection.graph.legal_batching_axes),
        routing=RoutingPolicy.LEAST_LOADED,
        queue_discipline=QueueDiscipline.FIFO,
        cancellation_behavior=CancellationBehavior.SAFE_POINT,
        maximum_retries=0,
        streaming_semantics=StreamingSemantics.TOKEN_COMMIT,
        request_classes=("reference",),
        tenant_isolation=True,
        workflow_identity_required=manifest_workflow is not None,
        quality_tiers=("reference-exact",),
        fallback_behavior=FallbackBehavior.REFERENCE_RUNTIME,
    )
    speculative_state = any(
        field.kind == "speculative" for field in package.manifest.state_contract.fields
    )
    serving = ServingGenome(
        node=_metadata("serving", inspection, resource=resource),
        topology=ServingTopology.AGGREGATED,
        prefill_policy=PrefillPolicy.WHOLE_PROMPT,
        incremental_prefill=False,
        prefill_chunk_tokens=inspection.supported_input_domain.maximum_prompt_tokens,
        decode_scheduling=DecodeScheduling.ROUND_ROBIN,
        continuous_batching=bool(inspection.graph.legal_batching_axes),
        maximum_batch_tokens=(
            inspection.supported_input_domain.maximum_prompt_tokens * min(4, maximum_queue_depth)
        ),
        speculative_decoding=False,
        draft_model_id=None,
        verification_policy=(
            "reference-differential-with-speculative-head-disabled"
            if speculative_state
            else "reference-differential"
        ),
        model_cascade=(),
        decode_chunk_tokens=1,
        request_migration=False,
        worker_roles=("prefill_decode",),
    )

    ownership = {
        "request": StateOwnership.REQUEST,
        "session": StateOwnership.SESSION,
        "replicated_read_only": StateOwnership.SHARED_REPLICATED,
    }[package.manifest.state_contract.ownership]
    consistency = (
        ConsistencyModel.READ_ONLY_REPLICATED
        if ownership == StateOwnership.SHARED_REPLICATED
        else ConsistencyModel.EXCLUSIVE
    )
    retention = (
        RetentionPolicy.SESSION
        if ownership == StateOwnership.SESSION
        else RetentionPolicy.REQUEST_LIFETIME
    )
    states = tuple(
        StateSpec(
            node=_metadata(
                f"state.field.{field.field_id}",
                inspection,
                resource=resource,
                state_invariants=(
                    "one declared owner",
                    f"mutation atomicity is {package.manifest.state_contract.mutation_atomicity}",
                ),
                extensions=Extensions(
                    root={
                        "sloforge.dev/reference-state": {
                            "dtype": field.dtype,
                            "kind": field.kind,
                            "mutable": field.mutable,
                            "persistent_across_tokens": field.persistent_across_tokens,
                            "quantization": field.quantization,
                        }
                    }
                ),
            ),
            state_id=field.field_id,
            kind=_state_kind(field),
            cache_key_fields=("request_id", "model_hash"),
            ownership=ownership,
            layout=StateLayout.CONTIGUOUS,
            precision=_precision(field.dtype),
            retention=retention,
            replication_factor=1,
            migratable=package.manifest.state_contract.migration_supported,
            offload_tier="none",
            checkpoint_interval_tokens=0,
            eviction_policy=(
                "release-on-request-boundary"
                if field.reset_at_request_boundary
                else "retain-per-contract"
            ),
            recomputable=field.reset_at_request_boundary,
            consistency=consistency,
            recovery_behavior="discard-uncommitted-state",
            maximum_bytes_per_request=_state_bytes(field),
        )
        for field in package.manifest.state_contract.fields
    )
    state = StateGenome(
        node=_metadata(
            "state",
            inspection,
            resource=resource,
            state_invariants=(
                "no use after release",
                "no partially committed state is visible",
            ),
        ),
        states=states,
        migration_chunk_bytes=4096,
        prefetch_enabled=False,
    )

    distributed = DistributedGenome(
        node=_metadata("distributed", inspection, resource=resource),
        parallelism=ParallelismSpec(
            node=_metadata("distributed.parallelism", inspection, resource=resource)
        ),
        rank_placement=(
            Placement(
                node=_metadata("distributed.rank.0", inspection, resource=resource),
                logical_rank=0,
                host_id="localhost",
                device_id="cpu",
            ),
        ),
        expert_placement=(),
        collective_dag=(
            CollectiveStep(
                node=_metadata("distributed.collective.local", inspection, resource=resource),
                step_id="local-identity",
                kind=CollectiveKind.SEND_RECV,
                dependencies=(),
                algorithm="identity",
                transport="shared_memory",
                ranks=(0,),
                chunk_bytes=1,
            ),
        ),
        prefill_decode_transfer="none",
        failure_domains=("process",),
        recovery_variant_ids=("baseline-restart",),
    )

    symbolic_dimensions = tuple(
        SymbolicDimension(
            node=_metadata(f"tensor.dimension.{dimension.name}", inspection, resource=resource),
            name=dimension.name,
            minimum=dimension.minimum,
            maximum=dimension.maximum,
            divisible_by=dimension.multiple_of,
        )
        for dimension in inspection.graph.symbolic_dimensions
    )
    input_values = tuple(
        TensorValue(
            node=_metadata(
                f"tensor.value.{domain.name}",
                inspection,
                resource=resource,
                extensions=Extensions(
                    root={
                        "sloforge.dev/reference-tensor": {
                            "contiguous": domain.contiguous,
                            "dtype": domain.dtype,
                        }
                    }
                ),
            ),
            value_id=domain.name,
            shape=tuple(dimension.name for dimension in domain.dimensions),
            dtype=_precision(domain.dtype),
            strides=_strides(domain),
            layout="row_major" if domain.contiguous is not False else "declared_strided",
        )
        for domain in inspection.graph.input_tensors
    )
    if not input_values:
        input_values = (
            TensorValue(
                node=_metadata("tensor.value.request", inspection, resource=resource),
                value_id="request",
                shape=(),
                dtype=Precision.INT64,
                strides=(),
                layout="scalar",
            ),
        )
    values = list(input_values)
    operators: list[TensorOperator] = []
    previous_value_id = input_values[0].value_id
    previous_value = input_values[0]
    declared_states = {item.state_id for item in states}
    for index, recovered in enumerate(inspection.graph.operators):
        output_id = f"op-value-{index:05d}"
        state_dependency = next(
            (
                state_id
                for state_id in (*recovered.state_writes, *recovered.state_reads)
                if state_id in declared_states
            ),
            None,
        )
        values.append(
            TensorValue(
                node=_metadata(f"tensor.value.{output_id}", inspection, resource=resource),
                value_id=output_id,
                shape=previous_value.shape,
                dtype=previous_value.dtype,
                strides=previous_value.strides,
                layout=previous_value.layout,
                state_dependency=state_dependency,
            )
        )
        operators.append(
            TensorOperator(
                node=_metadata(
                    f"tensor.operator.{recovered.operator_id}",
                    inspection,
                    resource=resource,
                    extensions=Extensions(
                        root={
                            "sloforge.dev/recovered-operation": {
                                "category": recovered.category,
                                "declared_inputs": list(recovered.inputs),
                                "location": recovered.location.model_dump(mode="json"),
                                "state_reads": list(recovered.state_reads),
                                "state_writes": list(recovered.state_writes),
                            }
                        }
                    ),
                ),
                operator_id=recovered.operator_id,
                operator=recovered.symbol,
                inputs=(previous_value_id,),
                outputs=(output_id,),
                numerical_contract="preserve reference Python evaluation order and dtype",
            )
        )
        previous_value_id = output_id
        previous_value = values[-1]
    tensor = TensorGenome(
        node=_metadata(
            "tensor",
            inspection,
            resource=resource,
            diagnostic_obligations=True,
            extensions=Extensions(
                root={
                    "sloforge.dev/recovered-graph": {
                        "aliases": [
                            alias.model_dump(mode="json") for alias in inspection.graph.aliases
                        ],
                        "control_flow": [
                            flow.model_dump(mode="json") for flow in inspection.graph.control_flow
                        ],
                    }
                }
            ),
        ),
        symbolic_dimensions=symbolic_dimensions,
        values=tuple(values),
        operators=tuple(operators),
        graph_inputs=tuple(value.value_id for value in input_values),
        graph_outputs=(previous_value_id,),
    )

    reference_hash = dict(package.source_hashes)[package.manifest.reference_module]
    kernel = KernelGenome(
        node=_metadata("kernel", inspection, resource=resource),
        kernels=(
            KernelSpec(
                node=_metadata("kernel.reference", inspection, resource=resource),
                kernel_id="reference-runtime",
                source_artifact=EvidenceReference(
                    evidence_id="reference-source",
                    artifact_uri=f"reference://{package.manifest.reference_module}",
                    digest=ArtifactDigest(value=reference_hash),
                    claim_ids=("reference-implementation",),
                ),
                backend=KernelBackend.PYTORCH,
                target_architecture="cpu",
                launch=LaunchConfiguration(block_x=1),
                tile_shape=(1,),
                warp_strategy="none",
                shared_memory_bytes=0,
                register_estimate=0,
                vector_width=1,
                layout_assumptions=tuple(sorted({value.layout for value in values})),
                supported_shapes=ShapeDomain(constraints=_scope(inspection)),
                supported_dtypes=tuple(sorted({value.dtype for value in values}, key=str)),
                deterministic=inspection.semantic_contract.deterministic_for_seed,
                numerical_tolerance=0.0,
                benchmark_evidence=(),
                fallback_kernel_id="reference-runtime",
            ),
        ),
    )
    recovery = RecoveryGenome(
        node=_metadata("recovery", inspection, resource=resource),
        transitions=(
            RecoveryTransition(
                node=_metadata(
                    "recovery.transition.baseline-restart",
                    inspection,
                    resource=resource,
                    hot_swap=HotSwapCategory.WORKER_RESTART,
                ),
                transition_id="baseline-restart",
                safe_point=TransitionPoint.REQUEST_BOUNDARY,
                source_state_contract=state_contract_id,
                target_state_contract=state_contract_id,
                state_conversion_artifact=None,
                state_transfer="recompute",
                active_stream_behavior="drain",
                rollback_transition_id="baseline-restart",
                failure_invariants=(
                    "committed tokens remain committed",
                    "uncommitted state is discarded",
                ),
                operator_action_required=False,
            ),
        ),
        shadow_mode=True,
        canary_mode=True,
        degraded_mode_ids=("reference-only",),
    )

    identity = hashlib.sha256(
        f"{inspection.package_hash}\0{inspection.manifest_hash}\0{seed}".encode()
    ).hexdigest()
    return InferenceGenome(
        genome_id=f"genome-{identity[:24]}",
        seed=seed,
        source_model=ArtifactDigest(value=inspection.package_hash),
        workflow=workflow,
        request=request,
        serving=serving,
        state=state,
        distributed=distributed,
        tensor=tensor,
        kernel=kernel,
        recovery=recovery,
        extensions=Extensions(
            root={
                "sloforge.dev/inspection": {
                    "manifest_hash": inspection.manifest_hash,
                    "package_id": inspection.package_id,
                    "workload_contract_hash": workload_contract_hash,
                    "hardware_contract_hash": hardware_contract_hash,
                }
            }
        ),
    )


def initialize_genesis_run(
    package_path: Path,
    inspection: InspectionResult,
    output_directory: Path,
    *,
    seed: int,
    workload_contract_hash: str | None = None,
    hardware_contract_hash: str | None = None,
) -> InitializedGenesisRun:
    """Persist the genome and generate its conservative runtime as one transaction-like step."""

    genome = compile_inference_genome(
        package_path,
        inspection,
        seed=seed,
        workload_contract_hash=workload_contract_hash,
        hardware_contract_hash=hardware_contract_hash,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    write_canonical(genome, output_directory / "inference_genome.json")
    runtime = generate_baseline_runtime(
        package_path,
        inspection,
        output_directory / "generated_runtime",
        seed=seed,
        genome_hash=canonical_hash(genome),
    )
    return InitializedGenesisRun(
        output_directory=output_directory,
        genome=genome,
        genome_hash=canonical_hash(genome),
        runtime=runtime,
    )


__all__ = [
    "GenomeCompilationError",
    "InitializedGenesisRun",
    "compile_inference_genome",
    "initialize_genesis_run",
]
