from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from sloforge.genesis.ir import (
    Candidate,
    Counterexample,
    Extensions,
    GenesisMigrationError,
    InferenceGenome,
    Transformation,
    canonical_hash,
    canonical_json,
    load_candidate,
    load_counterexample,
    load_inference_genome,
    load_transformation,
    migrate_document,
)

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests/fixtures/genesis"


def _digest(character: str = "a") -> dict[str, str]:
    return {"algorithm": "sha256", "value": character * 64}


def _evidence(evidence_id: str = "evidence-1") -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "artifact_uri": f"artifact://{evidence_id}",
        "digest": _digest("b"),
        "claim_ids": ["claim.semantic"],
    }


def _obligation(obligation_id: str = "proof.semantic") -> dict[str, Any]:
    return {
        "obligation_id": obligation_id,
        "property": "outputs match reference in the declared bounded domain",
        "minimum_level": "level_1_differential",
        "scope": "batch=1..4, sequence=1..64",
        "assumptions": ["IEEE-754 round-to-nearest"],
        "required": True,
    }


def _node(stable_id: str, *, hot_swap: str = "request_boundary") -> dict[str, Any]:
    return {
        "stable_id": stable_id,
        "semantic_contract": {
            "contract_id": f"contract.{stable_id}",
            "category": "exact",
            "input_domain": ["batch=1..4", "sequence=1..64"],
            "output_guarantees": ["reference-equivalent tokens"],
            "state_invariants": ["single owner"],
            "numerical_contract": "float32 exact sampler decisions",
            "deterministic": True,
        },
        "resource_requirements": {
            "peak_device_bytes": 0,
            "peak_host_bytes": 1048576,
            "queue_entries": 8,
            "worker_processes": 1,
            "communication_buffer_bytes": 0,
        },
        "legal_rewrite_rules": ["genesis.rules/conservative-v1"],
        "proof_obligations": [_obligation()],
        "hardware_preconditions": [
            {
                "architecture": "cpu",
                "minimum_device_memory_bytes": 0,
                "required_features": [],
                "forbidden_features": [],
            }
        ],
        "software_preconditions": [
            {"requirements": [{"package": "python", "version_range": ">=3.11,<3.14"}]}
        ],
        "quality_implications": [],
        "expected_performance": [
            {"metric": "latency", "expected_delta": 0.0, "unit": "ms", "model_id": "baseline"}
        ],
        "uncertainty": {"method": "bounded fixture", "confidence": 1.0, "lower": 0.0, "upper": 0.0},
        "hot_swap_category": hot_swap,
        "lineage_references": [],
        "evidence_references": [],
        "frozen": False,
        "extensions": {},
    }


def genome_dict(seed: int = 73129) -> dict[str, Any]:
    step_node = _node("workflow.step.decode")
    state_node = _node("state.kv")
    operator_node = _node("tensor.op.embedding")
    collective_node = _node("distributed.collective.local")
    kernel_node = _node("kernel.reference")
    transition_node = _node("recovery.transition.noop")
    return {
        "schema_version": "1.0.0",
        "api_version": "sloforge.io/genesis/v1",
        "kind": "InferenceGenome",
        "genome_id": "hybrid-decoder-baseline",
        "seed": seed,
        "source_model": _digest("a"),
        "workflow": {
            "node": _node("workflow"),
            "steps": [
                {
                    "node": step_node,
                    "kind": "model_invocation",
                    "target": "hybrid-decoder",
                    "branch_probability": 1.0,
                    "expected_latency_ms": 4.0,
                    "deadline_ms": 100.0,
                    "priority": 0,
                    "maximum_iterations": 0,
                    "model_cascade_targets": [],
                    "expected_future_requests": 0.0,
                    "shared_prefix_group": None,
                    "cancellation_behavior": "safe_point",
                }
            ],
            "edges": [],
            "entry_step_id": "workflow.step.decode",
            "workflow_deadline_ms": 100.0,
        },
        "request": {
            "node": _node("request"),
            "admission_control": "bounded_fifo",
            "maximum_queue_depth": 8,
            "default_priority": 0,
            "default_deadline_ms": 100.0,
            "batching_eligible": True,
            "routing": "least_loaded",
            "queue_discipline": "fifo",
            "cancellation_behavior": "safe_point",
            "maximum_retries": 0,
            "streaming_semantics": "token_commit",
            "request_classes": ["interactive"],
            "tenant_isolation": True,
            "workflow_identity_required": False,
            "quality_tiers": ["exact"],
            "fallback_behavior": "reference_runtime",
        },
        "serving": {
            "node": _node("serving"),
            "topology": "aggregated",
            "prefill_policy": "whole_prompt",
            "incremental_prefill": False,
            "prefill_chunk_tokens": 64,
            "decode_scheduling": "round_robin",
            "continuous_batching": True,
            "maximum_batch_tokens": 256,
            "speculative_decoding": False,
            "draft_model_id": None,
            "verification_policy": "reference-differential",
            "model_cascade": [],
            "decode_chunk_tokens": 1,
            "request_migration": False,
            "worker_roles": ["prefill_decode"],
        },
        "state": {
            "node": _node("state"),
            "states": [
                {
                    "node": state_node,
                    "state_id": "kv",
                    "kind": "kv",
                    "cache_key_fields": ["request_id", "model_hash"],
                    "ownership": "request",
                    "layout": "contiguous",
                    "precision": "float32",
                    "retention": "request_lifetime",
                    "replication_factor": 1,
                    "migratable": False,
                    "offload_tier": "none",
                    "checkpoint_interval_tokens": 0,
                    "eviction_policy": "release-on-complete",
                    "recomputable": True,
                    "consistency": "exclusive",
                    "recovery_behavior": "discard-uncommitted",
                    "maximum_bytes_per_request": 65536,
                }
            ],
            "migration_chunk_bytes": 4096,
            "prefetch_enabled": False,
            "conversion_artifact": None,
        },
        "distributed": {
            "node": _node("distributed"),
            "parallelism": {
                "node": _node("distributed.parallelism"),
                "tensor": 1,
                "pipeline": 1,
                "data": 1,
                "expert": 1,
                "context": 1,
            },
            "rank_placement": [
                {
                    "node": _node("distributed.rank.0"),
                    "logical_rank": 0,
                    "host_id": "localhost",
                    "device_id": "cpu:0",
                    "numa_domain": None,
                    "network_rail": None,
                }
            ],
            "expert_placement": [],
            "collective_dag": [
                {
                    "node": collective_node,
                    "step_id": "local",
                    "kind": "send_recv",
                    "dependencies": [],
                    "algorithm": "identity",
                    "transport": "shared_memory",
                    "ranks": [0],
                    "chunk_bytes": 1,
                    "overlap_group": None,
                }
            ],
            "prefill_decode_transfer": "none",
            "failure_domains": ["process"],
            "recovery_variant_ids": ["restart-reference"],
        },
        "tensor": {
            "node": _node("tensor"),
            "symbolic_dimensions": [
                {
                    "node": _node("tensor.dimension.batch"),
                    "name": "batch",
                    "minimum": 1,
                    "maximum": 4,
                    "divisible_by": 1,
                },
                {
                    "node": _node("tensor.dimension.sequence"),
                    "name": "sequence",
                    "minimum": 1,
                    "maximum": 64,
                    "divisible_by": 1,
                },
            ],
            "values": [
                {
                    "node": _node("tensor.value.tokens"),
                    "value_id": "tokens",
                    "shape": ["batch", "sequence"],
                    "dtype": "int8",
                    "strides": ["sequence", "1"],
                    "layout": "row_major",
                    "alias_group": None,
                    "state_dependency": None,
                },
                {
                    "node": _node("tensor.value.hidden"),
                    "value_id": "hidden",
                    "shape": ["batch", "sequence", "16"],
                    "dtype": "float32",
                    "strides": ["sequence*16", "16", "1"],
                    "layout": "row_major",
                    "alias_group": None,
                    "state_dependency": "kv",
                },
            ],
            "operators": [
                {
                    "node": operator_node,
                    "operator_id": "embedding",
                    "operator": "aten.embedding",
                    "inputs": ["tokens"],
                    "outputs": ["hidden"],
                    "fused_operators": [],
                    "decomposition": [],
                    "quantization": "none",
                    "sparse": False,
                    "numerical_contract": "float32 reference",
                }
            ],
            "graph_inputs": ["tokens"],
            "graph_outputs": ["hidden"],
            "rewrite_history": [],
        },
        "kernel": {
            "node": _node("kernel"),
            "kernels": [
                {
                    "node": kernel_node,
                    "kernel_id": "reference.embedding",
                    "source_artifact": _evidence("kernel-source"),
                    "backend": "pytorch",
                    "target_architecture": "cpu",
                    "launch": {
                        "block_x": 1,
                        "block_y": 1,
                        "block_z": 1,
                        "warps": 1,
                        "pipeline_stages": 1,
                    },
                    "tile_shape": [1],
                    "warp_strategy": "none",
                    "shared_memory_bytes": 0,
                    "register_estimate": 0,
                    "vector_width": 1,
                    "layout_assumptions": ["row_major"],
                    "supported_shapes": {"constraints": ["batch=1..4", "sequence=1..64"]},
                    "supported_dtypes": ["float32"],
                    "deterministic": True,
                    "numerical_tolerance": 0.0,
                    "benchmark_evidence": [],
                    "fallback_kernel_id": "reference.embedding",
                }
            ],
        },
        "recovery": {
            "node": _node("recovery"),
            "transitions": [
                {
                    "node": transition_node,
                    "transition_id": "restart-reference",
                    "safe_point": "request_boundary",
                    "source_state_contract": "contract.state.kv",
                    "target_state_contract": "contract.state.kv",
                    "state_conversion_artifact": None,
                    "state_transfer": "recompute",
                    "active_stream_behavior": "drain",
                    "rollback_transition_id": "restart-reference",
                    "failure_invariants": ["committed tokens remain committed"],
                    "operator_action_required": False,
                }
            ],
            "shadow_mode": True,
            "canary_mode": True,
            "degraded_mode_ids": ["reference-only"],
        },
        "extensions": {"sloforge.io/fixture": "canonical"},
    }


def transformation_dict() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "api_version": "sloforge.io/genesis/v1",
        "kind": "Transformation",
        "transformation_id": "transform.incremental-prefill-v1",
        "family": "scheduler_transformation",
        "source_pattern": {
            "region": "serving",
            "node_ids": ["serving"],
            "structural_constraints": ["prefill_policy=whole_prompt"],
        },
        "target_pattern": {
            "region": "serving",
            "node_ids": ["serving"],
            "structural_constraints": ["prefill_policy=incremental"],
        },
        "semantic_category": "policy",
        "designation": "policy",
        "preconditions": ["prefill is token-prefix pure"],
        "postconditions": ["chunks preserve token order"],
        "expected_quality_cost": [],
        "expected_resource_change": [
            {
                "metric": "temporary_memory",
                "lower": -1024.0,
                "expected": -512.0,
                "upper": 0.0,
                "unit": "bytes",
            }
        ],
        "expected_performance_change": [
            {"metric": "p95_ttft", "lower": -5.0, "expected": -2.0, "upper": 0.0, "unit": "ms"}
        ],
        "affected_regions": ["serving.prefill_policy"],
        "verification_obligations": [_obligation("proof.incremental-prefix")],
        "required_verifier_stages": ["differential", "property", "bounded_model_check"],
        "required_benchmark_stages": ["digital_twin", "end_to_end"],
        "rollback_strategy": "request-boundary policy swap",
        "proposal_source": "deterministic-local-search",
        "parent_transformations": [],
        "learned_constraints": [],
        "counterexample_references": [],
        "lineage_references": [],
        "extensions": {},
    }


def candidate_dict(seed: int = 73129) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "api_version": "sloforge.io/genesis/v1",
        "kind": "Candidate",
        "candidate_id": "candidate-reference",
        "seed": seed,
        "genome_hash": _digest("c"),
        "parent_candidate_ids": [],
        "transformation_ids": [],
        "state": "STATICALLY_VALID",
        "lifecycle": [
            {
                "sequence": 0,
                "from_state": None,
                "to_state": "PROPOSED",
                "reason": "created",
                "evidence": [],
            },
            {
                "sequence": 1,
                "from_state": "PROPOSED",
                "to_state": "STATICALLY_VALID",
                "reason": "trusted IR validation passed",
                "evidence": [_evidence("static-validation")],
            },
        ],
        "budget": {
            "wall_time_seconds": 60.0,
            "cpu_time_seconds": 60.0,
            "gpu_time_seconds": 0.0,
            "cloud_cost_usd": 0.0,
            "external_synthesis_cost_usd": 0.0,
            "candidate_count": 4,
            "compilation_count": 2,
            "benchmark_count": 1,
            "verifier_time_seconds": 30.0,
        },
        "usage": {
            "wall_time_seconds": 1.0,
            "cpu_time_seconds": 1.0,
            "gpu_time_seconds": 0.0,
            "cloud_cost_usd": 0.0,
            "external_synthesis_cost_usd": 0.0,
            "candidate_count": 1,
            "compilation_count": 0,
            "benchmark_count": 0,
            "verifier_time_seconds": 0.5,
        },
        "extensions": {},
    }


def counterexample_dict(seed: int = 73129) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "api_version": "sloforge.io/genesis/v1",
        "kind": "Counterexample",
        "counterexample_id": "cx.noncontiguous-stride",
        "candidate_id": "candidate-fused",
        "transformation_id": "transform.fused-state-update",
        "violated_contract": "contract.tensor.stride-domain",
        "scope": "transformation_family",
        "payload": {
            "kind": "tensor",
            "input": {
                "shape": [1, 2],
                "strides": [4, 2],
                "dtype": "float32",
                "values_hex": "0000803f00000040",
                "non_contiguous": True,
            },
        },
        "reproduction": {
            "executable": "sloforge",
            "arguments": ["redteam", "replay", "--counterexample", "cx.noncontiguous-stride"],
            "timeout_seconds": 30,
            "seed": seed,
        },
        "environment": [
            {"name": "platform", "value": "cpu"},
            {"name": "python", "value": "3.11"},
        ],
        "expected": {"description": "strided values are updated independently", "artifact": None},
        "observed": {
            "description": "candidate reads contiguous storage order",
            "artifact": _evidence("failure-output"),
        },
        "minimized": True,
        "parent_counterexample_id": "cx.noncontiguous-stride-original",
        "lineage_references": [
            {"lineage_id": "constraint.stride-contiguous", "relation": "constrained_by"}
        ],
        "extensions": {},
    }


@pytest.mark.parametrize(
    ("filename", "model", "loader"),
    [
        ("inference-genome-v1.json", InferenceGenome, load_inference_genome),
        ("transformation-v1.json", Transformation, load_transformation),
        ("candidate-v1.json", Candidate, load_candidate),
        ("counterexample-v1.json", Counterexample, load_counterexample),
    ],
)
def test_golden_documents_round_trip_and_validate_schema(
    filename: str,
    model: type[InferenceGenome | Transformation | Candidate | Counterexample],
    loader: Any,
) -> None:
    path = FIXTURES / filename
    document = loader(path)
    reparsed = model.model_validate_json(canonical_json(document))
    assert canonical_json(reparsed) == canonical_json(document)
    schema_path = (
        ROOT
        / {
            InferenceGenome: "schemas/inference_genome/inference-genome-v1.schema.json",
            Transformation: "schemas/transformation/transformation-v1.schema.json",
            Candidate: "schemas/candidate/candidate-v1.schema.json",
            Counterexample: "schemas/counterexample/counterexample-v1.schema.json",
        }[model]
    )
    jsonschema.validate(json.loads(canonical_json(document)), json.loads(schema_path.read_text()))


def test_golden_hashes_are_stable() -> None:
    expected = json.loads((FIXTURES / "canonical-hashes-v1.json").read_text())
    for filename, digest in expected.items():
        assert canonical_hash(json.loads((FIXTURES / filename).read_text())) == digest


@given(st.integers(min_value=0, max_value=2**64 - 1))
def test_seed_round_trip_is_deterministic(seed: int) -> None:
    document = InferenceGenome.model_validate_json(json.dumps(genome_dict(seed)))
    reparsed = InferenceGenome.model_validate_json(canonical_json(document))
    assert reparsed.seed == seed
    assert canonical_hash(reparsed) == canonical_hash(document)


def test_extensions_are_namespace_qualified_and_unknown_core_fields_fail() -> None:
    with pytest.raises(ValidationError):
        Extensions(root={"not-qualified": True})
    source = genome_dict()
    source["untrusted_guess"] = True
    with pytest.raises(ValidationError):
        InferenceGenome.model_validate_json(json.dumps(source))


def test_mutable_node_requires_rewrite_rule_and_proof_obligation() -> None:
    source = genome_dict()
    source["serving"]["node"]["legal_rewrite_rules"] = []
    with pytest.raises(ValidationError, match="legal rewrite"):
        InferenceGenome.model_validate_json(json.dumps(source))
    source = genome_dict()
    source["serving"]["node"]["proof_obligations"] = []
    with pytest.raises(ValidationError, match="proof obligations"):
        InferenceGenome.model_validate_json(json.dumps(source))


def test_candidate_lifecycle_and_budget_are_checked() -> None:
    source = candidate_dict()
    source["lifecycle"][1]["sequence"] = 2
    with pytest.raises(ValidationError, match="contiguous"):
        Candidate.model_validate_json(json.dumps(source))
    source = candidate_dict()
    source["usage"]["candidate_count"] = 5
    with pytest.raises(ValidationError, match="budget"):
        Candidate.model_validate_json(json.dumps(source))
    source = candidate_dict()
    source["lifecycle"][1]["to_state"] = "COMPILED"
    source["state"] = "COMPILED"
    with pytest.raises(ValidationError, match="skipped"):
        Candidate.model_validate_json(json.dumps(source))


def test_approximate_transformation_requires_quality_cost() -> None:
    source = transformation_dict()
    source["designation"] = "approximate_within_quality_budget"
    with pytest.raises(ValidationError, match="quality cost"):
        Transformation.model_validate_json(json.dumps(source))


def test_counterexample_payload_is_discriminated_and_typed() -> None:
    source = counterexample_dict()
    source["payload"]["kind"] = "arbitrary_python"
    with pytest.raises(ValidationError):
        Counterexample.model_validate_json(json.dumps(source))


def test_python_integer_domain_matches_rust_wire_widths() -> None:
    source = genome_dict(seed=(1 << 64) - 1)
    assert InferenceGenome.model_validate_json(json.dumps(source)).seed == (1 << 64) - 1
    source["seed"] = 1 << 64
    with pytest.raises(ValidationError):
        InferenceGenome.model_validate_json(json.dumps(source))
    source = genome_dict()
    source["request"]["default_priority"] = 1 << 63
    with pytest.raises(ValidationError):
        InferenceGenome.model_validate_json(json.dumps(source))


def test_genome_structural_closure_rejects_dangling_and_cyclic_references() -> None:
    source = genome_dict()
    source["workflow"]["edges"] = [
        {
            "node": _node("workflow.edge.self"),
            "source_id": "workflow.step.decode",
            "target_id": "workflow.step.decode",
            "condition": "always",
            "probability": 1.0,
        }
    ]
    with pytest.raises(ValidationError, match="workflow DAG"):
        InferenceGenome.model_validate_json(json.dumps(source))
    source = genome_dict()
    source["distributed"]["collective_dag"][0]["dependencies"] = ["local"]
    with pytest.raises(ValidationError, match="collective DAG"):
        InferenceGenome.model_validate_json(json.dumps(source))

    source = genome_dict()
    source["distributed"]["collective_dag"][0]["ranks"] = [7]
    with pytest.raises(ValidationError, match="declared logical ranks"):
        InferenceGenome.model_validate_json(json.dumps(source))

    source = genome_dict()
    source["tensor"]["values"][1]["state_dependency"] = "missing-state"
    with pytest.raises(ValidationError, match="state_dependency"):
        InferenceGenome.model_validate_json(json.dumps(source))

    source = genome_dict()
    duplicate = copy.deepcopy(source["tensor"]["operators"][0])
    duplicate["operator_id"] = "second-producer"
    duplicate["node"]["stable_id"] = "tensor.op.second-producer"
    source["tensor"]["operators"].append(duplicate)
    with pytest.raises(ValidationError, match="single producer"):
        InferenceGenome.model_validate_json(json.dumps(source))

    source = genome_dict()
    source["kernel"]["kernels"][0]["fallback_kernel_id"] = "missing-kernel"
    with pytest.raises(ValidationError, match="kernel fallback"):
        InferenceGenome.model_validate_json(json.dumps(source))

    source = genome_dict()
    source["recovery"]["transitions"][0]["rollback_transition_id"] = "missing-transition"
    with pytest.raises(ValidationError, match="rollback transition"):
        InferenceGenome.model_validate_json(json.dumps(source))

    source = genome_dict()
    source["serving"]["node"]["stable_id"] = source["request"]["node"]["stable_id"]
    with pytest.raises(ValidationError, match="globally unique"):
        InferenceGenome.model_validate_json(json.dumps(source))


def test_candidate_and_transformation_identity_sets_are_unambiguous() -> None:
    candidate = candidate_dict()
    candidate["parent_candidate_ids"] = [candidate["candidate_id"]]
    with pytest.raises(ValidationError, match="own parent"):
        Candidate.model_validate_json(json.dumps(candidate))

    transformation = transformation_dict()
    transformation["affected_regions"] = ["unknown.path"]
    with pytest.raises(ValidationError, match="typed genome-region"):
        Transformation.model_validate_json(json.dumps(transformation))


def test_shared_python_rust_wire_accept_reject_corpus() -> None:
    cases = json.loads((FIXTURES / "wire-conformance-cases-v1.json").read_text(encoding="utf-8"))
    loaders = {
        "inference-genome-v1.json": load_inference_genome,
        "transformation-v1.json": load_transformation,
        "candidate-v1.json": load_candidate,
        "counterexample-v1.json": load_counterexample,
    }
    for case in cases:
        document = json.loads((FIXTURES / case["document"]).read_text(encoding="utf-8"))
        target = document
        parts = case["pointer"].lstrip("/").split("/")
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        final = int(parts[-1]) if isinstance(target, list) else parts[-1]
        target[final] = case["replacement"]
        if case["accepted"]:
            loaders[case["document"]](document)
        else:
            with pytest.raises(ValueError, match="invalid"):
                loaders[case["document"]](document)


def test_alpha_migration_is_lossless_and_does_not_mutate_input() -> None:
    source = genome_dict()
    source["schema_version"] = "0.1.0"
    source.pop("api_version")
    source["kind"] = "inference_genome"
    for region in (
        "workflow",
        "request",
        "serving",
        "state",
        "distributed",
        "tensor",
        "kernel",
        "recovery",
    ):
        source[f"{region}_genome"] = source.pop(region)
    original = copy.deepcopy(source)
    migrated = migrate_document(source)
    assert source == original
    assert migrated["schema_version"] == "1.0.0"
    assert InferenceGenome.model_validate_json(json.dumps(migrated)).genome_id
    with pytest.raises(GenesisMigrationError):
        migrate_document({"schema_version": "2.0.0", "kind": "InferenceGenome"})


@pytest.mark.parametrize(
    ("builder", "kind", "old_fields", "model"),
    [
        (
            transformation_dict,
            "transformation",
            (("transformation_id", "id"), ("verification_obligations", "verification")),
            Transformation,
        ),
        (candidate_dict, "candidate", (("candidate_id", "id"), ("lifecycle", "events")), Candidate),
        (
            counterexample_dict,
            "counterexample",
            (("counterexample_id", "id"), ("reproduction", "command")),
            Counterexample,
        ),
    ],
)
def test_alpha_migrations_cover_each_document_kind(
    builder: Any,
    kind: str,
    old_fields: tuple[tuple[str, str], ...],
    model: type[Transformation | Candidate | Counterexample],
) -> None:
    source = builder()
    source["schema_version"] = "v1alpha1"
    source.pop("api_version")
    source["kind"] = kind
    for stable, alpha in old_fields:
        source[alpha] = source.pop(stable)
    migrated = migrate_document(source)
    assert model.model_validate_json(json.dumps(migrated))
