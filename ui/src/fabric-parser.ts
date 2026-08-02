import { parseReportArtifact } from "./parser";
import type {
  ArtifactDocument,
  FabricArtifactBundle,
  FabricManifest,
} from "./fabric-types";

type UnknownRecord = Record<string, unknown>;

export class FabricArtifactValidationError extends Error {
  public readonly problems: string[];

  public constructor(problems: string[]) {
    super(`Invalid SLOForge Fabric artifact:\n${problems.join("\n")}`);
    this.name = "FabricArtifactValidationError";
    this.problems = problems;
  }
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateFabricBundle(input: unknown): string[] {
  const problems: string[] = [];
  const record = (value: unknown, path: string): UnknownRecord => {
    if (!isRecord(value)) {
      problems.push(`${path} must be an object`);
      return {};
    }
    return value;
  };
  const array = (value: unknown, path: string): unknown[] => {
    if (!Array.isArray(value)) {
      problems.push(`${path} must be an array`);
      return [];
    }
    return value;
  };
  const string = (value: unknown, path: string): void => {
    if (typeof value !== "string" || value.length === 0) {
      problems.push(`${path} must be a non-empty string`);
    }
  };
  const number = (value: unknown, path: string): void => {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      problems.push(`${path} must be a finite number`);
    }
  };
  const boolean = (value: unknown, path: string): void => {
    if (typeof value !== "boolean") problems.push(`${path} must be a boolean`);
  };
  const nullableString = (value: unknown, path: string): void => {
    if (value !== null) string(value, path);
  };
  const nullableNumber = (value: unknown, path: string): void => {
    if (value !== null) number(value, path);
  };
  const root = record(input, "$");
  string(root["artifact_type"], "$.artifact_type");
  if (root["artifact_type"] !== "sloforge.fabric.ui-bundle/v1")
    problems.push("$.artifact_type must be sloforge.fabric.ui-bundle/v1");

  const manifest = record(root["manifest"], "$.manifest");
  string(manifest["schema_version"], "$.manifest.schema_version");
  for (const field of [
    "baseline_plan_id",
    "diagnosis",
    "physical_plan_id",
    "recovery_final_state",
    "selected_counterfactual",
    "topology_fingerprint",
  ]) string(manifest[field], `$.manifest.${field}`);
  for (const field of ["counterfactuals_evaluated", "diagnosis_confidence", "p95_ttft_slo_ms", "seed"])
    number(manifest[field], `$.manifest.${field}`);
  for (const field of ["synthetic_hardware", "healthy_slo_attained", "degraded_slo_attained", "restored_slo_attained"])
    boolean(manifest[field], `$.manifest.${field}`);
  array(manifest["ground_truth_faults"], "$.manifest.ground_truth_faults").forEach((item, index) =>
    string(item, `$.manifest.ground_truth_faults[${index}]`),
  );
  array(manifest["artifacts"], "$.manifest.artifacts").forEach((item, index) => {
    const artifact = record(item, `$.manifest.artifacts[${index}]`);
    string(artifact["path"], `$.manifest.artifacts[${index}].path`);
    string(artifact["sha256"], `$.manifest.artifacts[${index}].sha256`);
  });
  for (const name of ["healthy", "degraded", "restored"]) {
    const summary = record(manifest[name], `$.manifest.${name}`);
    for (const field of ["makespan_ms", "p95_end_to_end_ms", "p95_ttft_ms", "p99_tpot_ms"])
      number(summary[field], `$.manifest.${name}.${field}`);
  }

  const topology = record(root["topology"], "$.topology");
  for (const field of ["api_version", "kind", "schema_version", "topology_id"])
    string(topology[field], `$.topology.${field}`);
  if (topology["kind"] !== "TopologyGraph") problems.push("$.topology.kind must be TopologyGraph");
  boolean(topology["container_limited"], "$.topology.container_limited");
  array(topology["nodes"], "$.topology.nodes").forEach((item, index) => {
    const node = record(item, `$.topology.nodes[${index}]`);
    for (const field of ["kind", "node_id"])
      string(node[field], `$.topology.nodes[${index}].${field}`);
    if (node["health"] !== undefined) string(node["health"], `$.topology.nodes[${index}].health`);
  });
  array(topology["edges"], "$.topology.edges").forEach((item, index) => {
    const edge = record(item, `$.topology.edges[${index}]`);
    for (const field of ["edge_id", "connection", "source_node_id", "target_node_id", "contention_domain", "sharing_group", "health"])
      string(edge[field], `$.topology.edges[${index}].${field}`);
    nullableNumber(edge["theoretical_bandwidth_gbps"], `$.topology.edges[${index}].theoretical_bandwidth_gbps`);
    number(edge["measurement_confidence"], `$.topology.edges[${index}].measurement_confidence`);
    for (const curveName of ["bandwidth_curve_gbps", "latency_curve_us"]) {
      array(edge[curveName], `$.topology.edges[${index}].${curveName}`).forEach((point, pointIndex) => {
        const curve = record(point, `$.topology.edges[${index}].${curveName}[${pointIndex}]`);
        for (const field of ["message_bytes", "median", "p95", "confidence_low", "confidence_high", "sample_count"])
          number(curve[field], `$.topology.edges[${index}].${curveName}[${pointIndex}].${field}`);
      });
    }
  });

  const profile = record(root["fabric_profile"], "$.fabric_profile");
  for (const field of ["kind", "profile_id", "schema_version"])
    string(profile[field], `$.fabric_profile.${field}`);
  if (profile["kind"] !== "FabricProfile") problems.push("$.fabric_profile.kind must be FabricProfile");
  array(profile["measurements"], "$.fabric_profile.measurements").forEach((item, index) => {
    const measurement = record(item, `$.fabric_profile.measurements[${index}]`);
    for (const field of ["measurement_id", "primitive", "transport"])
      string(measurement[field], `$.fabric_profile.measurements[${index}].${field}`);
    for (const field of ["concurrency", "confidence_high_us", "confidence_low_us", "message_bytes", "rank_count", "summary_median_us", "summary_p95_us", "warmup_count"])
      number(measurement[field], `$.fabric_profile.measurements[${index}].${field}`);
    array(measurement["samples"], `$.fabric_profile.measurements[${index}].samples`).forEach((sample, sampleIndex) => {
      const entry = record(sample, `$.fabric_profile.measurements[${index}].samples[${sampleIndex}]`);
      number(entry["duration_us"], `$.fabric_profile.measurements[${index}].samples[${sampleIndex}].duration_us`);
      boolean(entry["success"], `$.fabric_profile.measurements[${index}].samples[${sampleIndex}].success`);
    });
  });

  const plan = record(root["physical_plan"], "$.physical_plan");
  for (const field of ["api_version", "kind", "schema_version", "plan_id", "compiler_version", "bottleneck_prediction"])
    string(plan[field], `$.physical_plan.${field}`);
  if (plan["kind"] !== "PhysicalExecutionPlan")
    problems.push("$.physical_plan.kind must be PhysicalExecutionPlan");
  const parallelism = record(plan["parallelism"], "$.physical_plan.parallelism");
  for (const field of ["tensor_parallel_degree", "pipeline_parallel_degree", "data_parallel_degree", "expert_parallel_degree", "context_parallel_degree"])
    number(parallelism[field], `$.physical_plan.parallelism.${field}`);
  boolean(parallelism["prefill_decode_disaggregated"], "$.physical_plan.parallelism.prefill_decode_disaggregated");
  array(parallelism["groups"], "$.physical_plan.parallelism.groups").forEach((item, index) => {
    const group = record(item, `$.physical_plan.parallelism.groups[${index}]`);
    string(group["group_id"], `$.physical_plan.parallelism.groups[${index}].group_id`);
    string(group["kind"], `$.physical_plan.parallelism.groups[${index}].kind`);
    array(group["rank_ids"], `$.physical_plan.parallelism.groups[${index}].rank_ids`).forEach((rank, rankIndex) =>
      number(rank, `$.physical_plan.parallelism.groups[${index}].rank_ids[${rankIndex}]`),
    );
  });
  const placement = record(plan["rank_placement"], "$.physical_plan.rank_placement");
  array(placement["bindings"], "$.physical_plan.rank_placement.bindings").forEach((item, index) => {
    const binding = record(item, `$.physical_plan.rank_placement.bindings[${index}]`);
    number(binding["rank_id"], `$.physical_plan.rank_placement.bindings[${index}].rank_id`);
    for (const field of ["host_id", "gpu_id", "numa_domain_id", "process_cpu_affinity", "worker_role", "replica_id", "fault_domain"])
      string(binding[field], `$.physical_plan.rank_placement.bindings[${index}].${field}`);
    nullableString(binding["nic_id"], `$.physical_plan.rank_placement.bindings[${index}].nic_id`);
    nullableString(binding["network_rail_id"], `$.physical_plan.rank_placement.bindings[${index}].network_rail_id`);
  });
  const experts = record(plan["expert_placement"], "$.physical_plan.expert_placement");
  array(experts["assignments"], "$.physical_plan.expert_placement.assignments").forEach((item, index) => {
    const assignment = record(item, `$.physical_plan.expert_placement.assignments[${index}]`);
    string(assignment["expert_id"], `$.physical_plan.expert_placement.assignments[${index}].expert_id`);
    number(assignment["expected_load"], `$.physical_plan.expert_placement.assignments[${index}].expected_load`);
    array(assignment["rank_ids"], `$.physical_plan.expert_placement.assignments[${index}].rank_ids`).forEach((rank, rankIndex) =>
      number(rank, `$.physical_plan.expert_placement.assignments[${index}].rank_ids[${rankIndex}]`),
    );
  });
  const collectives = record(plan["collectives"], "$.physical_plan.collectives");
  array(collectives["operations"], "$.physical_plan.collectives.operations").forEach((item, index) => {
    const operation = record(item, `$.physical_plan.collectives.operations[${index}]`);
    for (const field of ["operation_id", "operation", "algorithm", "transport"])
      string(operation[field], `$.physical_plan.collectives.operations[${index}].${field}`);
    for (const field of ["channel_count", "expected_duration_us"])
      number(operation[field], `$.physical_plan.collectives.operations[${index}].${field}`);
    nullableString(operation["overlap_window_id"], `$.physical_plan.collectives.operations[${index}].overlap_window_id`);
    for (const field of ["participating_ranks", "rank_order"])
      array(operation[field], `$.physical_plan.collectives.operations[${index}].${field}`).forEach((rank, rankIndex) =>
        number(rank, `$.physical_plan.collectives.operations[${index}].${field}[${rankIndex}]`),
      );
    array(operation["rail_ids"], `$.physical_plan.collectives.operations[${index}].rail_ids`).forEach((rail, railIndex) =>
      string(rail, `$.physical_plan.collectives.operations[${index}].rail_ids[${railIndex}]`),
    );
  });
  const overlap = record(plan["communication_overlap"], "$.physical_plan.communication_overlap");
  array(overlap["windows"], "$.physical_plan.communication_overlap.windows").forEach((item, index) => {
    const window = record(item, `$.physical_plan.communication_overlap.windows[${index}]`);
    for (const field of ["window_id", "compute_operation_id", "communication_operation_id", "resource_contention", "fallback_serialization"])
      string(window[field], `$.physical_plan.communication_overlap.windows[${index}].${field}`);
    number(window["expected_overlap_fraction"], `$.physical_plan.communication_overlap.windows[${index}].expected_overlap_fraction`);
  });
  const kv = record(plan["kv_transfer"], "$.physical_plan.kv_transfer");
  array(kv["routes"], "$.physical_plan.kv_transfer.routes").forEach((item, index) => {
    const route = record(item, `$.physical_plan.kv_transfer.routes[${index}]`);
    for (const field of ["route_id", "transport_adapter"])
      string(route[field], `$.physical_plan.kv_transfer.routes[${index}].${field}`);
    for (const field of ["chunk_bytes", "expected_latency_us", "maximum_inflight_chunks"])
      number(route[field], `$.physical_plan.kv_transfer.routes[${index}].${field}`);
    boolean(route["overlap_with_decode"], `$.physical_plan.kv_transfer.routes[${index}].overlap_with_decode`);
    array(route["edge_path"], `$.physical_plan.kv_transfer.routes[${index}].edge_path`).forEach((edge, edgeIndex) =>
      string(edge, `$.physical_plan.kv_transfer.routes[${index}].edge_path[${edgeIndex}]`),
    );
    for (const field of ["producer_rank_ids", "consumer_rank_ids"])
      array(route[field], `$.physical_plan.kv_transfer.routes[${index}].${field}`).forEach((rank, rankIndex) =>
        number(rank, `$.physical_plan.kv_transfer.routes[${index}].${field}[${rankIndex}]`),
      );
  });
  for (const field of ["optimizer_history", "recovery_variants", "rejected_alternatives"])
    array(plan[field], `$.physical_plan.${field}`);
  const metrics = record(plan["predicted_metrics"], "$.physical_plan.predicted_metrics");
  for (const [name, item] of Object.entries(metrics)) {
    const metric = record(item, `$.physical_plan.predicted_metrics.${name}`);
    for (const field of ["estimate", "lower", "upper", "confidence"])
      number(metric[field], `$.physical_plan.predicted_metrics.${name}.${field}`);
    string(metric["unit"], `$.physical_plan.predicted_metrics.${name}.unit`);
  }

  const optimizer = record(root["optimizer"], "$.optimizer");
  string(optimizer["strategy"], "$.optimizer.strategy");
  number(optimizer["solver_time_ms"], "$.optimizer.solver_time_ms");
  number(optimizer["simulator_calls"], "$.optimizer.simulator_calls");
  for (const field of ["all_candidates", "pareto_frontier"])
    array(optimizer[field], `$.optimizer.${field}`).forEach((item, index) => {
      const candidate = record(item, `$.optimizer.${field}[${index}]`);
      string(candidate["candidate_id"], `$.optimizer.${field}[${index}].candidate_id`);
      boolean(candidate["feasible"], `$.optimizer.${field}[${index}].feasible`);
      for (const metric of ["communication_us", "cost_per_million_tokens", "goodput_tokens_per_second", "objective_score", "p95_ttft_ms", "p99_tpot_ms"])
        number(candidate[metric], `$.optimizer.${field}[${index}].${metric}`);
    });

  const diagnosis = record(root["diagnosis"], "$.diagnosis");
  string(diagnosis["diagnosis_id"], "$.diagnosis.diagnosis_id");
  number(diagnosis["confidence"], "$.diagnosis.confidence");
  nullableString(diagnosis["first_divergence_event_id"], "$.diagnosis.first_divergence_event_id");
  nullableNumber(diagnosis["first_divergence_ns"], "$.diagnosis.first_divergence_ns");
  array(diagnosis["hypotheses"], "$.diagnosis.hypotheses").forEach((item, index) => {
    const hypothesis = record(item, `$.diagnosis.hypotheses[${index}]`);
    for (const field of ["hypothesis_id", "kind", "target"])
      string(hypothesis[field], `$.diagnosis.hypotheses[${index}].${field}`);
    number(hypothesis["confidence"], `$.diagnosis.hypotheses[${index}].confidence`);
    nullableString(hypothesis["rejected_reason"], `$.diagnosis.hypotheses[${index}].rejected_reason`);
    array(hypothesis["supporting_evidence"], `$.diagnosis.hypotheses[${index}].supporting_evidence`);
    array(hypothesis["contradicting_evidence"], `$.diagnosis.hypotheses[${index}].contradicting_evidence`);
  });

  const counterfactuals = record(root["counterfactuals"], "$.counterfactuals");
  for (const field of ["diagnosis_id", "schema_version", "selected_scenario_id"])
    string(counterfactuals[field], `$.counterfactuals.${field}`);
  array(counterfactuals["evaluations"], "$.counterfactuals.evaluations").forEach((item, index) => {
    const evaluation = record(item, `$.counterfactuals.evaluations[${index}]`);
    for (const field of ["confidence", "expected_improvement_ms", "healthy_reference_residual_ms", "lower_improvement_ms", "upper_improvement_ms"])
      number(evaluation[field], `$.counterfactuals.evaluations[${index}].${field}`);
    string(evaluation["status"], `$.counterfactuals.evaluations[${index}].status`);
    const scenario = record(evaluation["scenario"], `$.counterfactuals.evaluations[${index}].scenario`);
    for (const field of ["scenario_id", "hypothesis_kind", "rationale"])
      string(scenario[field], `$.counterfactuals.evaluations[${index}].scenario.${field}`);
  });

  const recoveryPlan = record(root["recovery_plan"], "$.recovery_plan");
  string(recoveryPlan["recovery_id"], "$.recovery_plan.recovery_id");
  number(recoveryPlan["confidence"], "$.recovery_plan.confidence");
  boolean(recoveryPlan["external_mutation_authorized"], "$.recovery_plan.external_mutation_authorized");
  array(recoveryPlan["actions"], "$.recovery_plan.actions").forEach((item, index) => {
    const action = record(item, `$.recovery_plan.actions[${index}]`);
    for (const field of ["action_id", "kind", "scope"])
      string(action[field], `$.recovery_plan.actions[${index}].${field}`);
    number(action["order"], `$.recovery_plan.actions[${index}].order`);
    boolean(action["requires_external_mutation"], `$.recovery_plan.actions[${index}].requires_external_mutation`);
  });
  const migration = record(recoveryPlan["traffic_migration"], "$.recovery_plan.traffic_migration");
  for (const field of ["canary_fraction", "minimum_canary_samples", "minimum_shadow_samples", "shadow_fraction"])
    number(migration[field], `$.recovery_plan.traffic_migration.${field}`);
  boolean(migration["preserve_started_streams"], "$.recovery_plan.traffic_migration.preserve_started_streams");

  const execution = record(root["recovery_execution"], "$.recovery_execution");
  for (const field of ["recovery_id", "schema_version", "state"])
    string(execution[field], `$.recovery_execution.${field}`);
  array(execution["audit"], "$.recovery_execution.audit").forEach((item, index) => {
    const audit = record(item, `$.recovery_execution.audit[${index}]`);
    for (const field of ["event", "reason", "state_after", "state_before"])
      string(audit[field], `$.recovery_execution.audit[${index}].${field}`);
    for (const field of ["at_ms", "sequence"])
      number(audit[field], `$.recovery_execution.audit[${index}].${field}`);
  });
  array(execution["action_attempts"], "$.recovery_execution.action_attempts");

  const simulations = record(root["simulations"], "$.simulations");
  for (const name of ["healthy", "degraded", "restored"]) {
    const simulation = record(simulations[name], `$.simulations.${name}`);
    string(simulation["schema_version"], `$.simulations.${name}.schema_version`);
    array(simulation["applied_faults"], `$.simulations.${name}.applied_faults`);
    const simulationMetrics = record(simulation["metrics"], `$.simulations.${name}.metrics`);
    for (const field of ["cost_usd", "makespan_us", "operation_count", "overlap_efficiency", "predicted_lower_us", "predicted_upper_us", "processed_events", "total_transferred_bytes", "total_work_us"])
      number(simulationMetrics[field], `$.simulations.${name}.metrics.${field}`);
    array(simulationMetrics["resources"], `$.simulations.${name}.metrics.resources`);
  }
  array(root["timeline"], "$.timeline").forEach((item, index) => {
    const event = record(item, `$.timeline[${index}]`);
    for (const field of ["detail", "event", "evidence_uri"])
      string(event[field], `$.timeline[${index}].${field}`);
    for (const field of ["at_ms", "sequence"])
      number(event[field], `$.timeline[${index}].${field}`);
  });

  if (manifest["physical_plan_id"] !== plan["plan_id"])
    problems.push("$.manifest.physical_plan_id must match $.physical_plan.plan_id");
  const topologyFingerprint = record(plan["topology_fingerprint"], "$.physical_plan.topology_fingerprint");
  if (manifest["topology_fingerprint"] !== topologyFingerprint["value"])
    problems.push("$.manifest.topology_fingerprint must match the physical plan fingerprint");
  if (counterfactuals["diagnosis_id"] !== diagnosis["diagnosis_id"])
    problems.push("$.counterfactuals.diagnosis_id must match $.diagnosis.diagnosis_id");
  if (execution["recovery_id"] !== recoveryPlan["recovery_id"])
    problems.push("$.recovery_execution.recovery_id must match $.recovery_plan.recovery_id");
  if (manifest["recovery_final_state"] !== execution["state"])
    problems.push("$.manifest.recovery_final_state must match $.recovery_execution.state");
  const selectedScenario = counterfactuals["selected_scenario_id"];
  const selectedExists = Array.isArray(counterfactuals["evaluations"]) &&
    counterfactuals["evaluations"].some((item) => isRecord(item) && item["scenario"] !== undefined &&
      isRecord(item["scenario"]) && item["scenario"]["scenario_id"] === selectedScenario);
  if (!selectedExists) problems.push("$.counterfactuals.selected_scenario_id must reference an evaluation");
  for (const name of ["healthy", "degraded", "restored"]) {
    const summary = record(manifest[name], `$.manifest.${name}`);
    const simulation = record(simulations[name], `$.simulations.${name}`);
    const simulationMetrics = record(simulation["metrics"], `$.simulations.${name}.metrics`);
    const makespan = simulationMetrics["makespan_us"];
    if (typeof makespan === "number" && typeof summary["makespan_ms"] === "number" &&
      Math.abs(makespan / 1000 - summary["makespan_ms"]) > 1e-6)
      problems.push(`$.manifest.${name}.makespan_ms must be derived from $.simulations.${name}.metrics.makespan_us`);
  }
  return problems;
}

export function parseFabricArtifactBundle(input: unknown): FabricArtifactBundle {
  const problems = validateFabricBundle(input);
  if (problems.length > 0) throw new FabricArtifactValidationError(problems);
  return input as FabricArtifactBundle;
}

function parseManifest(input: unknown): FabricManifest {
  const problems: string[] = [];
  if (!isRecord(input)) {
    throw new FabricArtifactValidationError(["$ must be an object"]);
  }
  if (input["schema_version"] !== "sloforge.fabric.demo/v1")
    problems.push("$.schema_version must be sloforge.fabric.demo/v1");
  if (!Array.isArray(input["artifacts"])) {
    problems.push("$.artifacts must be an array");
  } else {
    input["artifacts"].forEach((item, index) => {
      if (!isRecord(item)) {
        problems.push(`$.artifacts[${index}] must be an object`);
        return;
      }
      if (typeof item["path"] !== "string" || item["path"].length === 0)
        problems.push(`$.artifacts[${index}].path must be a non-empty string`);
      if (typeof item["sha256"] !== "string" || !/^[a-f0-9]{64}$/.test(item["sha256"]))
        problems.push(`$.artifacts[${index}].sha256 must be a lowercase SHA-256`);
    });
  }
  if (problems.length > 0) throw new FabricArtifactValidationError(problems);
  return input as unknown as FabricManifest;
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function fetchJsonWithDigest(
  url: URL,
  expectedDigest: string,
  signal: AbortSignal,
): Promise<unknown> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Failed to load ${url.pathname}: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  const actualDigest = await sha256Hex(bytes);
  if (actualDigest !== expectedDigest) {
    throw new Error(`Integrity mismatch for ${url.pathname}: expected ${expectedDigest}, received ${actualDigest}`);
  }
  return JSON.parse(new TextDecoder().decode(bytes)) as unknown;
}

export async function fetchFabricBundleFromManifest(
  manifestUrl: URL,
  input: unknown,
  signal: AbortSignal,
): Promise<FabricArtifactBundle> {
  const manifest = parseManifest(input);
  const digests = new Map(manifest.artifacts.map((artifact) => [artifact.path, artifact.sha256]));
  const paths = {
    counterfactuals: "autopsy/counterfactuals.json",
    diagnosis: "autopsy/diagnosis.json",
    fabric_profile: "fabric-profile.json",
    optimizer: "optimizer.json",
    physical_plan: "physical-plan.json",
    recovery_execution: "recovery/execution.json",
    recovery_plan: "recovery/proposal.json",
    simulation_degraded: "simulations/degraded.json",
    simulation_healthy: "simulations/healthy.json",
    simulation_restored: "simulations/restored.json",
    timeline: "timeline.json",
    topology: "topology.json",
  } as const;
  const base = new URL("./", manifestUrl);
  const entries = Object.entries(paths);
  const loaded = await Promise.all(
    entries.map(async ([key, path]) => {
      const digest = digests.get(path);
      if (digest === undefined) throw new Error(`Manifest does not reference required artifact ${path}`);
      return [key, await fetchJsonWithDigest(new URL(path, base), digest, signal)] as const;
    }),
  );
  const data = Object.fromEntries(loaded) as Record<string, unknown>;
  return parseFabricArtifactBundle({
    artifact_type: "sloforge.fabric.ui-bundle/v1",
    counterfactuals: data["counterfactuals"],
    diagnosis: data["diagnosis"],
    fabric_profile: data["fabric_profile"],
    manifest,
    optimizer: data["optimizer"],
    physical_plan: data["physical_plan"],
    recovery_execution: data["recovery_execution"],
    recovery_plan: data["recovery_plan"],
    simulations: {
      degraded: data["simulation_degraded"],
      healthy: data["simulation_healthy"],
      restored: data["simulation_restored"],
    },
    timeline: data["timeline"],
    topology: data["topology"],
  });
}

export function parseArtifactDocument(input: unknown): ArtifactDocument {
  if (isRecord(input) && input["artifact_type"] === "sloforge.fabric.ui-bundle/v1") {
    return { kind: "fabric", value: parseFabricArtifactBundle(input) };
  }
  return { kind: "logical", value: parseReportArtifact(input) };
}

export async function fetchArtifactDocument(
  source: string,
  signal: AbortSignal,
): Promise<ArtifactDocument> {
  const url = new URL(source, window.location.href);
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Failed to load ${source}: HTTP ${response.status}`);
  const input = (await response.json()) as unknown;
  if (isRecord(input) && input["schema_version"] === "sloforge.fabric.demo/v1") {
    return { kind: "fabric", value: await fetchFabricBundleFromManifest(url, input, signal) };
  }
  return parseArtifactDocument(input);
}
