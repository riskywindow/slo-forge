import type { ReportArtifact } from "./types";

export class ArtifactValidationError extends Error {
  public readonly problems: string[];

  public constructor(problems: string[]) {
    super(`Invalid SLOForge report artifact:\n${problems.join("\n")}`);
    this.name = "ArtifactValidationError";
    this.problems = problems;
  }
}

type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function at(record: UnknownRecord, key: string): unknown {
  return record[key];
}

export function parseReportArtifact(input: unknown): ReportArtifact {
  const problems: string[] = [];
  if (!isRecord(input)) {
    throw new ArtifactValidationError(["$ must be an object"]);
  }

  const requireRecord = (value: unknown, path: string): UnknownRecord => {
    if (!isRecord(value)) {
      problems.push(`${path} must be an object`);
      return {};
    }
    return value;
  };
  const requireArray = (value: unknown, path: string): unknown[] => {
    if (!Array.isArray(value)) {
      problems.push(`${path} must be an array`);
      return [];
    }
    return value;
  };
  const requireString = (value: unknown, path: string): void => {
    if (typeof value !== "string" || value.length === 0) {
      problems.push(`${path} must be a non-empty string`);
    }
  };
  const requireFinite = (value: unknown, path: string): void => {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      problems.push(`${path} must be a finite number`);
    }
  };
  const requireBoolean = (value: unknown, path: string): void => {
    if (typeof value !== "boolean") problems.push(`${path} must be a boolean`);
  };
  const requireNullableFinite = (value: unknown, path: string): void => {
    if (value !== null) requireFinite(value, path);
  };
  const requireNullableString = (value: unknown, path: string): void => {
    if (value !== null) requireString(value, path);
  };
  const requireStringArray = (value: unknown, path: string): void => {
    const items = requireArray(value, path);
    items.forEach((item, index) => requireString(item, `${path}[${index}]`));
  };
  const requireNumericRecord = (value: unknown, path: string): UnknownRecord => {
    const record = requireRecord(value, path);
    for (const [key, item] of Object.entries(record)) {
      requireFinite(item, `${path}.${key}`);
    }
    return record;
  };
  const requireFields = (
    record: UnknownRecord,
    path: string,
    fields: readonly string[],
    validator: (value: unknown, fieldPath: string) => void,
  ): void => {
    for (const field of fields) validator(at(record, field), `${path}.${field}`);
  };

  const plan = requireRecord(at(input, "plan"), "$.plan");
  const metadata = requireRecord(at(plan, "metadata"), "$.plan.metadata");
  const model = requireRecord(at(plan, "model"), "$.plan.model");
  const engine = requireRecord(at(plan, "engine"), "$.plan.engine");
  const hardware = requireRecord(at(plan, "hardware"), "$.plan.hardware");
  const topology = requireRecord(
    at(plan, "replica_topology"),
    "$.plan.replica_topology",
  );
  const routing = requireRecord(at(plan, "routing"), "$.plan.routing");
  const workload = requireRecord(at(plan, "workload"), "$.plan.workload");
  const slo = requireRecord(at(plan, "slo"), "$.plan.slo");
  const predicted = requireRecord(
    at(plan, "predicted_metrics"),
    "$.plan.predicted_metrics",
  );

  requireString(at(plan, "schema_version"), "$.plan.schema_version");
  requireString(at(plan, "api_version"), "$.plan.api_version");
  requireString(at(plan, "kind"), "$.plan.kind");
  requireString(at(metadata, "name"), "$.plan.metadata.name");
  requireString(at(metadata, "uid"), "$.plan.metadata.uid");
  requireString(at(metadata, "created_at"), "$.plan.metadata.created_at");
  requireFinite(at(metadata, "generation"), "$.plan.metadata.generation");
  requireString(at(model, "model_id"), "$.plan.model.model_id");
  requireFields(
    model,
    "$.plan.model",
    ["revision", "minimum_precision"],
    requireString,
  );
  requireFinite(
    at(model, "maximum_sequence_length"),
    "$.plan.model.maximum_sequence_length",
  );
  requireString(at(engine, "runtime"), "$.plan.engine.runtime");
  requireFields(
    engine,
    "$.plan.engine",
    ["dtype", "quantization", "version"],
    requireString,
  );
  requireFields(
    engine,
    "$.plan.engine",
    [
      "maximum_active_sequences",
      "maximum_batched_tokens",
      "tensor_parallelism",
    ],
    requireFinite,
  );
  requireFinite(at(hardware, "hourly_price_usd"), "$.plan.hardware.hourly_price_usd");
  requireFields(hardware, "$.plan.hardware", ["region"], requireString);
  requireFields(
    hardware,
    "$.plan.hardware",
    ["gpu_count", "system_memory_bytes"],
    requireFinite,
  );
  const cpu = requireRecord(at(hardware, "cpu"), "$.plan.hardware.cpu");
  requireFields(cpu, "$.plan.hardware.cpu", ["architecture", "model"], requireString);
  requireFinite(at(cpu, "logical_cores"), "$.plan.hardware.cpu.logical_cores");
  requireArray(at(hardware, "gpus"), "$.plan.hardware.gpus");
  requireFinite(at(topology, "initial_replicas"), "$.plan.replica_topology.initial_replicas");
  requireFields(
    topology,
    "$.plan.replica_topology",
    ["minimum_replicas", "maximum_replicas"],
    requireFinite,
  );
  requireStringArray(at(topology, "regions"), "$.plan.replica_topology.regions");
  requireString(at(routing, "kind"), "$.plan.routing.kind");
  requireArray(at(routing, "targets"), "$.plan.routing.targets").forEach((raw, index) => {
    const target = requireRecord(raw, `$.plan.routing.targets[${index}]`);
    requireString(at(target, "variant"), `$.plan.routing.targets[${index}].variant`);
    requireFinite(at(target, "weight"), `$.plan.routing.targets[${index}].weight`);
  });

  const admission = requireRecord(at(plan, "admission"), "$.plan.admission");
  requireFields(
    admission,
    "$.plan.admission",
    ["maximum_queue_time_ms", "queue_capacity"],
    requireFinite,
  );
  requireBoolean(
    at(admission, "reject_when_predicted_late"),
    "$.plan.admission.reject_when_predicted_late",
  );
  const batching = requireRecord(at(plan, "batching"), "$.plan.batching");
  requireBoolean(at(batching, "dynamic_batching"), "$.plan.batching.dynamic_batching");
  requireFields(
    batching,
    "$.plan.batching",
    [
      "maximum_active_sequences",
      "maximum_batch_delay_ms",
      "maximum_batched_tokens",
    ],
    requireFinite,
  );
  const autoscaling = requireRecord(at(plan, "autoscaling"), "$.plan.autoscaling");
  requireString(at(autoscaling, "mode"), "$.plan.autoscaling.mode");
  requireFields(
    autoscaling,
    "$.plan.autoscaling",
    ["safety_margin", "target_utilization"],
    requireFinite,
  );

  const requestClasses = requireArray(
    at(workload, "request_classes"),
    "$.plan.workload.request_classes",
  );
  requestClasses.forEach((raw, index) => {
    const item = requireRecord(raw, `$.plan.workload.request_classes[${index}]`);
    requireFields(
      item,
      `$.plan.workload.request_classes[${index}]`,
      ["name", "priority"],
      requireString,
    );
    requireFinite(at(item, "weight"), `$.plan.workload.request_classes[${index}].weight`);
    requireNullableFinite(
      at(item, "deadline_ms"),
      `$.plan.workload.request_classes[${index}].deadline_ms`,
    );
  });
  requireFields(workload, "$.plan.workload", ["duration_seconds", "seed"], requireFinite);
  const traceDigest = requireRecord(
    at(workload, "trace_digest"),
    "$.plan.workload.trace_digest",
  );
  requireFields(traceDigest, "$.plan.workload.trace_digest", ["algorithm", "value"], requireString);
  for (const distributionName of ["prompt_tokens", "output_tokens"] as const) {
    const path = `$.plan.workload.${distributionName}`;
    const distribution = requireRecord(at(workload, distributionName), path);
    requireString(at(distribution, "kind"), `${path}.kind`);
    requireNullableFinite(at(distribution, "fixed_value"), `${path}.fixed_value`);
    requireNullableFinite(at(distribution, "minimum"), `${path}.minimum`);
    requireNullableFinite(at(distribution, "maximum"), `${path}.maximum`);
    requireArray(at(distribution, "empirical"), `${path}.empirical`).forEach((raw, index) => {
      const sample = requireRecord(raw, `${path}.empirical[${index}]`);
      requireFinite(at(sample, "value"), `${path}.empirical[${index}].value`);
      requireFinite(at(sample, "weight"), `${path}.empirical[${index}].weight`);
    });
  }
  for (const constraintName of ["ttft", "inter_token_latency"] as const) {
    requireArray(at(slo, constraintName), `$.plan.slo.${constraintName}`).forEach(
      (raw, index) => {
        const constraint = requireRecord(raw, `$.plan.slo.${constraintName}[${index}]`);
        requireFinite(
          at(constraint, "maximum_ms"),
          `$.plan.slo.${constraintName}[${index}].maximum_ms`,
        );
        requireFinite(
          at(constraint, "percentile"),
          `$.plan.slo.${constraintName}[${index}].percentile`,
        );
      },
    );
  }
  requireNullableFinite(at(slo, "minimum_availability"), "$.plan.slo.minimum_availability");
  requireNumericRecord(at(slo, "objective_weights"), "$.plan.slo.objective_weights");

  for (const [name, raw] of Object.entries(predicted)) {
    const metric = requireRecord(raw, `$.plan.predicted_metrics.${name}`);
    for (const field of ["point", "lower", "upper", "confidence", "sample_count"] as const) {
      requireFinite(at(metric, field), `$.plan.predicted_metrics.${name}.${field}`);
    }
    requireString(at(metric, "unit"), `$.plan.predicted_metrics.${name}.unit`);
    requireStringArray(
      at(metric, "measurement_ids"),
      `$.plan.predicted_metrics.${name}.measurement_ids`,
    );
    const lower = at(metric, "lower");
    const point = at(metric, "point");
    const upper = at(metric, "upper");
    if (
      typeof lower === "number" &&
      typeof point === "number" &&
      typeof upper === "number" &&
      (lower > point || point > upper)
    ) {
      problems.push(`$.plan.predicted_metrics.${name} interval must contain its point`);
    }
  }

  const frontier = requireArray(at(input, "pareto_frontier"), "$.pareto_frontier");
  frontier.forEach((raw, index) => {
    const candidate = requireRecord(raw, `$.pareto_frontier[${index}]`);
    const configuration = requireRecord(
      at(candidate, "configuration"),
      `$.pareto_frontier[${index}].configuration`,
    );
    requireString(
      at(configuration, "config_id"),
      `$.pareto_frontier[${index}].configuration.config_id`,
    );
    requireFields(
      configuration,
      `$.pareto_frontier[${index}].configuration`,
      ["backend_candidate_id", "routing_policy"],
      requireString,
    );
    requireFields(
      configuration,
      `$.pareto_frontier[${index}].configuration`,
      ["concurrency", "max_batched_tokens", "replicas", "warm_replicas"],
      requireFinite,
    );
    requireBoolean(
      at(configuration, "chunked_prefill"),
      `$.pareto_frontier[${index}].configuration.chunked_prefill`,
    );
    requireBoolean(at(candidate, "feasible"), `$.pareto_frontier[${index}].feasible`);
    requireString(at(candidate, "fidelity"), `$.pareto_frontier[${index}].fidelity`);
    const measured = at(candidate, "measured");
    if (measured !== null) {
      requireNumericRecord(measured, `$.pareto_frontier[${index}].measured`);
    }
    requireNumericRecord(
      at(candidate, "predicted"),
      `$.pareto_frontier[${index}].predicted`,
    );
    requireNumericRecord(
      at(candidate, "uncertainty"),
      `$.pareto_frontier[${index}].uncertainty`,
    );
    requireNumericRecord(
      at(candidate, "constraint_margins"),
      `$.pareto_frontier[${index}].constraint_margins`,
    );
    requireStringArray(
      at(candidate, "rejection_reasons"),
      `$.pareto_frontier[${index}].rejection_reasons`,
    );
  });

  const metrics = requireRecord(at(input, "metrics"), "$.metrics");
  requireString(at(metrics, "selected_config_id"), "$.metrics.selected_config_id");
  for (const [name, value] of Object.entries(metrics)) {
    if (name !== "selected_config_id") requireFinite(value, `$.metrics.${name}`);
  }

  const controller = requireRecord(at(input, "controller"), "$.controller");
  requireString(at(controller, "schema_version"), "$.controller.schema_version");
  const validateWindow = (raw: unknown, path: string): void => {
    const window = requireRecord(raw, path);
    requireFields(
      window,
      path,
      [
        "arrival_rate_rps",
        "backend_error_rate",
        "concurrency",
        "interactive_fraction",
        "long_context_fraction",
        "observed_p95_ttft_ms",
        "output_p95",
        "prompt_p95",
        "replicas",
        "sample_count",
        "window_end_ms",
        "window_index",
        "window_start_ms",
      ],
      requireFinite,
    );
  };
  requireArray(at(controller, "windows"), "$.controller.windows").forEach((window, index) =>
    validateWindow(window, `$.controller.windows[${index}]`),
  );
  const validateSummary = (raw: unknown, path: string): void => {
    const summary = requireRecord(raw, path);
    requireString(at(summary, "controller"), `${path}.controller`);
    requireFields(
      summary,
      path,
      [
        "action_count",
        "cold_start_exposure",
        "estimated_cost_usd",
        "recovery_windows",
        "replica_oscillations",
        "rollback_count",
        "slo_violations",
      ],
      requireFinite,
    );
  };
  validateSummary(at(controller, "predictive"), "$.controller.predictive");
  validateSummary(at(controller, "reactive"), "$.controller.reactive");
  const validateDecision = (raw: unknown, path: string): void => {
    const decision = requireRecord(raw, path);
    requireFields(
      decision,
      path,
      ["decision_id", "generated_at", "controller_state_before", "controller_state_after"],
      requireString,
    );
    requireBoolean(at(decision, "canary"), `${path}.canary`);
    requireBoolean(at(decision, "rolled_back"), `${path}.rolled_back`);
    requireNullableString(at(decision, "rollback_reason"), `${path}.rollback_reason`);
    requireNullableFinite(
      at(decision, "actual_outcome_p95_ttft_ms"),
      `${path}.actual_outcome_p95_ttft_ms`,
    );
    requireStringArray(at(decision, "safety_checks"), `${path}.safety_checks`);
    validateWindow(at(decision, "observed"), `${path}.observed`);
    const chosen = requireRecord(at(decision, "chosen"), `${path}.chosen`);
    requireFields(
      chosen,
      `${path}.chosen`,
      ["action_type", "routing_policy", "variant"],
      requireString,
    );
    requireFields(
      chosen,
      `${path}.chosen`,
      ["target_concurrency", "target_replicas"],
      requireFinite,
    );
    requireNullableFinite(
      at(chosen, "admission_limit_rps"),
      `${path}.chosen.admission_limit_rps`,
    );
    const forecast = requireRecord(at(decision, "forecast"), `${path}.forecast`);
    requireFields(
      forecast,
      `${path}.forecast`,
      ["arrival_rate_rps", "drift_score", "horizon_windows", "lower_rps", "upper_rps"],
      requireFinite,
    );
  };
  for (const decisionsName of ["predictive_decisions", "reactive_decisions"] as const) {
    requireArray(at(controller, decisionsName), `$.controller.${decisionsName}`).forEach(
      (decision, index) =>
        validateDecision(decision, `$.controller.${decisionsName}[${index}]`),
    );
  }

  const chaos = requireRecord(at(input, "chaos"), "$.chaos");
  requireString(at(chaos, "schema_version"), "$.chaos.schema_version");
  requireString(at(chaos, "scenario_name"), "$.chaos.scenario_name");
  requireArray(at(chaos, "executions"), "$.chaos.executions").forEach((raw, index) => {
    const path = `$.chaos.executions[${index}]`;
    const execution = requireRecord(raw, path);
    requireBoolean(at(execution, "applied"), `${path}.applied`);
    requireNumericRecord(at(execution, "counters_before"), `${path}.counters_before`);
    requireNumericRecord(at(execution, "counters_during"), `${path}.counters_during`);
    const event = requireRecord(at(execution, "event"), `${path}.event`);
    requireFields(event, `${path}.event`, ["fault_id", "fault_type"], requireString);
    requireFields(
      event,
      `${path}.event`,
      ["duration_ms", "magnitude", "probability", "start_ms"],
      requireFinite,
    );
    requireNullableString(at(event, "backend_id"), `${path}.event.backend_id`);
    const diagnosis = requireRecord(at(execution, "diagnosis"), `${path}.diagnosis`);
    requireFields(
      diagnosis,
      `${path}.diagnosis`,
      ["expected_label", "fault_id", "predicted_label"],
      requireString,
    );
    requireFields(
      diagnosis,
      `${path}.diagnosis`,
      ["confidence", "counterfactual_improvement_ms", "diagnosis_latency_ms"],
      requireFinite,
    );
    requireBoolean(at(diagnosis, "correct"), `${path}.diagnosis.correct`);
    requireStringArray(at(diagnosis, "evidence"), `${path}.diagnosis.evidence`);
  });
  requireFinite(at(chaos, "diagnosis_accuracy"), "$.chaos.diagnosis_accuracy");
  requireFields(
    chaos,
    "$.chaos",
    [
      "false_positive_count",
      "false_positive_rate",
      "mean_diagnosis_latency_ms",
      "negative_window_count",
      "seed",
    ],
    requireFinite,
  );
  requireRecord(at(chaos, "confusion_matrix"), "$.chaos.confusion_matrix");
  requireString(at(chaos, "executed_at"), "$.chaos.executed_at");

  if (problems.length > 0) {
    throw new ArtifactValidationError(problems);
  }
  return input as ReportArtifact;
}

export async function fetchReportArtifact(
  source: string,
  signal?: AbortSignal,
): Promise<ReportArtifact> {
  const response = await fetch(source, signal === undefined ? {} : { signal });
  if (!response.ok) {
    throw new Error(`Could not load ${source}: HTTP ${response.status}`);
  }
  return parseReportArtifact(await response.json());
}
