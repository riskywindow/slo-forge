export type NumericMap = Record<string, number>;

export type IntervalMetric = {
  confidence: number;
  lower: number;
  point: number;
  sample_count: number;
  unit: string;
  upper: number;
  measurement_ids: string[];
};

export type CandidateConfiguration = {
  backend_candidate_id: string;
  chunked_prefill: boolean;
  concurrency: number;
  config_id: string;
  max_batched_tokens: number;
  replicas: number;
  routing_policy: string;
  warm_replicas: number;
};

export type ParetoCandidate = {
  configuration: CandidateConfiguration;
  constraint_margins: NumericMap;
  feasible: boolean;
  fidelity: string;
  measured: NumericMap | null;
  predicted: NumericMap;
  rejection_reasons: string[];
  uncertainty: NumericMap;
};

export type ServingBaselineName =
  | "documented-engine-default"
  | "manual-static"
  | "sloforge-plan";

export type ServingRegime =
  | "steady"
  | "bursty"
  | "short-prompts"
  | "long-prompts"
  | "mixed";

export type ServingMetrics = {
  availability: number;
  cold_start_p95_ms: number;
  cost_per_million_tokens: number;
  goodput_tokens_s: number;
  p95_e2e_ms: number;
  p95_ttft_ms: number;
  p99_itl_ms: number;
  throughput_tokens_s: number;
};

export type ServingBaselineEntry = {
  baseline: ServingBaselineName;
  configuration: CandidateConfiguration;
  feasible: boolean;
  fidelity: "calibrated_prediction";
  metrics: ServingMetrics;
  regime: ServingRegime;
  rejection_reasons: string[];
  uncertainty: ServingMetrics;
};

export type ServingBaselines = {
  measurement_claim: string;
  results: ServingBaselineEntry[];
  schema_version: "sloforge.serving-baselines/v1";
  source_profile_id: string;
};

export type Distribution = {
  empirical: { value: number; weight: number }[];
  fixed_value: number | null;
  kind: string;
  maximum: number | null;
  minimum: number | null;
};

export type WorkloadClass = {
  deadline_ms: number | null;
  name: string;
  priority: string;
  weight: number;
};

export type ControllerWindow = {
  arrival_rate_rps: number;
  backend_error_rate: number;
  concurrency: number;
  interactive_fraction: number;
  long_context_fraction: number;
  observed_p95_ttft_ms: number;
  output_p95: number;
  prompt_p95: number;
  replicas: number;
  sample_count: number;
  window_end_ms: number;
  window_index: number;
  window_start_ms: number;
};

export type ControllerAction = {
  action_type: string;
  admission_limit_rps: number | null;
  routing_policy: string;
  target_concurrency: number;
  target_replicas: number;
  variant: string;
};

export type ControllerDecision = {
  actual_outcome_p95_ttft_ms: number | null;
  canary: boolean;
  chosen: ControllerAction;
  controller_state_after: string;
  controller_state_before: string;
  decision_id: string;
  forecast: {
    arrival_rate_rps: number;
    drift_score: number;
    horizon_windows: number;
    lower_rps: number;
    upper_rps: number;
  };
  generated_at: string;
  observed: ControllerWindow;
  rollback_reason: string | null;
  rolled_back: boolean;
  safety_checks: string[];
};

export type ControllerSummary = {
  action_count: number;
  cold_start_exposure: number;
  controller: string;
  estimated_cost_usd: number;
  recovery_windows: number;
  replica_oscillations: number;
  rollback_count: number;
  slo_violations: number;
};

export type ChaosExecution = {
  applied: boolean;
  counters_before: NumericMap;
  counters_during: NumericMap;
  diagnosis: {
    confidence: number;
    correct: boolean;
    counterfactual_improvement_ms: number;
    diagnosis_latency_ms: number;
    evidence: string[];
    expected_label: string;
    fault_id: string;
    predicted_label: string;
  };
  event: {
    backend_id: string | null;
    duration_ms: number;
    fault_id: string;
    fault_type: string;
    magnitude: number;
    probability: number;
    start_ms: number;
  };
};

export type ReportArtifact = {
  chaos: {
    diagnosis_accuracy: number;
    executed_at: string;
    executions: ChaosExecution[];
    false_positive_count: number;
    false_positive_rate: number;
    negative_window_count: number;
    confusion_matrix: Record<string, Record<string, number>>;
    mean_diagnosis_latency_ms: number;
    scenario_name: string;
    schema_version: string;
    seed: number;
  };
  controller: {
    predictive: ControllerSummary;
    predictive_decisions: ControllerDecision[];
    reactive: ControllerSummary;
    reactive_decisions: ControllerDecision[];
    schema_version: string;
    windows: ControllerWindow[];
  };
  metrics: Record<string, number | string> & { selected_config_id: string };
  pareto_frontier: ParetoCandidate[];
  serving_baselines?: ServingBaselines | null;
  plan: {
    api_version: string;
    schema_version: string;
    kind: string;
    metadata: {
      created_at: string;
      generation: number;
      name: string;
      uid: string;
    };
    model: {
      model_id: string;
      revision: string;
      maximum_sequence_length: number;
      minimum_precision: string;
    };
    engine: {
      dtype: string;
      maximum_active_sequences: number;
      maximum_batched_tokens: number;
      quantization: string;
      runtime: string;
      tensor_parallelism: number;
      version: string;
    };
    hardware: {
      cpu: { architecture: string; logical_cores: number; model: string };
      gpu_count: number;
      gpus: { model?: string }[];
      hourly_price_usd: number;
      region: string;
      system_memory_bytes: number;
    };
    replica_topology: {
      initial_replicas: number;
      maximum_replicas: number;
      minimum_replicas: number;
      regions: string[];
    };
    routing: {
      kind: string;
      targets: { variant: string; weight: number }[];
    };
    admission: {
      maximum_queue_time_ms: number;
      queue_capacity: number;
      reject_when_predicted_late: boolean;
    };
    batching: {
      dynamic_batching: boolean;
      maximum_active_sequences: number;
      maximum_batch_delay_ms: number;
      maximum_batched_tokens: number;
    };
    autoscaling: {
      mode: string;
      safety_margin: number;
      target_utilization: number;
    };
    predicted_metrics: Record<string, IntervalMetric>;
    provenance: Record<string, unknown>;
    slo: {
      ttft: { maximum_ms: number; percentile: number }[];
      inter_token_latency: { maximum_ms: number; percentile: number }[];
      minimum_availability: number | null;
      objective_weights: NumericMap;
    };
    workload: {
      duration_seconds: number;
      output_tokens: Distribution;
      prompt_tokens: Distribution;
      request_classes: WorkloadClass[];
      seed: number;
      trace_digest: { algorithm: string; value: string };
    };
  };
};
