export type FabricCurvePoint = {
  confidence_high: number;
  confidence_low: number;
  median: number;
  message_bytes: number;
  p95: number;
  robust_dispersion: number;
  sample_count: number;
};

export type FabricTopologyNode = {
  health?: string;
  host_id?: string;
  kind: string;
  node_id: string;
  numa_domain_id?: string | null;
  product?: string;
  speed_gbps?: number | null;
  transport?: string;
};

export type FabricTopologyEdge = {
  bandwidth_curve_gbps: FabricCurvePoint[];
  connection: string;
  contention_domain: string;
  edge_id: string;
  health: string;
  latency_curve_us: FabricCurvePoint[];
  measurement_confidence: number;
  sharing_group: string;
  source_node_id: string;
  target_node_id: string;
  theoretical_bandwidth_gbps: number | null;
};

export type FabricTopology = {
  api_version: string;
  container_limited: boolean;
  discovery_warnings: string[];
  edges: FabricTopologyEdge[];
  kind: "TopologyGraph";
  nodes: FabricTopologyNode[];
  schema_version: string;
  topology_id: string;
};

export type FabricMeasurement = {
  concurrency: number;
  confidence_high_us: number;
  confidence_low_us: number;
  measurement_id: string;
  message_bytes: number;
  primitive: string;
  rank_count: number;
  samples: {
    bytes_transferred: number;
    duration_us: number;
    failure_reason: string | null;
    success: boolean;
  }[];
  summary_median_us: number;
  summary_p95_us: number;
  transport: string;
  warmup_count: number;
};

export type FabricProfile = {
  kind: "FabricProfile";
  measurements: FabricMeasurement[];
  profile_id: string;
  schema_version: string;
};

export type PhysicalMetric = {
  confidence: number;
  estimate: number;
  lower: number;
  unit: string;
  upper: number;
};

export type RankBinding = {
  fault_domain: string;
  gpu_id: string;
  host_id: string;
  network_rail_id: string | null;
  nic_id: string | null;
  numa_domain_id: string;
  process_cpu_affinity: string;
  rank_id: number;
  replica_id: string;
  worker_role: string;
};

export type ExpertAssignment = {
  capacity_factor: number;
  expected_load: number;
  expert_id: string;
  rank_ids: number[];
};

export type CollectiveOperation = {
  algorithm: string;
  channel_count: number;
  expected_duration_us: number;
  operation_id: string;
  operation: string;
  overlap_window_id: string | null;
  participating_ranks: number[];
  rail_ids: string[];
  transport: string;
};

export type KVRoute = {
  chunk_bytes: number;
  consumer_rank_ids: number[];
  edge_path: string[];
  expected_latency_us: number;
  maximum_inflight_chunks: number;
  overlap_with_decode: boolean;
  producer_rank_ids: number[];
  route_id: string;
  transport_adapter: string;
};

export type OverlapWindow = {
  communication_operation_id: string;
  compute_operation_id: string;
  expected_overlap_fraction: number;
  fallback_serialization: string;
  resource_contention: string;
  window_id: string;
};

export type ParallelismGroup = {
  group_id: string;
  kind: string;
  rank_ids: number[];
};

export type PhysicalPlan = {
  api_version: string;
  bottleneck_prediction: string;
  collectives: { operations: CollectiveOperation[] };
  communication_overlap: { windows: OverlapWindow[] };
  compiler_version: string;
  expert_placement: {
    assignments: ExpertAssignment[];
    hot_expert_strategy: string;
  };
  kind: "PhysicalExecutionPlan";
  kv_transfer: { routes: KVRoute[] };
  optimizer_history: {
    candidate_id: string;
    decision: string;
    phase: string;
    reason_code: string;
    sequence: number;
    simulator_calls: number;
    solver_time_ms: number;
  }[];
  parallelism: {
    context_parallel_degree: number;
    data_parallel_degree: number;
    expert_parallel_degree: number;
    groups: ParallelismGroup[];
    pipeline_parallel_degree: number;
    prefill_decode_disaggregated: boolean;
    replica_groups: ParallelismGroup[];
    tensor_parallel_degree: number;
  };
  plan_id: string;
  predicted_metrics: Record<string, PhysicalMetric>;
  rank_placement: { bindings: RankBinding[] };
  recovery_variants: unknown[];
  rejected_alternatives: unknown[];
  schema_version: string;
  topology_fingerprint: { algorithm: string; value: string };
};

export type PhysicalCandidate = {
  availability: number;
  candidate_id: string;
  communication_us: number;
  cost_per_million_tokens: number;
  feasible: boolean;
  goodput_tokens_per_second: number;
  gpu_ids: string[];
  objective_score: number;
  p95_ttft_ms: number;
  p99_tpot_ms: number;
  rejection_codes: string[];
};

export type PhysicalOptimizer = {
  all_candidates: PhysicalCandidate[];
  pareto_frontier: PhysicalCandidate[];
  selected: PhysicalPlan;
  simulator_calls: number;
  solver_time_ms: number;
  strategy: string;
};

export type EvidenceSignal = {
  event_ids: string[];
  explanation: string;
  metric: string;
  observed: number;
  supports_hypothesis: boolean;
  threshold: number;
};

export type DiagnosisHypothesis = {
  confidence: number;
  contradicting_evidence: EvidenceSignal[];
  hypothesis_id: string;
  kind: string;
  rejected_reason: string | null;
  supporting_evidence: EvidenceSignal[];
  target: string;
};

export type FabricDiagnosis = {
  confidence: number;
  diagnosis_id: string;
  first_divergence_event_id: string | null;
  first_divergence_ns: number | null;
  hypotheses: DiagnosisHypothesis[];
};

export type CounterfactualEvaluation = {
  confidence: number;
  expected_improvement_ms: number;
  healthy_reference_residual_ms: number;
  lower_improvement_ms: number;
  rejected_reason: string | null;
  scenario: {
    hypothesis_kind: string;
    rationale: string;
    scenario_id: string;
  };
  status: string;
  upper_improvement_ms: number;
};

export type FabricCounterfactuals = {
  diagnosis_id: string;
  evaluations: CounterfactualEvaluation[];
  rejected_scenario_ids: string[];
  schema_version: string;
  selected_scenario_id: string;
};

export type RecoveryAction = {
  action_id: string;
  kind: string;
  order: number;
  requires_external_mutation: boolean;
  scope: string;
  target_ids: string[];
  timeout_seconds: number;
};

export type RecoveryPlan = {
  actions: RecoveryAction[];
  confidence: number;
  expected_slo_improvement: Record<string, PhysicalMetric>;
  external_mutation_authorized: boolean;
  recovery_id: string;
  schema_version: string;
  traffic_migration: {
    canary_fraction: number;
    minimum_canary_samples: number;
    minimum_shadow_samples: number;
    preserve_started_streams: boolean;
    shadow_fraction: number;
  };
};

export type RecoveryAuditRecord = {
  at_ms: number;
  event: string;
  reason: string;
  sequence: number;
  state_after: string;
  state_before: string;
};

export type RecoveryExecution = {
  action_attempts: {
    action_id: string;
    attempted_at_ms: number;
    detail: string;
    succeeded: boolean;
  }[];
  audit: RecoveryAuditRecord[];
  recovery_id: string;
  schema_version: string;
  state: string;
};

export type SimulationMetrics = {
  cost_usd: number;
  makespan_us: number;
  operation_count: number;
  overlap_efficiency: number;
  predicted_lower_us: number;
  predicted_upper_us: number;
  processed_events: number;
  resources: {
    busy_time_us: number;
    max_concurrent: number;
    resource_id: string;
    transferred_bytes: number;
    utilization: number;
  }[];
  total_transferred_bytes: number;
  total_work_us: number;
};

export type FabricSimulation = {
  applied_faults: string[];
  metrics: SimulationMetrics;
  schema_version: string;
};

export type DemoMetricSummary = {
  makespan_ms: number;
  p95_end_to_end_ms: number;
  p95_ttft_ms: number;
  p99_tpot_ms: number;
};

export type FabricManifest = {
  artifacts: { path: string; sha256: string }[];
  baseline_plan_id: string;
  counterfactuals_evaluated: number;
  degraded: DemoMetricSummary;
  degraded_slo_attained: boolean;
  diagnosis: string;
  diagnosis_confidence: number;
  ground_truth_faults: string[];
  healthy: DemoMetricSummary;
  healthy_slo_attained: boolean;
  p95_ttft_slo_ms: number;
  physical_plan_id: string;
  recovery_final_state: string;
  restored: DemoMetricSummary;
  restored_slo_attained: boolean;
  schema_version: "sloforge.fabric.demo/v1";
  seed: number;
  selected_counterfactual: string;
  synthetic_hardware: boolean;
  topology_fingerprint: string;
};

export type FabricTimelineEvent = {
  at_ms: number;
  detail: string;
  event: string;
  evidence_uri: string;
  sequence: number;
};

export type FabricArtifactBundle = {
  artifact_type: "sloforge.fabric.ui-bundle/v1";
  counterfactuals: FabricCounterfactuals;
  diagnosis: FabricDiagnosis;
  fabric_profile: FabricProfile;
  manifest: FabricManifest;
  optimizer: PhysicalOptimizer;
  physical_plan: PhysicalPlan;
  recovery_execution: RecoveryExecution;
  recovery_plan: RecoveryPlan;
  simulations: {
    degraded: FabricSimulation;
    healthy: FabricSimulation;
    restored: FabricSimulation;
  };
  timeline: FabricTimelineEvent[];
  topology: FabricTopology;
};

export type ArtifactDocument =
  | { kind: "fabric"; value: FabricArtifactBundle }
  | { kind: "logical"; value: import("./types").ReportArtifact };
