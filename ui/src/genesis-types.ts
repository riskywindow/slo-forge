export const GENOME_REGIONS = [
  "workflow",
  "request",
  "serving",
  "state",
  "distributed",
  "tensor",
  "kernel",
  "recovery",
] as const;

export type GenomeRegionName = (typeof GENOME_REGIONS)[number];

export type GenesisDigest = {
  algorithm: "sha256";
  value: string;
};

export type GenesisDemoSummary = {
  accepted_candidate_id: string;
  accepted_genome_hash: string;
  active_stream_preserved: boolean;
  baseline_genome_hash: string;
  capsule_digest: string;
  capsule_external_production_eligible: boolean;
  capsule_local_evolution_eligible: boolean;
  capsule_promotion_eligible: boolean;
  cross_layer_accepted: boolean;
  evolution_promoted: boolean;
  hardware_backed: boolean;
  kernel_causal_attribution: boolean;
  kernel_measurement_scope: string;
  learned_constraint_ids: string[];
  minimized_counterexample_ids: string[];
  operator_count: number;
  package_id: string;
  physical_degradation_triggered: boolean;
  rejected_candidate_ids: string[];
  runtime_differential_passed: boolean;
  schema_version: "1.0.0";
  seed: number;
  state_field_count: number;
};

export type ProofObligation = {
  minimum_level: string;
  obligation_id: string;
  property: string;
  required: boolean;
  scope: string;
};

export type GenomeRegion = {
  node: {
    frozen: boolean;
    legal_rewrite_rules: string[];
    proof_obligations: ProofObligation[];
    stable_id: string;
  };
};

export type GenesisGenome = {
  genome_id: string;
  kind: "InferenceGenome";
  schema_version: "1.0.0";
} & Record<GenomeRegionName, GenomeRegion>;

export type GenesisMutation = {
  expected_upside: number;
  family: string;
  invalidity_risk: number;
  regions: GenomeRegionName[];
  transformation_id: string;
};

export type GenesisCandidateDesign = {
  candidate_id: string;
  mutations: GenesisMutation[];
  parent_candidate_ids: string[];
  proposal_engine: string;
  seed: number;
};

export type CandidateLifecycle = {
  from_state: string | null;
  reason: string;
  sequence: number;
  to_state: string;
};

export type GenesisCandidate = {
  candidate_id: string;
  genome_hash: GenesisDigest;
  lifecycle: CandidateLifecycle[];
  parent_candidate_ids: string[];
  state: string;
  transformation_ids: string[];
};

export type GenesisCandidateBundle = {
  candidate: GenesisCandidate;
  design: GenesisCandidateDesign;
};

export type CounterexampleEvent = {
  action: string;
  at_step: number;
  request_id: string;
  worker_id: string;
};

export type GenesisCounterexample = {
  candidate_id: string;
  counterexample_id: string;
  expected: { description: string };
  minimized: boolean;
  observed: { description: string };
  parent_counterexample_id: string | null;
  payload: { events: CounterexampleEvent[]; kind: string };
  reproduction: { seed: number; timeout_seconds: number };
  scope: string;
  transformation_id: string | null;
  violated_contract: string;
};

export type CapsuleClaim = {
  category: string;
  claim_id: string;
  evidence_ids: string[];
  level: string;
  promotion_required: boolean;
  result: string;
  scope: {
    assumptions: string[];
    exclusions: string[];
    input_domain: string[];
    shape_domain: string[];
  };
  statement: string;
};

export type CapsuleEvidence = {
  deterministic_seed: number;
  evidence_class: string;
  evidence_id: string;
  issuer: string;
  level: string;
  result: string;
};

export type CapsuleBenchmark = {
  benchmark_id: string;
  hardware_fingerprint: GenesisDigest;
  repetitions: number;
  sample_count: number;
  summary: {
    confidence_high: number;
    confidence_low: number;
    effect_size: number;
    median: number;
    metric: string;
    objective: string;
    practical_significance_threshold: number;
    regression_probability: number;
    tail_percentile: number;
    unit: string;
  };
  warmup_iterations: number;
};

export type GenesisCapsule = {
  benchmarks: CapsuleBenchmark[];
  capsule_digest: GenesisDigest;
  claims: CapsuleClaim[];
  evidence: CapsuleEvidence[];
  hardware: { architectures: string[]; restrictions: string[] };
  identity: { candidate_genome_hash: GenesisDigest };
};

export type BenchmarkDefinition = {
  benchmark_id: string;
  confidence: number;
  hardware_backed: boolean;
  metric: string;
  repetitions: number;
  unit: string;
  warmup: number;
};

export type BenchmarkSamples = {
  hardware_fingerprint: GenesisDigest;
  samples: { execution_ordinal: number; seed: number; trial: number; value: number }[];
  workload_fingerprint: GenesisDigest;
};

export type PerformanceSimulationRequest = {
  deadline_ms: number | null;
  modeled_service_units: number;
  ordinal: number;
  policy_batch_limit: number;
};

export type PerformanceSimulation = {
  candidate_genome_hash: string;
  candidate_id: string;
  comparison_permitted: false;
  deadline_order_exercised: boolean;
  events: (PerformanceSimulationRequest & { completion_units: number })[];
  hardware_backed: false;
  policy_bytecode_sha256: string;
  queue_policy: string;
  raw_requests: PerformanceSimulationRequest[];
  result: "pass";
  runtime_manifest_sha256: string;
  schema_version: "genesis.candidate-simulation.v1";
  seed: number;
  workload_path: string;
  workload_sha256: string;
};

export type EvolutionAudit = {
  action: string;
  candidate_id: string | null;
  observed_at_ms: number;
  phase_after: string;
  phase_before: string;
  reason: string;
  sequence: number;
};

export type EvolutionSnapshot = {
  active_trigger: string | null;
  active_streams: {
    capsule_id: string;
    externally_visible_output: boolean;
    stream_id: string;
  }[];
  audit: EvolutionAudit[];
  champion: { capsule_digest: string; capsule_id: string; genome_hash: string };
  challengers: {
    spec: { candidate_id: string; capsule: { capsule_digest: string; capsule_id: string } };
    status: string;
  }[];
  phase: string;
  previous_champion: {
    capsule_digest: string;
    capsule_id: string;
    genome_hash: string;
  } | null;
  seed: number;
};

export type LineageCase = {
  lineage_seed_count: number;
  lineage_seed_ids: string[];
  reverification_required: boolean;
  unseeded_count: number;
};

export type LineageTransferReport = {
  affected_evidence_count: number;
  cases: {
    empty_lineage: LineageCase;
    related_lineage: LineageCase;
    stale_dependency_after_invalidation: LineageCase;
    stale_dependency_before_invalidation: LineageCase;
    unrelated_lineage: LineageCase;
  };
  performance_hypothesis_evaluated: boolean;
  related_seed_retrieved: boolean;
  schema_version: "1.0.0";
  scope: string;
  seed: number;
  stale_seed_suppressed_after_invalidation: boolean;
};

export type GenesisArtifactBundle = {
  artifact_type: "sloforge.genesis.ui-bundle/v1";
  baseline_samples: BenchmarkSamples | null;
  benchmark_definition: BenchmarkDefinition | null;
  candidate_samples: BenchmarkSamples | null;
  candidates: GenesisCandidateBundle[];
  capsule: GenesisCapsule;
  counterexamples: GenesisCounterexample[];
  evolution: EvolutionSnapshot;
  genome: GenesisGenome;
  lineage: LineageTransferReport;
  performance_simulation: PerformanceSimulation | null;
  summary: GenesisDemoSummary;
};
