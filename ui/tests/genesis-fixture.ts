import {
  GENOME_REGIONS,
  type GenesisArtifactBundle,
  type GenesisGenome,
} from "../src/genesis-types";

const capsuleDigest = "a".repeat(64);
const acceptedGenomeHash = "b".repeat(64);
const baselineGenomeHash = "c".repeat(64);
const hardwareFingerprint = "d".repeat(64);
const workloadFingerprint = "e".repeat(64);

const genome = {
  genome_id: "genome-hybrid-demo",
  kind: "InferenceGenome",
  schema_version: "1.0.0",
  ...Object.fromEntries(
    GENOME_REGIONS.map((region) => [
      region,
      {
        node: {
          frozen: region === "workflow",
          legal_rewrite_rules: region === "workflow" ? [] : ["genesis.rules/exact-v1"],
          proof_obligations: [{
            minimum_level: "level_1_differential",
            obligation_id: `${region}.reference-differential`,
            property: `${region} matches the reference contract`,
            required: true,
            scope: "batch=1, sequence=1..64",
          }],
          stable_id: region,
        },
      },
    ]),
  ),
} as GenesisGenome;

export const genesisFixture: GenesisArtifactBundle = {
  artifact_type: "sloforge.genesis.ui-bundle/v1",
  summary: {
    accepted_candidate_id: "candidate-corrected",
    accepted_genome_hash: acceptedGenomeHash,
    active_stream_preserved: true,
    baseline_genome_hash: baselineGenomeHash,
    capsule_digest: capsuleDigest,
    capsule_promotion_eligible: true,
    cross_layer_accepted: true,
    evolution_promoted: true,
    hardware_backed: false,
    learned_constraint_ids: ["constraint-cancel-before-emit"],
    minimized_counterexample_ids: ["counterexample-minimized"],
    operator_count: 79,
    package_id: "hybrid-decoder-zero-day",
    physical_degradation_triggered: true,
    rejected_candidate_ids: ["candidate-fast"],
    runtime_differential_passed: true,
    schema_version: "1.0.0",
    seed: 73129,
    state_field_count: 5,
  },
  genome,
  candidates: [
    {
      candidate: {
        candidate_id: "candidate-fast",
        genome_hash: { algorithm: "sha256", value: "f".repeat(64) },
        lifecycle: [
          { from_state: null, reason: "local proposal", sequence: 0, to_state: "PROPOSED" },
          { from_state: "PROPOSED", reason: "typed policy accepted", sequence: 1, to_state: "COMPILED" },
          { from_state: "COMPILED", reason: "token emitted after cancellation", sequence: 2, to_state: "SEMANTIC_REJECTED" },
        ],
        parent_candidate_ids: [],
        state: "SEMANTIC_REJECTED",
        transformation_ids: ["deadline-batch-fast"],
      },
      design: {
        candidate_id: "candidate-fast",
        mutations: [{
          expected_upside: 0.18,
          family: "batching_transformation",
          invalidity_risk: 0.3,
          regions: ["request", "serving"],
          transformation_id: "deadline-batch-fast",
        }],
        parent_candidate_ids: [],
        proposal_engine: "fixture",
        seed: 73129,
      },
    },
    {
      candidate: {
        candidate_id: "candidate-corrected",
        genome_hash: { algorithm: "sha256", value: acceptedGenomeHash },
        lifecycle: [
          { from_state: null, reason: "corrective proposal", sequence: 0, to_state: "PROPOSED" },
          { from_state: "PROPOSED", reason: "runtime differential passed", sequence: 1, to_state: "REFERENCE_TESTED" },
          { from_state: "REFERENCE_TESTED", reason: "bounded verifier passed", sequence: 2, to_state: "PROPERTY_TESTED" },
        ],
        parent_candidate_ids: [],
        state: "PROPERTY_TESTED",
        transformation_ids: ["deadline-batch-corrected"],
      },
      design: {
        candidate_id: "candidate-corrected",
        mutations: [{
          expected_upside: 0.14,
          family: "batching_transformation",
          invalidity_risk: 0.05,
          regions: ["request", "serving"],
          transformation_id: "deadline-batch-corrected",
        }],
        parent_candidate_ids: [],
        proposal_engine: "corrective",
        seed: 73131,
      },
    },
  ],
  counterexamples: [{
    candidate_id: "candidate-fast",
    counterexample_id: "counterexample-minimized",
    expected: { description: "cancelled request emits no token" },
    minimized: true,
    observed: { description: "candidate emitted one committed token" },
    parent_counterexample_id: "counterexample-original",
    payload: {
      events: [
        { action: "admit", at_step: 0, request_id: "request-a", worker_id: "worker-0" },
        { action: "cancel", at_step: 1, request_id: "request-a", worker_id: "worker-0" },
        { action: "emit", at_step: 2, request_id: "request-a", worker_id: "worker-0" },
      ],
      kind: "request_trace",
    },
    reproduction: { seed: 73129, timeout_seconds: 30 },
    scope: "transformation_family",
    transformation_id: "deadline-batch-fast",
    violated_contract: "no committed token is emitted after cancellation",
  }],
  capsule: {
    benchmarks: [{
      benchmark_id: "local-service-model",
      hardware_fingerprint: { algorithm: "sha256", value: hardwareFingerprint },
      repetitions: 3,
      sample_count: 3,
      summary: {
        confidence_high: 78,
        confidence_low: 74,
        effect_size: 0.08,
        median: 76,
        metric: "mean_request_completion_time",
        objective: "minimize",
        practical_significance_threshold: 0.05,
        regression_probability: 0,
        tail_percentile: 78.4,
        unit: "simulated_milliseconds",
      },
      warmup_iterations: 1,
    }],
    capsule_digest: { algorithm: "sha256", value: capsuleDigest },
    claims: [{
      category: "semantic",
      claim_id: "claim:semantic",
      evidence_ids: ["evidence:semantic"],
      level: "level_2_property",
      promotion_required: true,
      result: "pass",
      scope: {
        assumptions: [],
        exclusions: ["unseen custom operators"],
        input_domain: ["HybridDecoder batch=1, sequence=1..64"],
        shape_domain: ["input_ids[1,1..64]"],
      },
      statement: "runtime matches the declared reference corpus",
    }],
    evidence: [{
      deterministic_seed: 73131,
      evidence_class: "semantic",
      evidence_id: "evidence:semantic",
      issuer: "operator_verifier",
      level: "level_2_property",
      result: "pass",
    }],
    hardware: {
      architectures: ["cpu"],
      restrictions: ["CPU-only local evidence; CUDA paths unexercised"],
    },
    identity: { candidate_genome_hash: { algorithm: "sha256", value: acceptedGenomeHash } },
  },
  benchmark_definition: {
    benchmark_id: "local-service-model",
    confidence: 0.95,
    hardware_backed: false,
    metric: "mean_request_completion_time",
    repetitions: 3,
    unit: "simulated_milliseconds",
    warmup: 1,
  },
  baseline_samples: {
    hardware_fingerprint: { algorithm: "sha256", value: hardwareFingerprint },
    samples: [
      { seed: 73129, trial: 0, value: 85 },
      { seed: 73130, trial: 1, value: 84 },
      { seed: 73131, trial: 2, value: 83 },
    ],
    workload_fingerprint: { algorithm: "sha256", value: workloadFingerprint },
  },
  candidate_samples: {
    hardware_fingerprint: { algorithm: "sha256", value: hardwareFingerprint },
    samples: [
      { seed: 73129, trial: 0, value: 77 },
      { seed: 73130, trial: 1, value: 76 },
      { seed: 73131, trial: 2, value: 75 },
    ],
    workload_fingerprint: { algorithm: "sha256", value: workloadFingerprint },
  },
  evolution: {
    active_trigger: "fabric_degradation",
    active_streams: [{
      capsule_id: "genesis-champion",
      externally_visible_output: true,
      stream_id: "active-stream-before-promotion",
    }],
    audit: [
      { action: "initialize", candidate_id: null, observed_at_ms: 0, phase_after: "idle", phase_before: "idle", reason: "validated champion loaded", sequence: 0 },
      { action: "begin_evolution", candidate_id: null, observed_at_ms: 10, phase_after: "evolving", phase_before: "idle", reason: "workload drift", sequence: 1 },
      { action: "promote", candidate_id: "candidate-corrected-next", observed_at_ms: 70, phase_after: "promoted", phase_before: "ready_to_promote", reason: "capsule revalidated", sequence: 2 },
      { action: "begin_evolution", candidate_id: null, observed_at_ms: 100, phase_after: "evolving", phase_before: "promoted", reason: "fabric degradation", sequence: 3 },
    ],
    champion: {
      capsule_digest: "1".repeat(64),
      capsule_id: "genesis-corrected",
      genome_hash: "2".repeat(64),
    },
    challengers: [{
      spec: {
        candidate_id: "candidate-corrected-next",
        capsule: { capsule_digest: "1".repeat(64), capsule_id: "genesis-corrected" },
      },
      status: "promoted",
    }],
    phase: "evolving",
    previous_champion: {
      capsule_digest: capsuleDigest,
      capsule_id: "genesis-champion",
      genome_hash: acceptedGenomeHash,
    },
    seed: 73129,
  },
  lineage: {
    affected_evidence_count: 2,
    cases: {
      empty_lineage: { lineage_seed_count: 0, lineage_seed_ids: [], reverification_required: true, unseeded_count: 5 },
      unrelated_lineage: { lineage_seed_count: 0, lineage_seed_ids: [], reverification_required: true, unseeded_count: 5 },
      related_lineage: { lineage_seed_count: 1, lineage_seed_ids: ["transformation-related"], reverification_required: true, unseeded_count: 4 },
      stale_dependency_before_invalidation: { lineage_seed_count: 1, lineage_seed_ids: ["transformation-related"], reverification_required: true, unseeded_count: 4 },
      stale_dependency_after_invalidation: { lineage_seed_count: 0, lineage_seed_ids: [], reverification_required: true, unseeded_count: 5 },
    },
    performance_hypothesis_evaluated: false,
    related_seed_retrieved: true,
    schema_version: "1.0.0",
    scope: "deterministic lineage retrieval and invalidation mechanics; no speedup claim",
    seed: 73129,
    stale_seed_suppressed_after_invalidation: true,
  },
};
