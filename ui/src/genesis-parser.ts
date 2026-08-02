import {
  GENOME_REGIONS,
  type GenesisArtifactBundle,
} from "./genesis-types";

type UnknownRecord = Record<string, unknown>;

export class GenesisArtifactValidationError extends Error {
  public readonly problems: string[];

  public constructor(problems: string[]) {
    super(`Invalid SLOForge Genesis artifact:\n${problems.join("\n")}`);
    this.name = "GenesisArtifactValidationError";
    this.problems = problems;
  }
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseGenesisArtifactBundle(input: unknown): GenesisArtifactBundle {
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
  const string = (value: unknown, path: string): value is string => {
    if (typeof value !== "string" || value.length === 0) {
      problems.push(`${path} must be a non-empty string`);
      return false;
    }
    return true;
  };
  const number = (value: unknown, path: string): value is number => {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      problems.push(`${path} must be a finite number`);
      return false;
    }
    return true;
  };
  const boolean = (value: unknown, path: string): value is boolean => {
    if (typeof value !== "boolean") {
      problems.push(`${path} must be a boolean`);
      return false;
    }
    return true;
  };
  const digest = (value: unknown, path: string): string | null => {
    const item = record(value, path);
    if (item["algorithm"] !== "sha256") problems.push(`${path}.algorithm must be sha256`);
    if (
      typeof item["value"] !== "string" ||
      !/^[a-f0-9]{64}$/.test(item["value"])
    ) {
      problems.push(`${path}.value must be a lowercase SHA-256`);
      return null;
    }
    return item["value"];
  };
  const strings = (value: unknown, path: string): string[] => {
    const output: string[] = [];
    array(value, path).forEach((item, index) => {
      if (string(item, `${path}[${index}]`)) output.push(item);
    });
    return output;
  };

  const root = record(input, "$");
  if (root["artifact_type"] !== "sloforge.genesis.ui-bundle/v1") {
    problems.push("$.artifact_type must be sloforge.genesis.ui-bundle/v1");
  }

  const summary = record(root["summary"], "$.summary");
  if (summary["schema_version"] !== "1.0.0") {
    problems.push("$.summary.schema_version must be 1.0.0");
  }
  for (const field of ["accepted_candidate_id", "package_id"]) {
    string(summary[field], `$.summary.${field}`);
  }
  for (const field of [
    "accepted_genome_hash",
    "baseline_genome_hash",
    "capsule_digest",
  ]) {
    if (typeof summary[field] !== "string" || !/^[a-f0-9]{64}$/.test(summary[field])) {
      problems.push(`$.summary.${field} must be a lowercase SHA-256`);
    }
  }
  for (const field of [
    "active_stream_preserved",
    "capsule_external_production_eligible",
    "capsule_local_evolution_eligible",
    "capsule_promotion_eligible",
    "cross_layer_accepted",
    "evolution_promoted",
    "hardware_backed",
    "kernel_causal_attribution",
    "physical_degradation_triggered",
    "runtime_differential_passed",
  ]) boolean(summary[field], `$.summary.${field}`);
  string(summary["kernel_measurement_scope"], "$.summary.kernel_measurement_scope");
  if (
    summary["capsule_external_production_eligible"] === true &&
    summary["capsule_local_evolution_eligible"] !== true
  ) {
    problems.push("$.summary external capsule eligibility requires local evolution eligibility");
  }
  for (const field of ["operator_count", "seed", "state_field_count"]) {
    if (number(summary[field], `$.summary.${field}`) && summary[field] < 0) {
      problems.push(`$.summary.${field} must be non-negative`);
    }
  }
  const rejectedIds = strings(summary["rejected_candidate_ids"], "$.summary.rejected_candidate_ids");
  const minimizedIds = strings(
    summary["minimized_counterexample_ids"],
    "$.summary.minimized_counterexample_ids",
  );
  strings(summary["learned_constraint_ids"], "$.summary.learned_constraint_ids");

  const genome = record(root["genome"], "$.genome");
  if (genome["kind"] !== "InferenceGenome") problems.push("$.genome.kind must be InferenceGenome");
  if (genome["schema_version"] !== "1.0.0") {
    problems.push("$.genome.schema_version must be 1.0.0");
  }
  string(genome["genome_id"], "$.genome.genome_id");
  const obligationIds = new Set<string>();
  for (const regionName of GENOME_REGIONS) {
    const region = record(genome[regionName], `$.genome.${regionName}`);
    const node = record(region["node"], `$.genome.${regionName}.node`);
    string(node["stable_id"], `$.genome.${regionName}.node.stable_id`);
    boolean(node["frozen"], `$.genome.${regionName}.node.frozen`);
    const rules = strings(
      node["legal_rewrite_rules"],
      `$.genome.${regionName}.node.legal_rewrite_rules`,
    );
    if (node["frozen"] === false && rules.length === 0) {
      problems.push(`$.genome.${regionName}.node mutable region requires a rewrite rule`);
    }
    const obligations = array(
      node["proof_obligations"],
      `$.genome.${regionName}.node.proof_obligations`,
    );
    if (obligations.length === 0) {
      problems.push(`$.genome.${regionName}.node.proof_obligations must not be empty`);
    }
    obligations.forEach((raw, index) => {
      const item = record(raw, `$.genome.${regionName}.node.proof_obligations[${index}]`);
      const idPath = `$.genome.${regionName}.node.proof_obligations[${index}].obligation_id`;
      if (string(item["obligation_id"], idPath)) {
        if (obligationIds.has(item["obligation_id"])) problems.push(`${idPath} must be unique`);
        obligationIds.add(item["obligation_id"]);
      }
      for (const field of ["minimum_level", "property", "scope"]) {
        string(item[field], `${idPath}.${field}`);
      }
      boolean(item["required"], `${idPath}.required`);
    });
  }

  const candidateIds = new Set<string>();
  const candidateStates = new Map<string, string>();
  const candidateGenomeHashes = new Map<string, string>();
  array(root["candidates"], "$.candidates").forEach((raw, index) => {
    const bundle = record(raw, `$.candidates[${index}]`);
    const candidate = record(bundle["candidate"], `$.candidates[${index}].candidate`);
    const design = record(bundle["design"], `$.candidates[${index}].design`);
    const candidateId = candidate["candidate_id"];
    if (string(candidateId, `$.candidates[${index}].candidate.candidate_id`)) {
      if (candidateIds.has(candidateId)) {
        problems.push(`$.candidates[${index}].candidate.candidate_id must be unique`);
      }
      candidateIds.add(candidateId);
      if (typeof candidate["state"] === "string") candidateStates.set(candidateId, candidate["state"]);
      const hash = digest(candidate["genome_hash"], `$.candidates[${index}].candidate.genome_hash`);
      if (hash !== null) candidateGenomeHashes.set(candidateId, hash);
    }
    if (design["candidate_id"] !== candidateId) {
      problems.push(`$.candidates[${index}] candidate and design identifiers must match`);
    }
    string(candidate["state"], `$.candidates[${index}].candidate.state`);
    const transformationIds = strings(
      candidate["transformation_ids"],
      `$.candidates[${index}].candidate.transformation_ids`,
    );
    const parentIds = strings(
      candidate["parent_candidate_ids"],
      `$.candidates[${index}].candidate.parent_candidate_ids`,
    );
    const designParentIds = strings(
      design["parent_candidate_ids"],
      `$.candidates[${index}].design.parent_candidate_ids`,
    );
    if (parentIds.join("\0") !== designParentIds.join("\0")) {
      problems.push(`$.candidates[${index}] candidate and design parents must match`);
    }
    number(design["seed"], `$.candidates[${index}].design.seed`);
    string(design["proposal_engine"], `$.candidates[${index}].design.proposal_engine`);
    const mutationIds: string[] = [];
    array(design["mutations"], `$.candidates[${index}].design.mutations`).forEach(
      (mutationRaw, mutationIndex) => {
        const mutation = record(
          mutationRaw,
          `$.candidates[${index}].design.mutations[${mutationIndex}]`,
        );
        if (
          string(
            mutation["transformation_id"],
            `$.candidates[${index}].design.mutations[${mutationIndex}].transformation_id`,
          )
        ) mutationIds.push(mutation["transformation_id"]);
        string(mutation["family"], `$.candidates[${index}].design.mutations[${mutationIndex}].family`);
        number(
          mutation["expected_upside"],
          `$.candidates[${index}].design.mutations[${mutationIndex}].expected_upside`,
        );
        number(
          mutation["invalidity_risk"],
          `$.candidates[${index}].design.mutations[${mutationIndex}].invalidity_risk`,
        );
        strings(mutation["regions"], `$.candidates[${index}].design.mutations[${mutationIndex}].regions`)
          .forEach((region) => {
            if (!GENOME_REGIONS.some((known) => known === region)) {
              problems.push(`$.candidates[${index}] mutation references unknown region ${region}`);
            }
          });
      },
    );
    if (transformationIds.join("\0") !== mutationIds.join("\0")) {
      problems.push(`$.candidates[${index}] transformation identifiers must match mutations`);
    }
    let previousState: string | null = null;
    array(candidate["lifecycle"], `$.candidates[${index}].candidate.lifecycle`).forEach(
      (eventRaw, eventIndex) => {
        const event = record(
          eventRaw,
          `$.candidates[${index}].candidate.lifecycle[${eventIndex}]`,
        );
        if (event["sequence"] !== eventIndex) {
          problems.push(`$.candidates[${index}].candidate.lifecycle[${eventIndex}].sequence must be contiguous`);
        }
        if (event["from_state"] !== previousState) {
          problems.push(`$.candidates[${index}].candidate.lifecycle[${eventIndex}].from_state must match prior state`);
        }
        string(event["to_state"], `$.candidates[${index}].candidate.lifecycle[${eventIndex}].to_state`);
        string(event["reason"], `$.candidates[${index}].candidate.lifecycle[${eventIndex}].reason`);
        previousState = typeof event["to_state"] === "string" ? event["to_state"] : null;
      },
    );
    if (previousState !== candidate["state"]) {
      problems.push(`$.candidates[${index}].candidate.state must match final lifecycle state`);
    }
  });
  const acceptedId = typeof summary["accepted_candidate_id"] === "string"
    ? summary["accepted_candidate_id"]
    : "";
  if (!candidateIds.has(acceptedId)) problems.push("$.summary.accepted_candidate_id must reference a candidate");
  if (candidateGenomeHashes.get(acceptedId) !== summary["accepted_genome_hash"]) {
    problems.push("$.summary.accepted_genome_hash must match the accepted candidate");
  }
  for (const rejectedId of rejectedIds) {
    if (!candidateIds.has(rejectedId)) problems.push(`rejected candidate ${rejectedId} is missing`);
    if (!candidateStates.get(rejectedId)?.endsWith("REJECTED")) {
      problems.push(`rejected candidate ${rejectedId} must have a rejected terminal state`);
    }
  }

  const counterexampleIds = new Set<string>();
  array(root["counterexamples"], "$.counterexamples").forEach((raw, index) => {
    const item = record(raw, `$.counterexamples[${index}]`);
    const id = item["counterexample_id"];
    if (string(id, `$.counterexamples[${index}].counterexample_id`)) {
      if (counterexampleIds.has(id)) problems.push(`$.counterexamples[${index}].counterexample_id must be unique`);
      counterexampleIds.add(id);
    }
    if (typeof item["candidate_id"] !== "string" || !candidateIds.has(item["candidate_id"])) {
      problems.push(`$.counterexamples[${index}].candidate_id must reference a candidate`);
    }
    boolean(item["minimized"], `$.counterexamples[${index}].minimized`);
    for (const field of ["scope", "violated_contract"]) {
      string(item[field], `$.counterexamples[${index}].${field}`);
    }
    const payload = record(item["payload"], `$.counterexamples[${index}].payload`);
    string(payload["kind"], `$.counterexamples[${index}].payload.kind`);
    array(payload["events"], `$.counterexamples[${index}].payload.events`).forEach(
      (eventRaw, eventIndex) => {
        const event = record(eventRaw, `$.counterexamples[${index}].payload.events[${eventIndex}]`);
        if (event["at_step"] !== eventIndex) {
          problems.push(`$.counterexamples[${index}].payload.events must have contiguous steps`);
        }
        for (const field of ["action", "request_id", "worker_id"]) {
          string(event[field], `$.counterexamples[${index}].payload.events[${eventIndex}].${field}`);
        }
      },
    );
    const reproduction = record(item["reproduction"], `$.counterexamples[${index}].reproduction`);
    number(reproduction["seed"], `$.counterexamples[${index}].reproduction.seed`);
    number(reproduction["timeout_seconds"], `$.counterexamples[${index}].reproduction.timeout_seconds`);
  });
  for (const id of minimizedIds) {
    if (!counterexampleIds.has(id)) problems.push(`minimized counterexample ${id} is missing`);
  }

  const capsule = record(root["capsule"], "$.capsule");
  const capsuleDigest = digest(capsule["capsule_digest"], "$.capsule.capsule_digest");
  const identity = record(capsule["identity"], "$.capsule.identity");
  const capsuleGenomeHash = digest(
    identity["candidate_genome_hash"],
    "$.capsule.identity.candidate_genome_hash",
  );
  if (capsuleDigest !== summary["capsule_digest"]) {
    problems.push("$.capsule.capsule_digest must match $.summary.capsule_digest");
  }
  if (capsuleGenomeHash !== summary["accepted_genome_hash"]) {
    problems.push("$.capsule identity must reference the accepted genome");
  }
  const evidenceIds = new Set<string>();
  const evidenceClasses = new Map<string, string>();
  const evidenceResults = new Map<string, string>();
  array(capsule["evidence"], "$.capsule.evidence").forEach((raw, index) => {
    const item = record(raw, `$.capsule.evidence[${index}]`);
    if (string(item["evidence_id"], `$.capsule.evidence[${index}].evidence_id`)) {
      evidenceIds.add(item["evidence_id"]);
      if (typeof item["evidence_class"] === "string") {
        evidenceClasses.set(item["evidence_id"], item["evidence_class"]);
      }
      if (typeof item["result"] === "string") {
        evidenceResults.set(item["evidence_id"], item["result"]);
      }
    }
    for (const field of ["evidence_class", "issuer", "level", "result"]) {
      string(item[field], `$.capsule.evidence[${index}].${field}`);
    }
    number(item["deterministic_seed"], `$.capsule.evidence[${index}].deterministic_seed`);
  });
  const performanceClaims: UnknownRecord[] = [];
  array(capsule["claims"], "$.capsule.claims").forEach((raw, index) => {
    const item = record(raw, `$.capsule.claims[${index}]`);
    for (const field of ["category", "claim_id", "level", "result", "statement"]) {
      string(item[field], `$.capsule.claims[${index}].${field}`);
    }
    boolean(item["promotion_required"], `$.capsule.claims[${index}].promotion_required`);
    const claimEvidenceIds = strings(
      item["evidence_ids"],
      `$.capsule.claims[${index}].evidence_ids`,
    );
    claimEvidenceIds.forEach((id) => {
      if (!evidenceIds.has(id)) problems.push(`$.capsule.claims[${index}] references missing evidence ${id}`);
    });
    if (item["category"] === "performance" && item["result"] === "pass") {
      performanceClaims.push({ ...item, evidence_ids: claimEvidenceIds });
    }
    const scope = record(item["scope"], `$.capsule.claims[${index}].scope`);
    for (const field of ["assumptions", "exclusions", "input_domain", "shape_domain"]) {
      strings(scope[field], `$.capsule.claims[${index}].scope.${field}`);
    }
  });
  const hardware = record(capsule["hardware"], "$.capsule.hardware");
  strings(hardware["architectures"], "$.capsule.hardware.architectures");
  strings(hardware["restrictions"], "$.capsule.hardware.restrictions");

  const benchmarks = array(capsule["benchmarks"], "$.capsule.benchmarks");
  if (benchmarks.length > 1) problems.push("$.capsule.benchmarks supports at most one entry");
  if (benchmarks.length === 1) {
    if (root["performance_simulation"] !== null) {
      problems.push("$.performance_simulation must be null when benchmark evidence is accepted");
    }
    const definition = record(root["benchmark_definition"], "$.benchmark_definition");
    for (const field of ["benchmark_id", "metric", "unit"]) {
      string(definition[field], `$.benchmark_definition.${field}`);
    }
    for (const field of ["confidence", "repetitions", "warmup"]) {
      number(definition[field], `$.benchmark_definition.${field}`);
    }
    boolean(definition["hardware_backed"], "$.benchmark_definition.hardware_backed");
    if (definition["hardware_backed"] !== summary["hardware_backed"]) {
      problems.push("$.benchmark_definition.hardware_backed must match $.summary.hardware_backed");
    }
    const parseSamples = (name: "baseline_samples" | "candidate_samples") => {
      const samples = record(root[name], `$.${name}`);
      const hardwareHash = digest(
        samples["hardware_fingerprint"],
        `$.${name}.hardware_fingerprint`,
      );
      const workloadHash = digest(
        samples["workload_fingerprint"],
        `$.${name}.workload_fingerprint`,
      );
      const executionOrdinals: number[] = [];
      const values = array(samples["samples"], `$.${name}.samples`);
      values.forEach((raw, index) => {
        const sample = record(raw, `$.${name}.samples[${index}]`);
        number(sample["seed"], `$.${name}.samples[${index}].seed`);
        number(sample["value"], `$.${name}.samples[${index}].value`);
        if (number(sample["execution_ordinal"], `$.${name}.samples[${index}].execution_ordinal`)) {
          executionOrdinals.push(sample["execution_ordinal"]);
        }
        if (sample["trial"] !== index) {
          problems.push(`$.${name}.samples[${index}].trial must be contiguous`);
        }
      });
      return { count: values.length, executionOrdinals, hardwareHash, workloadHash };
    };
    const baseline = parseSamples("baseline_samples");
    const candidate = parseSamples("candidate_samples");
    if (
      baseline.count !== definition["repetitions"] ||
      candidate.count !== definition["repetitions"]
    ) {
      problems.push("benchmark raw sample counts must match the declared repetitions");
    }
    if (baseline.hardwareHash !== candidate.hardwareHash) {
      problems.push("baseline and candidate hardware fingerprints must match");
    }
    if (baseline.workloadHash !== candidate.workloadHash) {
      problems.push("baseline and candidate workload fingerprints must match");
    }
    const executionOrdinals = [...baseline.executionOrdinals, ...candidate.executionOrdinals];
    if (
      executionOrdinals.length !== baseline.count + candidate.count ||
      new Set(executionOrdinals).size !== executionOrdinals.length ||
      executionOrdinals.some((ordinal) =>
        !Number.isInteger(ordinal) || ordinal < 0 || ordinal >= executionOrdinals.length
      )
    ) {
      problems.push("benchmark execution ordinals must uniquely cover randomized run order");
    }
    const raw = benchmarks[0];
    const index = 0;
    const benchmark = record(raw, `$.capsule.benchmarks[${index}]`);
    if (benchmark["benchmark_id"] !== definition["benchmark_id"]) {
      problems.push(`$.capsule.benchmarks[${index}].benchmark_id must match the definition`);
    }
    if (benchmark["sample_count"] !== candidate.count) {
      problems.push(`$.capsule.benchmarks[${index}].sample_count must match candidate samples`);
    }
    if (benchmark["repetitions"] !== definition["repetitions"]) {
      problems.push(`$.capsule.benchmarks[${index}].repetitions must match the definition`);
    }
    if (benchmark["warmup_iterations"] !== definition["warmup"]) {
      problems.push(`$.capsule.benchmarks[${index}].warmup_iterations must match the definition`);
    }
    const benchmarkHardware = digest(
      benchmark["hardware_fingerprint"],
      `$.capsule.benchmarks[${index}].hardware_fingerprint`,
    );
    if (benchmarkHardware !== candidate.hardwareHash) {
      problems.push(`$.capsule.benchmarks[${index}] hardware fingerprint must match raw samples`);
    }
    const summaryRecord = record(benchmark["summary"], `$.capsule.benchmarks[${index}].summary`);
    for (const field of [
      "confidence_high",
      "confidence_low",
      "effect_size",
      "median",
      "practical_significance_threshold",
      "regression_probability",
      "tail_percentile",
    ]) number(summaryRecord[field], `$.capsule.benchmarks[${index}].summary.${field}`);
    for (const field of ["metric", "objective", "unit"]) {
      string(summaryRecord[field], `$.capsule.benchmarks[${index}].summary.${field}`);
    }
    if (
      typeof summaryRecord["confidence_low"] === "number" &&
      typeof summaryRecord["median"] === "number" &&
      typeof summaryRecord["confidence_high"] === "number" &&
      !(summaryRecord["confidence_low"] <= summaryRecord["median"] &&
        summaryRecord["median"] <= summaryRecord["confidence_high"])
    ) problems.push(`$.capsule.benchmarks[${index}].summary confidence interval must contain the median`);
  } else {
    if (
      root["benchmark_definition"] !== null ||
      root["baseline_samples"] !== null ||
      root["candidate_samples"] !== null
    ) {
      problems.push("unbenchmarked capsule must not expose benchmark definitions or samples");
    }
    if (summary["hardware_backed"] !== false) {
      problems.push("unbenchmarked simulation evidence cannot be hardware-backed");
    }
    if (
      performanceClaims.length !== 1 ||
      performanceClaims[0]?.["promotion_required"] !== false
    ) {
      problems.push("unbenchmarked capsule requires one non-promotion performance claim");
    }
    const performanceEvidenceIds = performanceClaims[0]?.["evidence_ids"];
    if (
      !Array.isArray(performanceEvidenceIds) ||
      !performanceEvidenceIds.some((id) =>
        typeof id === "string" &&
        evidenceClasses.get(id) === "performance" &&
        evidenceResults.get(id) === "pass"
      )
    ) {
      problems.push("unbenchmarked performance claim requires passing performance evidence");
    }
    const simulation = record(root["performance_simulation"], "$.performance_simulation");
    if (simulation["schema_version"] !== "genesis.candidate-simulation.v1") {
      problems.push("$.performance_simulation.schema_version is unsupported");
    }
    if (simulation["result"] !== "pass") {
      problems.push("$.performance_simulation.result must be pass");
    }
    if (simulation["comparison_permitted"] !== false) {
      problems.push("$.performance_simulation must prohibit performance comparison");
    }
    if (simulation["hardware_backed"] !== false) {
      problems.push("$.performance_simulation must not claim hardware backing");
    }
    if (simulation["candidate_id"] !== summary["accepted_candidate_id"]) {
      problems.push("$.performance_simulation.candidate_id must match the accepted candidate");
    }
    if (simulation["candidate_genome_hash"] !== summary["accepted_genome_hash"]) {
      problems.push("$.performance_simulation.candidate_genome_hash must match the accepted genome");
    }
    if (simulation["seed"] !== summary["seed"]) {
      problems.push("$.performance_simulation.seed must match the demo seed");
    }
    for (const field of [
      "candidate_genome_hash",
      "candidate_id",
      "policy_bytecode_sha256",
      "queue_policy",
      "runtime_manifest_sha256",
      "workload_path",
      "workload_sha256",
    ]) string(simulation[field], `$.performance_simulation.${field}`);
    boolean(
      simulation["deadline_order_exercised"],
      "$.performance_simulation.deadline_order_exercised",
    );
    const parseSimulationRows = (field: "events" | "raw_requests") => {
      const rows = array(simulation[field], `$.performance_simulation.${field}`);
      rows.forEach((raw, index) => {
        const item = record(raw, `$.performance_simulation.${field}[${index}]`);
        if (item["ordinal"] !== index) {
          problems.push(`$.performance_simulation.${field}[${index}].ordinal must be contiguous`);
        }
        for (const numericField of ["modeled_service_units", "policy_batch_limit"]) {
          number(
            item[numericField],
            `$.performance_simulation.${field}[${index}].${numericField}`,
          );
        }
        if (item["deadline_ms"] !== null) {
          number(item["deadline_ms"], `$.performance_simulation.${field}[${index}].deadline_ms`);
        }
        if (field === "events") {
          number(
            item["completion_units"],
            `$.performance_simulation.${field}[${index}].completion_units`,
          );
        }
      });
      return rows;
    };
    const events = parseSimulationRows("events");
    const requests = parseSimulationRows("raw_requests");
    if (events.length === 0 || events.length !== requests.length) {
      problems.push("performance simulation events must cover every raw request");
    }
    events.forEach((raw, index) => {
      const event = isRecord(raw) ? raw : {};
      const request = requests[index];
      if (
        !isRecord(request) ||
        event["deadline_ms"] !== request["deadline_ms"] ||
        event["modeled_service_units"] !== request["modeled_service_units"] ||
        event["policy_batch_limit"] !== request["policy_batch_limit"]
      ) {
        problems.push(`$.performance_simulation.events[${index}] differs from raw request`);
      }
    });
  }

  const evolution = record(root["evolution"], "$.evolution");
  string(evolution["phase"], "$.evolution.phase");
  if (evolution["active_trigger"] !== null) {
    string(evolution["active_trigger"], "$.evolution.active_trigger");
  }
  if (evolution["seed"] !== summary["seed"]) problems.push("$.evolution.seed must match the demo seed");
  const champion = record(evolution["champion"], "$.evolution.champion");
  for (const field of ["capsule_digest", "capsule_id", "genome_hash"]) {
    string(champion[field], `$.evolution.champion.${field}`);
  }
  const previousChampion = evolution["previous_champion"] === null
    ? null
    : record(evolution["previous_champion"], "$.evolution.previous_champion");
  if (
    summary["evolution_promoted"] === true &&
    previousChampion?.["capsule_digest"] !== summary["capsule_digest"]
  ) {
    problems.push("$.evolution.previous_champion must reference the original demo capsule");
  }
  const auditActions = new Set<string>();
  array(evolution["audit"], "$.evolution.audit").forEach((raw, index) => {
    const item = record(raw, `$.evolution.audit[${index}]`);
    if (item["sequence"] !== index) problems.push(`$.evolution.audit[${index}].sequence must be contiguous`);
    for (const field of ["action", "phase_after", "phase_before", "reason"]) {
      string(item[field], `$.evolution.audit[${index}].${field}`);
    }
    number(item["observed_at_ms"], `$.evolution.audit[${index}].observed_at_ms`);
    if (typeof item["action"] === "string") auditActions.add(item["action"]);
  });
  if (summary["evolution_promoted"] === true && !auditActions.has("promote")) {
    problems.push("$.summary.evolution_promoted requires a promote audit record");
  }
  if (
    summary["physical_degradation_triggered"] === true &&
    evolution["active_trigger"] !== "fabric_degradation"
  ) problems.push("$.summary physical degradation must match the active evolution trigger");
  const activeStreams = array(evolution["active_streams"], "$.evolution.active_streams");
  activeStreams.forEach((raw, index) => {
    const item = record(raw, `$.evolution.active_streams[${index}]`);
    string(item["capsule_id"], `$.evolution.active_streams[${index}].capsule_id`);
    string(item["stream_id"], `$.evolution.active_streams[${index}].stream_id`);
    boolean(
      item["externally_visible_output"],
      `$.evolution.active_streams[${index}].externally_visible_output`,
    );
  });
  if (
    summary["active_stream_preserved"] === true &&
    !activeStreams.some(
      (raw) => isRecord(raw) && raw["capsule_id"] === previousChampion?.["capsule_id"],
    )
  ) problems.push("$.summary active stream preservation must be evidenced by a pinned stream");

  const lineage = record(root["lineage"], "$.lineage");
  if (lineage["schema_version"] !== "1.0.0") problems.push("$.lineage.schema_version must be 1.0.0");
  if (lineage["seed"] !== summary["seed"]) problems.push("$.lineage.seed must match the demo seed");
  string(lineage["scope"], "$.lineage.scope");
  boolean(lineage["related_seed_retrieved"], "$.lineage.related_seed_retrieved");
  boolean(
    lineage["stale_seed_suppressed_after_invalidation"],
    "$.lineage.stale_seed_suppressed_after_invalidation",
  );
  boolean(
    lineage["performance_hypothesis_evaluated"],
    "$.lineage.performance_hypothesis_evaluated",
  );
  number(lineage["affected_evidence_count"], "$.lineage.affected_evidence_count");
  const cases = record(lineage["cases"], "$.lineage.cases");
  for (const caseName of [
    "empty_lineage",
    "unrelated_lineage",
    "related_lineage",
    "stale_dependency_before_invalidation",
    "stale_dependency_after_invalidation",
  ]) {
    const item = record(cases[caseName], `$.lineage.cases.${caseName}`);
    strings(item["lineage_seed_ids"], `$.lineage.cases.${caseName}.lineage_seed_ids`);
    number(item["lineage_seed_count"], `$.lineage.cases.${caseName}.lineage_seed_count`);
    number(item["unseeded_count"], `$.lineage.cases.${caseName}.unseeded_count`);
    boolean(
      item["reverification_required"],
      `$.lineage.cases.${caseName}.reverification_required`,
    );
  }

  if (problems.length > 0) throw new GenesisArtifactValidationError(problems);
  return input as GenesisArtifactBundle;
}
