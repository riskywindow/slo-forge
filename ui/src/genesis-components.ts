import { escapeHtml, intervalPlot } from "./charts";
import {
  GENOME_REGIONS,
  type GenesisArtifactBundle,
  type GenesisCandidateBundle,
} from "./genesis-types";

function heading(kicker: string, title: string, description: string): string {
  return `<header class="section-heading"><p class="eyebrow">${escapeHtml(kicker)}</p><h2>${escapeHtml(title)}</h2><p>${escapeHtml(description)}</p></header>`;
}

function median(values: number[]): number {
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  if (ordered.length % 2 === 1) return ordered[middle] ?? 0;
  return ((ordered[middle - 1] ?? 0) + (ordered[middle] ?? 0)) / 2;
}

function shortHash(value: string): string {
  return `${value.slice(0, 10)}…${value.slice(-6)}`;
}

function stateClass(candidate: GenesisCandidateBundle, acceptedId: string): string {
  if (candidate.candidate.candidate_id === acceptedId) return "selected";
  if (candidate.candidate.state.endsWith("REJECTED")) return "danger";
  return "";
}

function summaryPanel(bundle: GenesisArtifactBundle): string {
  const { summary } = bundle;
  const accepted = bundle.candidates.find(
    ({ candidate }) => candidate.candidate_id === summary.accepted_candidate_id,
  );
  const affected = accepted?.design.mutations.flatMap(({ regions }) => regions) ?? [];
  return `<section class="genesis-hero hero" id="genesis-overview">
    <div><p class="eyebrow">Proof-carrying synthesis · seed ${summary.seed}</p><h1>Genesis evidence</h1><p><strong>${escapeHtml(summary.package_id)}</strong> was inspected, synthesized, rejected when unsafe, corrected, and evolved. Every value below is read from the supplied demo, capsule, controller, or lineage artifact.</p></div>
    <aside class="artifact-stamp"><span>${summary.hardware_backed ? "Hardware backed" : "CPU / simulator evidence"}</span><code>${escapeHtml(shortHash(summary.capsule_digest))}</code></aside>
  </section>
  <section class="summary-strip" aria-label="Genesis run summary">
    <article class="metric-card"><p>Recovered graph</p><strong>${summary.operator_count}</strong><small>${summary.state_field_count} persistent state fields</small></article>
    <article class="metric-card"><p>Accepted candidate</p><strong>${affected.length} layers</strong><small>${escapeHtml(summary.accepted_candidate_id)}</small></article>
    <article class="metric-card"><p>Verifier rejection</p><strong>${summary.rejected_candidate_ids.length}</strong><small>${summary.minimized_counterexample_ids.length} minimized counterexample</small></article>
    <article class="metric-card"><p>Rollout</p><strong>${summary.evolution_promoted ? "Promoted" : "Not promoted"}</strong><small>${summary.active_stream_preserved ? "active stream preserved" : "stream preservation not evidenced"}</small></article>
  </section>`;
}

function genomePanel(bundle: GenesisArtifactBundle): string {
  const regionCards = GENOME_REGIONS.map((name, index) => {
    const node = bundle.genome[name].node;
    const obligations = node.proof_obligations;
    return `<article class="genome-region ${node.frozen ? "frozen" : "mutable"}">
      <span class="genome-index">${String(index + 1).padStart(2, "0")}</span>
      <div><h3>${escapeHtml(name)}</h3><p>${node.frozen ? "Frozen" : "Mutable"} · ${node.legal_rewrite_rules.length} legal rule${node.legal_rewrite_rules.length === 1 ? "" : "s"}</p><small>${obligations.length} proof obligation${obligations.length === 1 ? "" : "s"} · ${escapeHtml(obligations[0]?.minimum_level ?? "none")}</small></div>
      ${index === GENOME_REGIONS.length - 1 ? "" : '<i aria-hidden="true"></i>'}
    </article>`;
  }).join("");
  const frozen = GENOME_REGIONS.filter((name) => bundle.genome[name].node.frozen).length;
  return `<section id="genesis-genome" class="panel-section">
    ${heading("InferenceGenome", "The serving stack is one typed mutation surface", `All eight canonical regions come from ${bundle.genome.genome_id}; ${frozen} are frozen and ${GENOME_REGIONS.length - frozen} are mutable in this run.`)}
    <div class="genome-graph card" role="list" aria-label="InferenceGenome regions">${regionCards}</div>
  </section>`;
}

function searchPanel(bundle: GenesisArtifactBundle): string {
  const cards = bundle.candidates.map((item) => {
    const { candidate, design } = item;
    const transformations = design.mutations.map((mutation) => `<li><code>${escapeHtml(mutation.transformation_id)}</code><span>${escapeHtml(mutation.family)} · ${mutation.regions.map(escapeHtml).join(" + ")}</span><small>expected upside ${(mutation.expected_upside * 100).toFixed(1)}% · invalidity risk ${(mutation.invalidity_risk * 100).toFixed(1)}%</small></li>`).join("");
    const latest = candidate.lifecycle.at(-1);
    const parents = candidate.parent_candidate_ids.length === 0
      ? "baseline genome"
      : candidate.parent_candidate_ids.join(", ");
    return `<article class="card candidate-card">
      <div class="candidate-title"><span class="status-pill ${stateClass(item, bundle.summary.accepted_candidate_id)}">${escapeHtml(candidate.state)}</span><code>${escapeHtml(candidate.candidate_id)}</code></div>
      <p class="candidate-parent">← ${escapeHtml(parents)}</p>
      <ul class="transformation-list">${transformations}</ul>
      <p class="candidate-reason">${escapeHtml(latest?.reason ?? "No lifecycle evidence")}</p>
    </article>`;
  }).join("");
  return `<section id="genesis-search" class="panel-section">
    ${heading("Candidate lineage", "Every proposal remains on the evaluated frontier", "Ancestry, transformation families, affected genome regions, declared upside, risk, and terminal verifier state are shown without collapsing negative results.")}
    <div class="ancestry-root card"><span>Baseline genome</span><code>${escapeHtml(shortHash(bundle.summary.baseline_genome_hash))}</code><i aria-hidden="true"></i></div>
    <div class="candidate-grid">${cards}</div>
  </section>`;
}

function verificationPanel(bundle: GenesisArtifactBundle): string {
  const required = GENOME_REGIONS.flatMap(
    (name) => bundle.genome[name].node.proof_obligations.filter(({ required: value }) => value),
  );
  const passedClaims = bundle.capsule.claims.filter(({ result }) => result === "pass");
  const claimRows = bundle.capsule.claims.map((claim) => {
    const evidence = claim.evidence_ids
      .map((id) => bundle.capsule.evidence.find((item) => item.evidence_id === id))
      .filter((item) => item !== undefined);
    return `<tr><td>${escapeHtml(claim.category)}</td><td><span class="status-pill ${claim.result === "pass" ? "healthy" : "danger"}">${escapeHtml(claim.result)}</span></td><td>${escapeHtml(claim.level)}</td><td>${escapeHtml(evidence.map(({ issuer }) => issuer).join(", "))}</td><td>${escapeHtml(claim.scope.input_domain.join("; "))}</td></tr>`;
  }).join("");
  const counterexample = bundle.counterexamples.find(({ minimized }) => minimized) ?? bundle.counterexamples[0];
  const trace = counterexample?.payload.events.map((event) => `<li><span>${event.at_step}</span><strong>${escapeHtml(event.action)}</strong><code>${escapeHtml(event.request_id)}</code></li>`).join("") ?? "";
  const counterexampleCard = counterexample === undefined ? "" : `<article class="card counterexample-card">
    <p class="eyebrow">Minimized counterexample</p><h3>${escapeHtml(counterexample.violated_contract)}</h3>
    <p><strong>Expected:</strong> ${escapeHtml(counterexample.expected.description)}</p><p><strong>Observed:</strong> ${escapeHtml(counterexample.observed.description)}</p>
    <ol class="counterexample-trace">${trace}</ol>
    <small>${escapeHtml(counterexample.counterexample_id)} · ${escapeHtml(counterexample.scope)} · seed ${counterexample.reproduction.seed}</small>
  </article>`;
  return `<section id="genesis-verification" class="panel-section">
    ${heading("Independent evidence", "Verification is scoped per claim", `${required.length} required genome obligations are declared; the capsule separately carries ${bundle.capsule.claims.length} scoped claims, of which ${passedClaims.length} report passing evidence. This view does not turn those counts into a universal proof.`)}
    <div class="card table-card"><div class="table-heading"><h3>Capsule claim coverage</h3><p>Claim level, checker, and input scope remain visible.</p></div><div class="table-scroll"><table><thead><tr><th>Claim</th><th>Result</th><th>Level</th><th>Issuer</th><th>Declared input domain</th></tr></thead><tbody>${claimRows}</tbody></table></div></div>
    ${counterexampleCard}
  </section>`;
}

function benchmarkPanel(bundle: GenesisArtifactBundle): string {
  const benchmark = bundle.capsule.benchmarks[0];
  if (benchmark === undefined) {
    const claim = bundle.capsule.claims.find(
      ({ category, result }) => category === "performance" && result === "pass",
    );
    const simulation = bundle.performance_simulation;
    const assumptions = claim?.scope.assumptions.join("; ") ?? "No performance scope supplied";
    const completion = simulation?.events.at(-1)?.completion_units;
    return `<section id="genesis-benchmark" class="panel-section">
      ${heading("Performance evidence", "No accepted performance claim", claim?.statement ?? "This capsule carries no accepted benchmark comparison.")}
      <article class="card benchmark-summary">
        <span class="status-pill">Simulation only</span>
        <dl class="definition-grid"><div><dt>Comparison</dt><dd>Not permitted</dd></div><div><dt>Hardware backed</dt><dd>No</dd></div><div><dt>Raw benchmark samples</dt><dd>None</dd></div><div><dt>Simulation events</dt><dd>${simulation?.events.length ?? 0}</dd></div><div><dt>Queue policy</dt><dd>${escapeHtml(simulation?.queue_policy ?? "unavailable")}</dd></div><div><dt>Completion units</dt><dd>${completion ?? "unavailable"}</dd></div></dl>
        <p>${escapeHtml(assumptions)}. Capsule eligibility: ${bundle.summary.capsule_local_evolution_eligible ? "local evolution" : "not local-evolution eligible"}; ${bundle.summary.capsule_external_production_eligible ? "external production" : "not external-production eligible"}.</p>
      </article>
    </section>`;
  }
  const baselineSamples = bundle.baseline_samples;
  const candidateSamples = bundle.candidate_samples;
  const definition = bundle.benchmark_definition;
  if (baselineSamples === null || candidateSamples === null || definition === null) {
    return `<section id="genesis-benchmark" class="panel-section">${heading("Performance evidence", "Benchmark evidence unavailable", "The capsule declares a benchmark but the required definition or raw samples are absent; no performance result is rendered.")}</section>`;
  }
  const baselineMedian = median(baselineSamples.samples.map(({ value }) => value));
  const candidateMedian = median(candidateSamples.samples.map(({ value }) => value));
  const plot = intervalPlot([{
    label: "Candidate",
    lower: benchmark.summary.confidence_low,
    point: benchmark.summary.median,
    unit: benchmark.summary.unit,
    upper: benchmark.summary.confidence_high,
  }]);
  return `<section id="genesis-benchmark" class="panel-section">
    ${heading("Performance evidence", "Raw samples and uncertainty stay attached", `${definition.benchmark_id} uses ${definition.repetitions} randomized-order repetitions after ${definition.warmup} warmup; it is ${definition.hardware_backed ? "hardware-backed" : "a deterministic simulator result, not hardware timing"}.`)}
    <div class="split-grid benchmark-evidence"><article class="card chart-card">${plot}</article><article class="card benchmark-summary">
      <dl class="definition-grid"><div><dt>Baseline median</dt><dd>${baselineMedian.toFixed(3)}</dd></div><div><dt>Candidate median</dt><dd>${candidateMedian.toFixed(3)}</dd></div><div><dt>Effect size</dt><dd>${(benchmark.summary.effect_size * 100).toFixed(1)}%</dd></div><div><dt>Confidence</dt><dd>${(definition.confidence * 100).toFixed(0)}%</dd></div><div><dt>Raw samples</dt><dd>${baselineSamples.samples.length + candidateSamples.samples.length}</dd></div><div><dt>Unit</dt><dd>${escapeHtml(definition.unit)}</dd></div></dl>
      <p>${escapeHtml(bundle.capsule.hardware.restrictions.join("; "))}</p>
    </article></div>
  </section>`;
}

function evolutionPanel(bundle: GenesisArtifactBundle): string {
  const { evolution } = bundle;
  const timeline = evolution.audit.map((event) => {
    const css = event.action === "promote" ? "restored" : event.action === "begin_evolution" && event.sequence > 2 ? "fault" : "";
    return `<li class="${css}"><time>${event.observed_at_ms} ms</time><div><strong>${escapeHtml(event.action)}</strong><p>${escapeHtml(event.reason)}</p><code>${escapeHtml(event.phase_before)} → ${escapeHtml(event.phase_after)}</code></div></li>`;
  }).join("");
  const challenger = evolution.challengers[0];
  return `<section id="genesis-evolution" class="panel-section">
    ${heading("Champion–challenger", "Promotion retains evidence and stream ownership", `The controller audit is ordered by observed time and ends in ${evolution.phase}; current trigger: ${evolution.active_trigger ?? "none"}.`)}
    <div class="rollout-grid"><article class="card rollout-identity"><p class="eyebrow">Previous champion</p><h3>${escapeHtml(evolution.previous_champion?.capsule_id ?? "none")}</h3><code>${escapeHtml(shortHash(evolution.previous_champion?.capsule_digest ?? ""))}</code></article><article class="card rollout-identity"><p class="eyebrow">Current champion</p><h3>${escapeHtml(evolution.champion.capsule_id)}</h3><code>${escapeHtml(shortHash(evolution.champion.capsule_digest))}</code></article><article class="card rollout-identity"><p class="eyebrow">Challenger state</p><h3>${escapeHtml(challenger?.status ?? "none")}</h3><small>${evolution.active_streams.length} active stream${evolution.active_streams.length === 1 ? "" : "s"} pinned to prior capsules</small></article></div>
    <div class="card rollout-timeline"><ol class="fabric-timeline">${timeline}</ol></div>
  </section>`;
}

function lineagePanel(bundle: GenesisArtifactBundle): string {
  const entries = Object.entries(bundle.lineage.cases);
  const rows = entries.map(([name, item]) => {
    const total = item.lineage_seed_count + item.unseeded_count;
    const percent = total === 0 ? 0 : (item.lineage_seed_count / total) * 100;
    return `<div class="lineage-case"><code>${escapeHtml(name)}</code><span><i style="width:${percent.toFixed(1)}%"></i></span><strong>${item.lineage_seed_count} transferred / ${item.unseeded_count} unseeded</strong><small>${item.reverification_required ? "reverification required" : "reverification not declared"}</small></div>`;
  }).join("");
  return `<section id="genesis-lineage" class="panel-section">
    ${heading("Optimization lineage", "Transfer is useful only while its evidence remains valid", bundle.lineage.scope)}
    <div class="card lineage-cases">${rows}</div>
    <div class="lineage-callouts"><article class="card"><span class="status-pill healthy">Retrieved</span><h3>${bundle.lineage.related_seed_retrieved ? "Related transformation found" : "No related seed found"}</h3><p>${escapeHtml(bundle.lineage.cases.related_lineage.lineage_seed_ids.join(", ") || "No transferred transformation")}</p></article><article class="card"><span class="status-pill ${bundle.lineage.stale_seed_suppressed_after_invalidation ? "healthy" : "danger"}">Invalidation</span><h3>${bundle.lineage.affected_evidence_count} evidence records affected</h3><p>${bundle.lineage.stale_seed_suppressed_after_invalidation ? "The stale transformation disappeared after dependency invalidation." : "Stale transfer was not suppressed."}</p></article></div>
    ${bundle.lineage.performance_hypothesis_evaluated ? "" : '<p class="evidence-note">This lineage artifact evaluates retrieval and invalidation mechanics only; it makes no speedup claim.</p>'}
  </section>`;
}

export function renderGenesisDashboard(bundle: GenesisArtifactBundle): string {
  return `<main id="dashboard" class="genesis-dashboard">${summaryPanel(bundle)}${genomePanel(bundle)}${searchPanel(bundle)}${verificationPanel(bundle)}${benchmarkPanel(bundle)}${evolutionPanel(bundle)}${lineagePanel(bundle)}</main>`;
}
