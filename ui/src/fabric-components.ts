import { escapeHtml } from "./charts";
import {
  counterfactualRanking,
  fabricProfileLabel,
  formatBytes,
  hostIds,
  hotExperts,
  kvLinkEvidence,
  paretoCandidates,
  percent,
  placementsByHost,
  representativeFabricMeasurements,
  requestMetricComparisons,
  resourceHotspots,
  topologyInventory,
  workerPools,
} from "./fabric-transforms";
import type { FabricArtifactBundle, FabricMeasurement } from "./fabric-types";

function sectionHeading(kicker: string, title: string, description: string): string {
  return `<header class="section-heading"><p class="eyebrow">${escapeHtml(kicker)}</p><h2>${escapeHtml(title)}</h2><p>${escapeHtml(description)}</p></header>`;
}

function metricCard(name: string, value: string, detail: string): string {
  return `<article class="metric-card"><p>${escapeHtml(name)}</p><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small></article>`;
}

function physicalTopologySvg(bundle: FabricArtifactBundle): string {
  const hosts = hostIds(bundle);
  const byHost = placementsByHost(bundle);
  const hostWidth = 450;
  const height = 430;
  const width = Math.max(960, hosts.length * (hostWidth + 20));
  const hostMarkup = hosts.map((host, hostIndex) => {
    const x = 20 + hostIndex * (hostWidth + 20);
    const bindings = byHost.get(host) ?? [];
    const numaDomains = [...new Set(bindings.map((binding) => binding.numa_domain_id))].sort();
    const cards = bindings.map((binding, index) => {
      const column = index % 4;
      const row = Math.floor(index / 4);
      const cardX = x + 24 + column * 100;
      const cardY = 105 + row * 70;
      const roleClass = binding.worker_role === "prefill" ? "prefill-rank" : "decode-rank";
      return `<g class="fabric-rank ${roleClass}" role="graphics-symbol" tabindex="0" aria-label="Rank ${binding.rank_id}, ${binding.worker_role}, ${binding.gpu_id}, ${binding.numa_domain_id}, ${binding.nic_id ?? "no NIC"}">
        <rect x="${cardX}" y="${cardY}" width="86" height="50" rx="4" />
        <text x="${cardX + 8}" y="${cardY + 20}">R${binding.rank_id} · ${escapeHtml(binding.worker_role.slice(0, 3))}</text>
        <text class="sub" x="${cardX + 8}" y="${cardY + 37}">${escapeHtml(binding.gpu_id.split("/").at(-1) ?? binding.gpu_id)}</text>
      </g>`;
    }).join("");
    const numa = numaDomains.map((domain, index) =>
      `<text class="numa-label" x="${x + 25 + index * 200}" y="90">${escapeHtml(domain)} · CPU ${escapeHtml(bindings.find((item) => item.numa_domain_id === domain)?.process_cpu_affinity ?? "unknown")}</text>`,
    ).join("");
    const nics = [...new Set(bindings.map((binding) => binding.nic_id).filter((value): value is string => value !== null))];
    return `<g class="fabric-host">
      <rect x="${x}" y="45" width="${hostWidth}" height="260" rx="8" />
      <text class="host-label" x="${x + 20}" y="72">${escapeHtml(host)}</text>
      ${numa}${cards}
      <text class="nic-label" x="${x + 22}" y="287">${escapeHtml(nics.join(" · "))}</text>
    </g>`;
  }).join("");
  const rails = bundle.topology.nodes.filter((node) => node.kind === "network_rail");
  const railMarkup = rails.map((rail, index) => {
    const y = 340 + index * 42;
    const health = rail.health ?? "unknown";
    return `<g class="fabric-rail" role="graphics-symbol" tabindex="0" aria-label="Network rail ${escapeHtml(rail.node_id)}, health ${escapeHtml(health)}">
      <line x1="70" x2="${width - 70}" y1="${y}" y2="${y}" />
      <text x="80" y="${y - 9}">${escapeHtml(rail.node_id)} · ${escapeHtml(health)}</text>
    </g>`;
  }).join("");
  return `<svg class="fabric-topology chart" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="fabric-topology-title fabric-topology-desc">
    <title id="fabric-topology-title">Physical topology and rank placement</title>
    <desc id="fabric-topology-desc">${hosts.length} hosts with rank-to-GPU, NUMA, NIC, and network rail assignments. Green ranks run prefill and blue ranks run decode.</desc>
    ${hostMarkup}${railMarkup}
  </svg>`;
}

function profileCurveSvg(measurements: FabricMeasurement[], profileLabel: string): string {
  const width = 940;
  const height = 270;
  const margin = { bottom: 45, left: 72, right: 25, top: 25 };
  const maxDuration = Math.max(...measurements.map((item) => item.confidence_high_us), 1);
  const barHeight = Math.min(28, (height - margin.top - margin.bottom) / Math.max(measurements.length, 1));
  const scale = (value: number): number => margin.left + (value / maxDuration) * (width - margin.left - margin.right);
  const rows = measurements.map((item, index) => {
    const y = margin.top + index * (barHeight + 8);
    const label = item.measurement_id.split("-b")[0] ?? item.primitive;
    return `<g class="profile-row" role="graphics-symbol" tabindex="0" aria-label="${escapeHtml(label)}, median ${item.summary_median_us.toFixed(3)} microseconds, confidence interval ${item.confidence_low_us.toFixed(3)} to ${item.confidence_high_us.toFixed(3)} microseconds">
      <text class="row-label" x="${margin.left - 8}" y="${y + barHeight * 0.7}" text-anchor="end">${escapeHtml(label.replaceAll("_", " "))}</text>
      <line class="profile-interval" x1="${scale(item.confidence_low_us)}" x2="${scale(item.confidence_high_us)}" y1="${y + barHeight / 2}" y2="${y + barHeight / 2}" />
      <rect class="profile-bar" x="${margin.left}" y="${y}" width="${Math.max(scale(item.summary_median_us) - margin.left, 1)}" height="${barHeight}" rx="2" />
      <text class="value-label" x="${scale(item.summary_median_us) + 7}" y="${y + barHeight * 0.7}">${item.summary_median_us.toFixed(1)} µs</text>
    </g>`;
  }).join("");
  return `<svg class="chart fabric-profile-chart" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="profile-title profile-desc">
    <title id="profile-title">Representative ${escapeHtml(profileLabel)} fabric operation durations</title>
    <desc id="profile-desc">${escapeHtml(profileLabel)} median durations with bootstrap confidence intervals from retained raw samples.</desc>
    ${rows}<text class="axis-label" x="${width - margin.right}" y="${height - 10}" text-anchor="end">duration · microseconds</text>
  </svg>`;
}

function paretoSvg(bundle: FabricArtifactBundle): string {
  const candidates = paretoCandidates(bundle);
  const width = 700;
  const height = 330;
  const margin = { bottom: 50, left: 60, right: 30, top: 25 };
  const maxCost = Math.max(...candidates.map((item) => item.cost_per_million_tokens), 1);
  const maxLatency = Math.max(...candidates.map((item) => item.p95_ttft_ms), 1);
  const selectedId = bundle.physical_plan.optimizer_history.find((event) => event.decision === "select")?.candidate_id;
  const points = candidates.map((candidate) => {
    const x = margin.left + (candidate.cost_per_million_tokens / maxCost) * (width - margin.left - margin.right);
    const y = height - margin.bottom - (candidate.p95_ttft_ms / maxLatency) * (height - margin.top - margin.bottom);
    const selected = candidate.candidate_id === selectedId;
    return `<g class="fabric-pareto-point ${selected ? "selected" : ""}" role="graphics-symbol" tabindex="0" aria-label="${escapeHtml(candidate.candidate_id)}, cost ${candidate.cost_per_million_tokens.toFixed(2)} dollars per million tokens, p95 TTFT ${candidate.p95_ttft_ms.toFixed(2)} milliseconds, communication ${candidate.communication_us.toFixed(2)} microseconds">
      ${selected ? `<circle class="selection-ring" cx="${x}" cy="${y}" r="11" />` : ""}<circle cx="${x}" cy="${y}" r="6" />
    </g>`;
  }).join("");
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="fabric-pareto-title fabric-pareto-desc">
    <title id="fabric-pareto-title">Physical-plan Pareto frontier</title>
    <desc id="fabric-pareto-desc">Candidate cost per million tokens versus predicted p95 time to first token. The selected physical candidate has a white ring.</desc>
    <line class="axis-line" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" />
    <line class="axis-line" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" />
    ${points}<text class="axis-label" x="${width - margin.right}" y="${height - 12}" text-anchor="end">cost · USD / million tokens</text>
    <text class="axis-label" x="${margin.left + 5}" y="${margin.top + 4}">p95 TTFT · ms</text>
  </svg>`;
}

function hero(bundle: FabricArtifactBundle): string {
  const manifest = bundle.manifest;
  const profileLabel = fabricProfileLabel(bundle);
  return `<section class="hero fabric-hero" aria-labelledby="page-title">
    <div><p class="eyebrow">Topology-aware execution evidence</p><h1 id="page-title">SLOForge Fabric</h1><p><strong>${escapeHtml(bundle.physical_plan.plan_id)}</strong> maps ${bundle.physical_plan.rank_placement.bindings.length} ranks onto ${hostIds(bundle).length} hosts and validates a causal recovery against deterministic simulator evidence.</p></div>
    <div class="artifact-stamp"><span>${escapeHtml(manifest.schema_version)} · seed ${manifest.seed} · ${escapeHtml(profileLabel)}</span><code>${escapeHtml(manifest.topology_fingerprint.slice(0, 24))}…</code></div>
  </section>
  <section class="summary-strip" aria-label="Fabric demonstration outcome">
    ${metricCard("Healthy p95 TTFT", `${manifest.healthy.p95_ttft_ms.toFixed(1)} ms`, `SLO ≤ ${manifest.p95_ttft_slo_ms.toFixed(1)} ms`)}
    ${metricCard("Degraded p95 TTFT", `${manifest.degraded.p95_ttft_ms.toFixed(1)} ms`, manifest.ground_truth_faults.join(" + "))}
    ${metricCard("Restored p95 TTFT", `${manifest.restored.p95_ttft_ms.toFixed(1)} ms`, manifest.restored_slo_attained ? "SLO restored" : "SLO remains violated")}
    ${metricCard("Recovery", manifest.recovery_final_state, `${manifest.counterfactuals_evaluated} counterfactuals evaluated`)}
  </section>`;
}

function topologyPanel(bundle: FabricArtifactBundle): string {
  const inventory = topologyInventory(bundle);
  const edges = bundle.topology.edges;
  return `<section id="fabric-topology" class="panel-section">
    ${sectionHeading("Physical compiler", "Topology graph and rank placement", "The graph is reconstructed from the versioned TopologyGraph and PhysicalExecutionPlan—not from display-time assumptions.")}
    <article class="card chart-card">${physicalTopologySvg(bundle)}<div class="fabric-legend"><span><i class="rank-key prefill"></i>prefill</span><span><i class="rank-key decode"></i>decode</span>${inventory.connections.map((entry) => `<span>${escapeHtml(entry.kind)} · ${entry.count}</span>`).join("")}</div></article>
    <div class="fabric-inventory">${inventory.nodes.map((entry) => metricCard(entry.kind.replaceAll("_", " "), entry.count.toString(), "discovered nodes")).join("")}</div>
    <div class="card table-card"><div class="table-heading"><h3>Physical topology edges</h3><p>${edges.length} explicit GPU, PCIe, NVLink, NIC, NUMA, and rail relationships from TopologyGraph</p></div><div class="table-scroll"><table><thead><tr><th>Connection</th><th>Source</th><th>Target</th><th>Contention / sharing</th><th>Bandwidth</th><th>Health</th></tr></thead><tbody>${edges.map((edge) => `<tr><td>${escapeHtml(edge.connection)}</td><td><code>${escapeHtml(edge.source_node_id)}</code></td><td><code>${escapeHtml(edge.target_node_id)}</code></td><td>${escapeHtml(edge.contention_domain)} · ${escapeHtml(edge.sharing_group)}</td><td>${edge.theoretical_bandwidth_gbps === null ? "unknown" : `${edge.theoretical_bandwidth_gbps.toFixed(1)} Gbps`} · ${edge.bandwidth_curve_gbps.length} curve points</td><td>${escapeHtml(edge.health)}</td></tr>`).join("")}</tbody></table></div></div>
  </section>`;
}

function placementPanel(bundle: FabricArtifactBundle): string {
  const pools = workerPools(bundle);
  const experts = hotExperts(bundle);
  const ranks = bundle.physical_plan.rank_placement.bindings;
  return `<section id="fabric-placement" class="panel-section">
    ${sectionHeading("Execution mapping", "Parallel groups, experts, and affinity", "Ranks retain GPU, NUMA, CPU, NIC, rail, replica, worker-role, and fault-domain bindings in the physical IR.")}
    <div class="split-grid">
      <article class="card fabric-pools"><h3>Prefill/decode pools</h3>${pools.map((pool) => `<div class="pool-row"><span>${escapeHtml(pool.kind)}</span><div>${pool.ranks.map((rank) => `<i title="rank ${rank}">R${rank}</i>`).join("")}</div></div>`).join("")}<dl class="definition-grid compact"><div><dt>Tensor parallel</dt><dd>${bundle.physical_plan.parallelism.tensor_parallel_degree}</dd></div><div><dt>Pipeline parallel</dt><dd>${bundle.physical_plan.parallelism.pipeline_parallel_degree}</dd></div><div><dt>Expert parallel</dt><dd>${bundle.physical_plan.parallelism.expert_parallel_degree}</dd></div><div><dt>Data parallel</dt><dd>${bundle.physical_plan.parallelism.data_parallel_degree}</dd></div></dl></article>
      <article class="card"><h3>Hot expert assignments</h3><div class="expert-list">${experts.map((expert) => `<div><code>${escapeHtml(expert.expert)}</code><span class="expert-load"><i style="width:${Math.min(expert.load * 100, 100).toFixed(1)}%"></i></span><small>${percent(expert.load)} · ${expert.ranks.map((rank) => `R${rank}`).join(", ")}</small></div>`).join("")}</div><p class="evidence-note">Strategy: ${escapeHtml(bundle.physical_plan.expert_placement.hot_expert_strategy)} · ${bundle.physical_plan.expert_placement.assignments.length} assignments</p></article>
    </div>
    <div class="card table-card"><div class="table-heading"><h3>Rank bindings</h3><p>${ranks.length} explicit process placements</p></div><div class="table-scroll"><table><thead><tr><th>Rank</th><th>Role</th><th>GPU</th><th>NUMA / CPU</th><th>NIC / rail</th><th>Fault domain</th></tr></thead><tbody>${ranks.map((rank) => `<tr><td>R${rank.rank_id}</td><td>${escapeHtml(rank.worker_role)}</td><td><code>${escapeHtml(rank.gpu_id)}</code></td><td>${escapeHtml(rank.numa_domain_id)} · ${escapeHtml(rank.process_cpu_affinity)}</td><td>${escapeHtml(rank.nic_id ?? "unknown")} · ${escapeHtml(rank.network_rail_id ?? "unknown")}</td><td>${escapeHtml(rank.fault_domain)}</td></tr>`).join("")}</tbody></table></div></div>
  </section>`;
}

function communicationPanel(bundle: FabricArtifactBundle): string {
  const measurements = representativeFabricMeasurements(bundle);
  const profileLabel = fabricProfileLabel(bundle);
  const routes = bundle.physical_plan.kv_transfer.routes;
  const linkEvidence = kvLinkEvidence(bundle);
  const collectives = bundle.physical_plan.collectives.operations;
  const overlaps = bundle.physical_plan.communication_overlap.windows;
  return `<section id="fabric-communication" class="panel-section">
    ${sectionHeading("Communication plan", "Collectives, KV flows, and overlap", "Selected operations and transfer paths are shown separately from calibrated measurements; absent compiler operations remain explicit.")}
    <div class="card chart-card">${profileCurveSvg(measurements, profileLabel)}</div>
    <div class="fabric-flow-grid">
      <article class="card"><h3>Collective flows</h3>${collectives.length === 0 ? `<p class="empty">No explicit collective operation is selected for this EP=${bundle.physical_plan.parallelism.expert_parallel_degree} plan. Collective calibration remains in the profile.</p>` : collectives.map((operation) => `<div class="flow-record"><strong>${escapeHtml(operation.operation)}</strong><span>${escapeHtml(operation.algorithm)} · ${escapeHtml(operation.transport)} · ${operation.channel_count} channels</span><small>${operation.participating_ranks.map((rank) => `R${rank}`).join(" → ")} · ${operation.rail_ids.map((rail) => escapeHtml(rail)).join(" + ")} · ${operation.expected_duration_us.toFixed(2)} µs</small></div>`).join("")}</article>
      <article class="card"><h3>KV transfer flows</h3>${routes.map((route) => `<div class="flow-record"><strong>${escapeHtml(route.route_id)}</strong><span>${escapeHtml(route.transport_adapter)} · ${formatBytes(route.chunk_bytes)} chunks</span><small>${route.producer_rank_ids.length} producers → ${route.consumer_rank_ids.length} consumers · ${route.expected_latency_us.toFixed(2)} µs · ${route.overlap_with_decode ? "overlapped" : "serialized"}</small><code>${escapeHtml(route.edge_path.join(" → "))}</code></div>`).join("")}</article>
      <article class="card"><h3>Compute–communication overlap</h3>${overlaps.length === 0 ? `<p class="empty">No explicit overlap window was emitted; KV routes independently declare ${routes.filter((route) => route.overlap_with_decode).length} decode-overlap path(s).</p>` : overlaps.map((overlap) => `<div class="flow-record"><strong>${escapeHtml(overlap.window_id)}</strong><span>${escapeHtml(overlap.compute_operation_id)} + ${escapeHtml(overlap.communication_operation_id)}</span><small>${percent(overlap.expected_overlap_fraction)} expected overlap · fallback ${escapeHtml(overlap.fallback_serialization)}</small></div>`).join("")}</article>
    </div>
    <div class="card table-card"><div class="table-heading"><h3>Profiled versus predicted KV link curves</h3><p>The link model is reconstructed from each selected edge's size-dependent latency and bandwidth curves. Profile mode: ${escapeHtml(profileLabel)}.</p></div><div class="table-scroll"><table><thead><tr><th>Route</th><th>Chunk</th><th>Link-curve prediction</th><th>Compiled route</th><th>Profile median</th><th>Profile interval</th><th>Provenance</th></tr></thead><tbody>${linkEvidence.map((evidence) => `<tr><td><code>${escapeHtml(evidence.routeId)}</code></td><td>${formatBytes(evidence.messageBytes)}</td><td>${evidence.predictedLinkLatencyUs.toFixed(3)} µs</td><td>${evidence.compiledLatencyUs.toFixed(3)} µs</td><td>${evidence.measuredLatencyUs === null ? "unavailable" : `${evidence.measuredLatencyUs.toFixed(3)} µs`}</td><td>${evidence.measuredConfidenceLowUs === null || evidence.measuredConfidenceHighUs === null ? "unavailable" : `${evidence.measuredConfidenceLowUs.toFixed(3)}–${evidence.measuredConfidenceHighUs.toFixed(3)} µs`}</td><td>${evidence.measurementId === null ? "no matching profile sample" : `${escapeHtml(evidence.measurementId)} · ${evidence.rawSampleCount} raw samples`}</td></tr>`).join("")}</tbody></table></div></div>
    <div class="card table-card"><div class="table-heading"><h3>Representative ${escapeHtml(profileLabel)} fabric profile</h3><p>${measurements.length} of ${bundle.fabric_profile.measurements.length} shape-nearest measurements are displayed; median, p95, and confidence are computed from retained raw samples.</p></div><div class="table-scroll"><table><thead><tr><th>Measurement</th><th>Shape</th><th>Median</th><th>p95</th><th>Confidence interval</th><th>Samples</th><th>Transport</th></tr></thead><tbody>${measurements.map((measurement) => `<tr><td><code>${escapeHtml(measurement.measurement_id)}</code></td><td>${formatBytes(measurement.message_bytes)} · r${measurement.rank_count} · c${measurement.concurrency}</td><td>${measurement.summary_median_us.toFixed(3)} µs</td><td>${measurement.summary_p95_us.toFixed(3)} µs</td><td>${measurement.confidence_low_us.toFixed(3)}–${measurement.confidence_high_us.toFixed(3)} µs</td><td>${measurement.samples.length} + ${measurement.warmup_count} warmup</td><td>${escapeHtml(measurement.transport)}</td></tr>`).join("")}</tbody></table></div></div>
  </section>`;
}

function predictionPanel(bundle: FabricArtifactBundle): string {
  const metrics = requestMetricComparisons(bundle);
  const hotspots = resourceHotspots(bundle);
  return `<section id="fabric-evidence" class="panel-section">
    ${sectionHeading("Digital twin", "Predicted, observed, degraded, and restored", "Physical-plan intervals remain distinct from request metrics measured by deterministic replay; resource utilization exposes the current contention surface.")}
    <div class="card table-card"><div class="table-scroll"><table><thead><tr><th>Request metric</th><th>Compiler prediction</th><th>Healthy replay</th><th>Degraded replay</th><th>Restored replay</th></tr></thead><tbody>${metrics.map((metric) => `<tr><td>${escapeHtml(metric.name.replaceAll("_", " "))}</td><td>${metric.predicted === null ? "not modeled" : `${metric.predicted.toFixed(2)} ms`}</td><td>${metric.healthy.toFixed(2)} ms</td><td class="warning-text">${metric.degraded.toFixed(2)} ms</td><td>${metric.restored.toFixed(2)} ms</td></tr>`).join("")}</tbody></table></div></div>
    <div class="split-grid fabric-evidence-grid"><article class="card chart-card">${paretoSvg(bundle)}<p class="evidence-note">${bundle.optimizer.pareto_frontier.length} Pareto candidates · ${escapeHtml(bundle.optimizer.strategy)} strategy · ${bundle.optimizer.simulator_calls} compiler simulator calls recorded</p></article><article class="card"><h3>Resource hotspots</h3><div class="hotspot-list">${hotspots.map((hotspot) => `<div><code title="${escapeHtml(hotspot.resource)}">${escapeHtml(hotspot.resource)}</code><span><i style="width:${Math.min(hotspot.degraded * 100, 100).toFixed(1)}%"></i></span><small>healthy ${percent(hotspot.healthy)} · degraded ${percent(hotspot.degraded)} · restored ${percent(hotspot.restored)}</small></div>`).join("")}</div></article></div>
  </section>`;
}

function autopsyPanel(bundle: FabricArtifactBundle): string {
  const diagnosis = bundle.diagnosis;
  const hypotheses = diagnosis.hypotheses;
  const selected = bundle.counterfactuals.selected_scenario_id;
  const counterfactuals = counterfactualRanking(bundle);
  return `<section id="fabric-autopsy" class="panel-section">
    ${sectionHeading("Causal debugger", "Fault, diagnosis, and counterfactual repair", "Autopsy aligns the first divergence with physical evidence, records contradictions, and tests repair hypotheses through simulator replay.")}
    <div class="autopsy-summary card"><div><p class="eyebrow">Current bottleneck</p><h3>${escapeHtml(bundle.manifest.diagnosis.replaceAll("_", " "))}</h3><p>Confidence ${percent(bundle.manifest.diagnosis_confidence)} · first divergence ${diagnosis.first_divergence_ns === null ? "unknown" : `${(diagnosis.first_divergence_ns / 1000).toFixed(3)} µs`}</p></div><div><p class="eyebrow">Injected ground truth</p>${bundle.manifest.ground_truth_faults.map((fault) => `<span class="status-pill danger">${escapeHtml(fault.replaceAll("_", " "))}</span>`).join(" ")}</div></div>
    <div class="causal-grid">
      <article class="card"><h3>Hypothesis evidence</h3>${hypotheses.slice(0, 8).map((hypothesis, index) => `<div class="hypothesis ${hypothesis.rejected_reason === null ? "supported" : "rejected"}"><span class="causal-index">${index + 1}</span><div><strong>${escapeHtml(hypothesis.kind.replaceAll("_", " "))}</strong><small>${percent(hypothesis.confidence)} · target ${escapeHtml(hypothesis.target)} · ${hypothesis.supporting_evidence.length} support · ${hypothesis.contradicting_evidence.length} contradiction</small><p>${escapeHtml(hypothesis.supporting_evidence[0]?.explanation ?? hypothesis.rejected_reason ?? "Evidence retained in diagnosis artifact")}</p></div></div>`).join("")}</article>
      <article class="card"><h3>Counterfactual repairs</h3>${counterfactuals.map((evaluation) => `<div class="counterfactual ${evaluation.scenario.scenario_id === selected ? "selected" : ""}"><div><strong>${escapeHtml(evaluation.scenario.scenario_id)}</strong>${evaluation.scenario.scenario_id === selected ? '<span class="status-pill selected">selected</span>' : ""}</div><p>${escapeHtml(evaluation.scenario.rationale)}</p><span class="improvement-bar"><i style="width:${Math.min(evaluation.expected_improvement_ms / Math.max(...counterfactuals.map((item) => item.expected_improvement_ms), 1) * 100, 100).toFixed(1)}%"></i></span><small>${evaluation.expected_improvement_ms.toFixed(1)} ms expected · ${evaluation.lower_improvement_ms.toFixed(1)}–${evaluation.upper_improvement_ms.toFixed(1)} ms · ${escapeHtml(evaluation.status)}</small></div>`).join("")}</article>
    </div>
  </section>`;
}

function recoveryPanel(bundle: FabricArtifactBundle): string {
  const recovery = bundle.recovery_plan;
  const audit = bundle.recovery_execution.audit.filter((record) => record.event === "transition" || record.event === "wait");
  return `<section id="fabric-recovery" class="panel-section">
    ${sectionHeading("Self-healing runtime", "Shadow, canary, promotion, and rollback guards", `The recovery audit is an idempotent state-machine record. Started streams are ${recovery.traffic_migration.preserve_started_streams ? "preserved" : "not preserved"} and external mutation is ${recovery.external_mutation_authorized ? "authorized" : "disabled"}.`)}
    <div class="summary-strip">
      ${metricCard("Final state", bundle.recovery_execution.state, `${recovery.actions.length} guarded actions`)}
      ${metricCard("Shadow", percent(recovery.traffic_migration.shadow_fraction), `${recovery.traffic_migration.minimum_shadow_samples} minimum samples`)}
      ${metricCard("Canary", percent(recovery.traffic_migration.canary_fraction), `${recovery.traffic_migration.minimum_canary_samples} minimum samples`)}
      ${metricCard("Rollback guards", `${recovery.rollback_criteria.length} armed`, `${bundle.recovery_execution.action_attempts.filter((attempt) => !attempt.succeeded).length} failed action attempts`)}
    </div>
    <article class="card recovery-state-machine" aria-label="Recovery state machine audit">${audit.map((record, index) => `<div class="recovery-state ${record.state_after === "COMPLETED" ? "complete" : ""}"><span>${record.at_ms.toFixed(0)} ms</span><strong>${escapeHtml(record.state_after)}</strong><small>${escapeHtml(record.reason)}</small>${index < audit.length - 1 ? '<i aria-hidden="true"></i>' : ""}</div>`).join("")}</article>
    <article class="card"><h3>Abort and rollback criteria</h3><div class="flow-record">${[...recovery.abort_criteria.map((criterion) => ({ ...criterion, kind: "abort" })), ...recovery.rollback_criteria.map((criterion) => ({ ...criterion, kind: "rollback" }))].map((criterion) => `<span><strong>${escapeHtml(criterion.kind)}</strong> · ${escapeHtml(criterion.metric)} ${escapeHtml(criterion.comparator)} ${criterion.threshold} over ${criterion.window_seconds.toFixed(0)} s</span>`).join("")}</div></article>
    <div class="card table-card"><div class="table-heading"><h3>Recovery actions and safety</h3><p>External mutation authorized: ${recovery.external_mutation_authorized ? "yes" : "no"} · started streams preserved: ${recovery.traffic_migration.preserve_started_streams ? "yes" : "no"}</p></div><div class="table-scroll"><table><thead><tr><th>Order</th><th>Action</th><th>Scope</th><th>Targets</th><th>Timeout</th><th>Attempt</th></tr></thead><tbody>${recovery.actions.map((action) => { const attempt = bundle.recovery_execution.action_attempts.find((item) => item.action_id === action.action_id); return `<tr><td>${action.order}</td><td>${escapeHtml(action.kind)}</td><td>${escapeHtml(action.scope)}</td><td>${action.target_ids.map((target) => escapeHtml(target)).join(" · ")}</td><td>${action.timeout_seconds.toFixed(0)} s</td><td><span class="status-pill ${attempt?.succeeded === true ? "healthy" : "danger"}">${attempt?.succeeded === true ? "passed" : "not passed"}</span></td></tr>`; }).join("")}</tbody></table></div></div>
  </section>`;
}

function timelinePanel(bundle: FabricArtifactBundle): string {
  return `<section id="fabric-timeline" class="panel-section">
    ${sectionHeading("Artifact timeline", "From SLO regression to restoration", "Every timestamp and detail is read from timeline.json and links to its originating evidence artifact.")}
    <ol class="fabric-timeline">${bundle.timeline.map((event) => `<li class="${event.event === "SLO_REGRESSION" ? "fault" : event.event === "SLO_RESTORED" || event.event === "COMPLETED" ? "restored" : ""}"><time>${event.at_ms.toFixed(3)} ms</time><div><strong>${escapeHtml(event.event.replaceAll("_", " "))}</strong><p>${escapeHtml(event.detail)}</p><code>${escapeHtml(event.evidence_uri)}</code></div></li>`).join("")}</ol>
  </section>`;
}

export function renderFabricDashboard(bundle: FabricArtifactBundle): string {
  return `<main id="dashboard" class="fabric-dashboard">
    ${hero(bundle)}
    ${topologyPanel(bundle)}
    ${placementPanel(bundle)}
    ${communicationPanel(bundle)}
    ${predictionPanel(bundle)}
    ${autopsyPanel(bundle)}
    ${recoveryPanel(bundle)}
    ${timelinePanel(bundle)}
  </main>`;
}
