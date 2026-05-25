import {
  escapeHtml,
  histogramPlot,
  intervalPlot,
  scatterPlot,
  timelinePlot,
  workloadDonut,
} from "./charts";
import {
  activeDecisions,
  formatMetric,
  histogram,
  measuredPredictionPairs,
  paretoPoints,
  selectedCandidate,
  sloAttainmentByWindow,
  weightedQuantile,
} from "./transforms";
import type { IntervalMetric, ReportArtifact } from "./types";

const metricLabels: Record<string, string> = {
  availability: "Availability",
  cold_start_p95_ms: "Cold start p95",
  cost_per_million_tokens: "Cost",
  goodput_tokens_s: "Goodput",
  p95_e2e_ms: "End-to-end p95",
  p95_ttft_ms: "TTFT p95",
  p99_itl_ms: "ITL p99",
  throughput_tokens_s: "Throughput",
};

function label(value: string): string {
  return metricLabels[value] ?? value.replaceAll("_", " ");
}

function sectionHeading(kicker: string, title: string, description: string): string {
  return `<header class="section-heading"><p class="eyebrow">${escapeHtml(kicker)}</p><h2>${escapeHtml(title)}</h2><p>${escapeHtml(description)}</p></header>`;
}

function metricCard(name: string, value: string, detail: string): string {
  return `<article class="metric-card"><p>${escapeHtml(name)}</p><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small></article>`;
}

function planPanel(report: ReportArtifact): string {
  const { plan } = report;
  const ttft = plan.slo.ttft[0];
  const itl = plan.slo.inter_token_latency[0];
  return `<section id="plan" class="panel-section">
    ${sectionHeading("Compiled deployment", "The selected plan", "The versioned IR ties runtime, capacity, policy, and SLOs to evidence-backed predictions.")}
    <div class="plan-grid">
      <article class="card plan-identity">
        <span class="status-pill selected">Selected</span>
        <h3>${escapeHtml(plan.metadata.name)}</h3>
        <p class="model-name">${escapeHtml(plan.model.model_id)}</p>
        <dl class="definition-grid">
          <div><dt>Plan ID</dt><dd><code>${escapeHtml(plan.metadata.uid)}</code></dd></div>
          <div><dt>IR</dt><dd>${escapeHtml(plan.schema_version)}</dd></div>
          <div><dt>Created</dt><dd>${escapeHtml(new Date(plan.metadata.created_at).toLocaleString())}</dd></div>
          <div><dt>Trace seed</dt><dd>${plan.workload.seed}</dd></div>
        </dl>
      </article>
      <article class="card">
        <h3>Runtime shape</h3>
        <dl class="definition-grid compact">
          <div><dt>Engine</dt><dd>${escapeHtml(`${plan.engine.runtime} ${plan.engine.version}`)}</dd></div>
          <div><dt>Precision</dt><dd>${escapeHtml(`${plan.engine.dtype} · ${plan.engine.quantization}`)}</dd></div>
          <div><dt>Parallelism</dt><dd>${plan.engine.tensor_parallelism}× tensor</dd></div>
          <div><dt>Batch tokens</dt><dd>${plan.batching.maximum_batched_tokens.toLocaleString()}</dd></div>
          <div><dt>Concurrency</dt><dd>${plan.batching.maximum_active_sequences}</dd></div>
          <div><dt>Queue bound</dt><dd>${plan.admission.queue_capacity}</dd></div>
        </dl>
      </article>
      <article class="card">
        <h3>Hard constraints</h3>
        <ul class="constraint-list">
          ${ttft === undefined ? "" : `<li><span>p${ttft.percentile} TTFT</span><strong>≤ ${ttft.maximum_ms} ms</strong></li>`}
          ${itl === undefined ? "" : `<li><span>p${itl.percentile} ITL</span><strong>≤ ${itl.maximum_ms} ms</strong></li>`}
          ${plan.slo.minimum_availability === null ? "" : `<li><span>Availability</span><strong>≥ ${(plan.slo.minimum_availability * 100).toFixed(1)}%</strong></li>`}
          <li><span>Routing</span><strong>${escapeHtml(plan.routing.kind)}</strong></li>
          <li><span>Autoscaling</span><strong>${escapeHtml(plan.autoscaling.mode)}</strong></li>
        </ul>
      </article>
    </div>
  </section>`;
}

function evidencePanel(report: ReportArtifact): string {
  const intervals = Object.entries(report.plan.predicted_metrics).map(([name, metric]) => ({
    label: label(name),
    lower: metric.lower,
    point: metric.point,
    unit: metric.unit,
    upper: metric.upper,
  }));
  const pairRows = measuredPredictionPairs(report);
  const representatives = new Map<string, (typeof pairRows)[number]>();
  for (const pair of pairRows) {
    const candidate = report.pareto_frontier.find(
      (item) => item.configuration.config_id === pair.candidate,
    );
    const key = `${candidate?.configuration.backend_candidate_id ?? pair.candidate}:${pair.metric}`;
    if (!representatives.has(key)) representatives.set(key, pair);
  }
  return `<section id="evidence" class="panel-section">
    ${sectionHeading("Model evidence", "Measured versus predicted", "Intervals retain sample count and measurement provenance; held-out deltas remain visible even when the model loses.")}
    <div class="card chart-card">${intervalPlot(intervals)}</div>
    <div class="card table-card">
      <div class="table-heading"><h3>Representative held-out comparisons</h3><p>One row per backend and metric; repeated configuration probes are collapsed.</p></div>
      <div class="table-scroll"><table><thead><tr><th scope="col">Backend</th><th scope="col">Metric</th><th scope="col">Measured</th><th scope="col">Predicted</th><th scope="col">Abs. error</th><th scope="col">Model interval ±</th></tr></thead>
      <tbody>${[...representatives.values()]
        .map((pair) => {
          const candidate = report.pareto_frontier.find(
            (item) => item.configuration.config_id === pair.candidate,
          );
          const unit = report.plan.predicted_metrics[pair.metric]?.unit ?? "";
          return `<tr><td>${escapeHtml(candidate?.configuration.backend_candidate_id ?? pair.candidate)}</td><td>${escapeHtml(label(pair.metric))}</td><td>${escapeHtml(formatMetric(pair.measured, unit))}</td><td>${escapeHtml(formatMetric(pair.predicted, unit))}</td><td class="${pair.relativeError > 0.25 ? "warning-text" : ""}">${(pair.relativeError * 100).toFixed(1)}%</td><td>${escapeHtml(formatMetric(pair.uncertainty, unit))}</td></tr>`;
        })
        .join("")}</tbody></table></div>
    </div>
    <details class="provenance"><summary>Metric provenance</summary>${Object.entries(
      report.plan.predicted_metrics,
    )
      .map(
        ([name, metric]) =>
          `<p><strong>${escapeHtml(label(name))}</strong> · ${metric.sample_count} samples · ${(metric.confidence * 100).toFixed(0)}% interval · ${metric.measurement_ids.length} measurement references</p>`,
      )
      .join("")}</details>
  </section>`;
}

function frontierPanel(report: ReportArtifact): string {
  const selected = selectedCandidate(report);
  const measuredCount = report.pareto_frontier.filter(({ fidelity }) => fidelity === "measured").length;
  const predictedCount = report.pareto_frontier.length - measuredCount;
  return `<section id="frontier" class="panel-section">
    ${sectionHeading("Optimization", "Pareto frontier", "Each point is a feasible trade-off retained by the compiler; measured-fidelity and simulated points stay distinct.")}
    <div class="split-grid wide-chart">
      <article class="card chart-card">${scatterPlot(paretoPoints(report))}<div class="legend" aria-label="Chart legend"><span><i class="legend-dot measured"></i>Measured fidelity (${measuredCount})</span><span><i class="legend-dot predicted"></i>Predicted only (${predictedCount})</span><span><i class="legend-ring"></i>Selected</span></div></article>
      <aside class="card selected-candidate">
        <p class="eyebrow">Compiler output</p><h3>${escapeHtml(selected?.configuration.backend_candidate_id ?? report.metrics.selected_config_id)}</h3>
        ${selected === undefined ? '<p class="warning-text">The selected configuration is not present in this frontier.</p>' : `<dl class="definition-grid compact">
          <div><dt>Configuration</dt><dd><code>${escapeHtml(selected.configuration.config_id)}</code></dd></div>
          <div><dt>Fidelity</dt><dd>${escapeHtml(selected.fidelity)}</dd></div>
          <div><dt>Replicas</dt><dd>${selected.configuration.replicas} (${selected.configuration.warm_replicas} warm)</dd></div>
          <div><dt>Concurrency</dt><dd>${selected.configuration.concurrency}</dd></div>
          <div><dt>Routing</dt><dd>${escapeHtml(selected.configuration.routing_policy)}</dd></div>
          <div><dt>Batch cap</dt><dd>${selected.configuration.max_batched_tokens.toLocaleString()} tokens</dd></div>
        </dl>`}
      </aside>
    </div>
  </section>`;
}

function workloadPanel(report: ReportArtifact): string {
  const workload = report.plan.workload;
  const promptP50 = weightedQuantile(workload.prompt_tokens, 0.5);
  const promptP95 = weightedQuantile(workload.prompt_tokens, 0.95);
  const outputP50 = weightedQuantile(workload.output_tokens, 0.5);
  const outputP95 = weightedQuantile(workload.output_tokens, 0.95);
  return `<section id="workload" class="panel-section">
    ${sectionHeading("Trace shape", "Workload composition", "The charts are derived from the empirical token distributions embedded in the compiled plan.")}
    <div class="workload-grid">
      <article class="card class-mix"><h3>Request classes</h3><div class="donut-layout">${workloadDonut(workload.request_classes)}<ul>${workload.request_classes
        .map(
          (item, index) =>
            `<li><i class="swatch segment-${index % 4}"></i><span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.priority)} · ${(item.weight * 100).toFixed(1)}%${item.deadline_ms === null ? "" : ` · ${item.deadline_ms.toFixed(0)} ms deadline`}</small></span></li>`,
        )
        .join("")}</ul></div></article>
      <article class="card distribution"><div><h3>Prompt lengths</h3><p>p50 ${promptP50.toFixed(0)} · p95 ${promptP95.toFixed(0)} tokens</p></div>${histogramPlot(histogram(workload.prompt_tokens), "prompt-histogram", "Prompt-token distribution")}</article>
      <article class="card distribution"><div><h3>Output lengths</h3><p>p50 ${outputP50.toFixed(0)} · p95 ${outputP95.toFixed(0)} tokens</p></div>${histogramPlot(histogram(workload.output_tokens), "output-histogram", "Output-token distribution")}</article>
    </div>
  </section>`;
}

function runtimePanel(report: ReportArtifact): string {
  const windows = report.controller.windows;
  const decisions = report.controller.predictive_decisions;
  const active = activeDecisions(decisions);
  const ttftLimit = report.plan.slo.ttft[0]?.maximum_ms;
  const attainment = sloAttainmentByWindow(windows, ttftLimit);
  const attained = attainment.filter((window) => window.attained).length;
  const replayAttainment = report.metrics["replay_slo_attainment"];
  const replayAttainmentNumber =
    typeof replayAttainment === "number" ? replayAttainment : attained / Math.max(attainment.length, 1);
  return `<section id="runtime" class="panel-section">
    ${sectionHeading("Runtime verification", "Trace, SLO, and control timeline", "Window aggregates expose workload bursts, latency response, fault timing, and guarded controller actions on one time axis.")}
    <div class="summary-strip">
      ${metricCard("Replay SLO attainment", `${(replayAttainmentNumber * 100).toFixed(1)}%`, `${attained}/${attainment.length} controller windows within TTFT limit`)}
      ${metricCard("Predictive violations", report.controller.predictive.slo_violations.toString(), `${report.controller.reactive.slo_violations} with reactive baseline`)}
      ${metricCard("Controller cost", `$${report.controller.predictive.estimated_cost_usd.toFixed(4)}`, `$${report.controller.reactive.estimated_cost_usd.toFixed(4)} reactive`)}
      ${metricCard("Adaptations", active.length.toString(), `${report.controller.predictive.rollback_count} rollbacks`)}
    </div>
    <article class="card chart-card timeline-card">${timelinePlot(windows, decisions, report.chaos.executions, ttftLimit)}<div class="timeline-key"><span><i class="key-line ttft"></i>p95 TTFT</span><span><i class="key-line rate"></i>arrival rate</span><span><i class="key-block"></i>fault</span><span><i class="key-triangle"></i>action</span></div></article>
    <div class="card table-card"><div class="table-heading"><h3>Controller decision records</h3><p>Material actions, canaries, and rollbacks; holds remain in the artifact but are collapsed here.</p></div>
      ${active.length === 0 ? '<p class="empty">No material controller actions.</p>' : `<div class="table-scroll"><table><thead><tr><th scope="col">Window</th><th scope="col">Observed → forecast</th><th scope="col">Action</th><th scope="col">Target</th><th scope="col">State</th><th scope="col">Guard</th></tr></thead><tbody>${active
        .map(
          (decision) =>
            `<tr><td><code>${escapeHtml(decision.decision_id)}</code></td><td>${decision.observed.arrival_rate_rps.toFixed(1)} → ${decision.forecast.arrival_rate_rps.toFixed(1)} rps</td><td>${escapeHtml(decision.chosen.action_type)}${decision.canary ? ' <span class="status-pill">canary</span>' : ""}</td><td>${decision.chosen.target_replicas} replicas · c${decision.chosen.target_concurrency}</td><td>${escapeHtml(`${decision.controller_state_before} → ${decision.controller_state_after}`)}</td><td>${decision.rolled_back ? `<span class="status-pill danger">rolled back</span>` : '<span class="status-pill healthy">passed</span>'}</td></tr>`,
        )
        .join("")}</tbody></table></div>`}
    </div>
  </section>`;
}

function topologyPanel(report: ReportArtifact): string {
  const { plan } = report;
  const lastWindow = report.controller.windows.at(-1);
  const errorRate = lastWindow?.backend_error_rate ?? 0;
  const health = errorRate > 0.1 ? "unhealthy" : errorRate > 0.01 ? "degraded" : "healthy";
  const latestReplicas = lastWindow?.replicas ?? plan.replica_topology.initial_replicas;
  const targets = plan.routing.targets;
  return `<section id="topology" class="panel-section">
    ${sectionHeading("Serving topology", "Backends and health", "The topology is reconstructed from routing targets, replica policy, and the final replay health window.")}
    <article class="card topology-card">
      <div class="topology-flow" role="img" aria-label="Gateway routes through ${escapeHtml(plan.routing.kind)} policy to ${targets.length} deployment target with ${latestReplicas} current replicas">
        <div class="topology-node gateway-node"><span>Ingress</span><strong>Rust gateway</strong><small>bounded queue · ${plan.admission.queue_capacity}</small></div>
        <div class="connector" aria-hidden="true"><span>${escapeHtml(plan.routing.kind)}</span></div>
        <div class="target-stack">${targets
          .map(
            (target) =>
              `<div class="topology-node backend-node"><span><i class="health-dot ${health}"></i>${escapeHtml(health)}</span><strong>${escapeHtml(target.variant)}</strong><small>${(target.weight * 100).toFixed(0)}% weight · ${(errorRate * 100).toFixed(2)}% errors</small></div>`,
          )
          .join("")}</div>
      </div>
      <div class="replica-rail"><div><span>Replica envelope</span><strong>${plan.replica_topology.minimum_replicas} min · ${latestReplicas} observed · ${plan.replica_topology.maximum_replicas} max</strong></div><div class="replicas" aria-label="${latestReplicas} active replicas">${Array.from({ length: plan.replica_topology.maximum_replicas }, (_, index) => `<i class="replica ${index < latestReplicas ? "active" : ""}" title="Replica ${index + 1}${index < latestReplicas ? " active" : " available"}"></i>`).join("")}</div></div>
      <dl class="topology-facts">
        <div><dt>Hardware</dt><dd>${escapeHtml(plan.hardware.gpu_count > 0 ? `${plan.hardware.gpu_count} GPU` : plan.hardware.cpu.model)}</dd></div>
        <div><dt>Region</dt><dd>${escapeHtml(plan.hardware.region)}</dd></div>
        <div><dt>Hourly unit cost</dt><dd>$${plan.hardware.hourly_price_usd.toFixed(2)}</dd></div>
        <div><dt>Scale target</dt><dd>${(plan.autoscaling.target_utilization * 100).toFixed(0)}% utilization</dd></div>
      </dl>
    </article>
  </section>`;
}

function chaosPanel(report: ReportArtifact): string {
  return `<section id="faults" class="panel-section">
    ${sectionHeading("Adversarial evidence", "Injected faults and diagnoses", "Known fault labels are compared with deterministic counter-based diagnoses; confidence and latency are preserved per execution.")}
    <div class="chaos-summary card"><div><p class="eyebrow">${escapeHtml(report.chaos.scenario_name)}</p><h3>${(report.chaos.diagnosis_accuracy * 100).toFixed(1)}% closed-set label agreement</h3></div><dl><div><dt>Mean execution latency</dt><dd>${report.chaos.mean_diagnosis_latency_ms.toFixed(3)} ms</dd></div><div><dt>Negative-window FPR</dt><dd>${(report.chaos.false_positive_rate * 100).toFixed(1)}% (${report.chaos.false_positive_count}/${report.chaos.negative_window_count})</dd></div><div><dt>Seed</dt><dd>${report.chaos.seed}</dd></div></dl></div>
    <div class="fault-grid">${report.chaos.executions
      .map(({ applied, diagnosis, event }) => {
        const outcome = diagnosis.correct ? "correct" : "incorrect";
        return `<article class="card fault-card ${outcome}"><div class="fault-head"><span class="fault-time">+${(event.start_ms / 1000).toFixed(1)}s · ${event.duration_ms.toFixed(0)}ms</span><span class="status-pill ${diagnosis.correct ? "healthy" : "danger"}">${outcome}</span></div><h3>${escapeHtml(event.fault_type.replaceAll("_", " "))}</h3><p>${event.backend_id === null ? "gateway / aggregate" : `backend · ${escapeHtml(event.backend_id)}`}</p><div class="diagnosis-arrow"><span>Diagnosed as</span><strong>${escapeHtml(diagnosis.predicted_label.replaceAll("_", " "))}</strong></div><dl><div><dt>Confidence</dt><dd>${(diagnosis.confidence * 100).toFixed(1)}%</dd></div><div><dt>Latency</dt><dd>${diagnosis.diagnosis_latency_ms.toFixed(1)} ms</dd></div></dl><p class="evidence-note">${escapeHtml(diagnosis.evidence.join("; "))}</p>${applied ? "" : '<p class="warning-text">Fault was not applied.</p>'}</article>`;
      })
      .join("")}</div>
  </section>`;
}

function overview(report: ReportArtifact): string {
  const metric = (name: string): IntervalMetric | undefined =>
    report.plan.predicted_metrics[name];
  const ttft = metric("p95_ttft_ms");
  const itl = metric("p99_itl_ms");
  const cost = metric("cost_per_million_tokens");
  const goodput = metric("goodput_tokens_s");
  return `<section class="hero" aria-labelledby="page-title">
    <div><p class="eyebrow">Artifact-backed deployment evidence</p><h1 id="page-title">${escapeHtml(report.plan.metadata.name)}</h1><p>SLOForge compiled <strong>${escapeHtml(report.plan.model.model_id)}</strong> into a measured, deployable operating point.</p></div>
    <div class="artifact-stamp"><span>IR ${escapeHtml(report.plan.schema_version)}</span><code>${escapeHtml(report.plan.metadata.uid)}</code></div>
  </section>
  <section class="summary-strip" aria-label="Selected plan predictions">
    ${metricCard("p95 TTFT", ttft === undefined ? "—" : formatMetric(ttft.point, ttft.unit), ttft === undefined ? "No interval" : `${formatMetric(ttft.lower, ttft.unit)}—${formatMetric(ttft.upper, ttft.unit)}`)}
    ${metricCard("p99 ITL", itl === undefined ? "—" : formatMetric(itl.point, itl.unit), itl === undefined ? "No interval" : `${formatMetric(itl.lower, itl.unit)}—${formatMetric(itl.upper, itl.unit)}`)}
    ${metricCard("Cost / Mt", cost === undefined ? "—" : formatMetric(cost.point, cost.unit), `${report.plan.replica_topology.initial_replicas} initial replicas`)}
    ${metricCard("Goodput", goodput === undefined ? "—" : formatMetric(goodput.point, goodput.unit), `${report.pareto_frontier.length} Pareto candidates`)}
  </section>`;
}

export function renderDashboard(report: ReportArtifact): string {
  return `<main id="dashboard">
    ${overview(report)}
    ${planPanel(report)}
    ${evidencePanel(report)}
    ${frontierPanel(report)}
    ${workloadPanel(report)}
    ${runtimePanel(report)}
    ${topologyPanel(report)}
    ${chaosPanel(report)}
  </main>`;
}
