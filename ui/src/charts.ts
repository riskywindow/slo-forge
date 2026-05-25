import type { ChaosExecution, ControllerDecision, ControllerWindow } from "./types";

export function escapeHtml(value: unknown): string {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function finiteExtent(values: number[]): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return [0, 1];
  const minimum = Math.min(...finite);
  const maximum = Math.max(...finite);
  if (minimum === maximum) return [minimum - 0.5, maximum + 0.5];
  return [minimum, maximum];
}

function linearScale(
  domainMinimum: number,
  domainMaximum: number,
  rangeMinimum: number,
  rangeMaximum: number,
): (value: number) => number {
  const span = domainMaximum - domainMinimum || 1;
  return (value) =>
    rangeMinimum + ((value - domainMinimum) / span) * (rangeMaximum - rangeMinimum);
}

function svgFrame(
  id: string,
  title: string,
  description: string,
  content: string,
  viewBox = "0 0 720 300",
): string {
  const safeId = id.replaceAll(/[^a-zA-Z0-9_-]/g, "-");
  return `<svg class="chart" viewBox="${viewBox}" role="img" aria-labelledby="${safeId}-title ${safeId}-desc">
    <title id="${safeId}-title">${escapeHtml(title)}</title>
    <desc id="${safeId}-desc">${escapeHtml(description)}</desc>
    ${content}
  </svg>`;
}

function axisTicks(
  scale: (value: number) => number,
  minimum: number,
  maximum: number,
  axis: "x" | "y",
  formatter: (value: number) => string,
): string {
  return Array.from({ length: 5 }, (_, index) => {
    const value = minimum + ((maximum - minimum) * index) / 4;
    const position = scale(value);
    return axis === "x"
      ? `<g aria-hidden="true"><line x1="${position}" y1="250" x2="${position}" y2="255"/><text x="${position}" y="273" text-anchor="middle">${escapeHtml(formatter(value))}</text></g>`
      : `<g aria-hidden="true"><line x1="54" y1="${position}" x2="660" y2="${position}" class="grid"/><text x="46" y="${position + 4}" text-anchor="end">${escapeHtml(formatter(value))}</text></g>`;
  }).join("");
}

export function scatterPlot(
  points: {
    fidelity: string;
    id: string;
    label: string;
    selected: boolean;
    uncertainty: number;
    x: number;
    y: number;
  }[],
): string {
  if (points.length === 0) return '<p class="empty">No frontier points in this artifact.</p>';
  const [xMin, xMax] = finiteExtent(points.map(({ x }) => x));
  const logValues = points.map(({ y }) => Math.log10(Math.max(y, 0.1)));
  const [logMin, logMax] = finiteExtent(logValues);
  const x = linearScale(xMin * 0.95, xMax * 1.05, 62, 660);
  const y = linearScale(logMin - 0.08, logMax + 0.08, 246, 22);
  const dots = points
    .map((point) => {
      const cx = x(point.x);
      const cy = y(Math.log10(Math.max(point.y, 0.1)));
      const className = [
        "frontier-point",
        point.fidelity === "measured" ? "measured" : "predicted",
        point.selected ? "selected" : "",
      ]
        .filter(Boolean)
        .join(" ");
      return `<g class="${className}" tabindex="0" role="graphics-symbol" aria-label="${escapeHtml(`${point.label}: $${point.x.toFixed(2)} per million tokens, ${point.y.toFixed(1)} milliseconds TTFT, ${point.fidelity}${point.selected ? ", selected" : ""}`)}">
        <circle cx="${cx}" cy="${cy}" r="${point.selected ? 8 : 5}"/>
        ${point.selected ? `<circle class="selection-ring" cx="${cx}" cy="${cy}" r="13"/>` : ""}
        <title>${escapeHtml(`${point.id} · ${point.label}`)}</title>
      </g>`;
    })
    .join("");
  const yTickScale = linearScale(logMin - 0.08, logMax + 0.08, 246, 22);
  const yTicks = axisTicks(
    yTickScale,
    logMin,
    logMax,
    "y",
    (value) => Math.pow(10, value).toFixed(0),
  );
  const content = `<g class="axes">
      <line x1="54" y1="250" x2="672" y2="250"/>
      ${axisTicks(x, xMin, xMax, "x", (value) => `$${value.toFixed(2)}`)}
      ${yTicks}
      <text class="axis-label" x="360" y="296" text-anchor="middle">Predicted cost / million tokens</text>
      <text class="axis-label" x="13" y="140" text-anchor="middle" transform="rotate(-90 13 140)">Predicted p95 TTFT, log ms</text>
    </g>${dots}`;
  return svgFrame(
    "pareto-chart",
    "Predicted Pareto frontier",
    "Cost per million tokens on the horizontal axis and logarithmic predicted p95 time to first token on the vertical axis. Measured-fidelity candidates are teal, predicted-only candidates are amber, and the selected plan has a ring.",
    content,
  );
}

export function intervalPlot(
  metrics: { label: string; lower: number; point: number; unit: string; upper: number }[],
): string {
  if (metrics.length === 0) return '<p class="empty">No interval metrics in this plan.</p>';
  const rowHeight = 52;
  const height = 34 + metrics.length * rowHeight;
  const content = metrics
    .map((metric, index) => {
      const magnitude = Math.max(Math.abs(metric.lower), Math.abs(metric.upper), 0.0001);
      const start = 260 + (metric.lower / magnitude) * 380;
      const end = 260 + (metric.upper / magnitude) * 380;
      const point = 260 + (metric.point / magnitude) * 380;
      const y = 28 + index * rowHeight;
      return `<g tabindex="0" role="graphics-symbol" aria-label="${escapeHtml(`${metric.label}: ${metric.point} ${metric.unit}, interval ${metric.lower} to ${metric.upper}`)}">
        <text x="12" y="${y + 5}" class="row-label">${escapeHtml(metric.label)}</text>
        <line x1="260" y1="${y}" x2="640" y2="${y}" class="interval-track"/>
        <line x1="${start}" y1="${y}" x2="${end}" y2="${y}" class="interval-range"/>
        <circle cx="${point}" cy="${y}" r="6" class="interval-point"/>
        <text x="654" y="${y + 5}" class="value-label">${escapeHtml(compact(metric.point))}</text>
      </g>`;
    })
    .join("");
  return svgFrame(
    "interval-chart",
    "Predicted metric confidence intervals",
    "Each row shows a metric point prediction and its calibrated confidence interval, normalized within the row. Exact values follow in the table.",
    content,
    `0 0 720 ${height}`,
  );
}

function compact(value: number): string {
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2);
}

export function histogramPlot(
  bins: { end: number; start: number; weight: number }[],
  id: string,
  title: string,
): string {
  if (bins.length === 0) return '<p class="empty">No empirical samples.</p>';
  const maxWeight = Math.max(...bins.map(({ weight }) => weight), 0.001);
  const barWidth = 590 / bins.length;
  const bars = bins
    .map((bin, index) => {
      const height = (bin.weight / maxWeight) * 130;
      const x = 68 + index * barWidth;
      return `<g tabindex="0" role="graphics-symbol" aria-label="${escapeHtml(`${bin.start.toFixed(0)} to ${bin.end.toFixed(0)} tokens: ${(bin.weight * 100).toFixed(1)} percent weight`)}">
        <rect x="${x + 2}" y="${180 - height}" width="${Math.max(barWidth - 4, 1)}" height="${height}"/>
      </g>`;
    })
    .join("");
  const content = `<line x1="62" y1="180" x2="670" y2="180" class="axis-line"/>
    ${bars}
    <text x="68" y="204">${escapeHtml(bins[0]?.start.toFixed(0) ?? "0")}</text>
    <text x="668" y="204" text-anchor="end">${escapeHtml(bins.at(-1)?.end.toFixed(0) ?? "0")}</text>
    <text class="axis-label" x="360" y="224" text-anchor="middle">tokens</text>`;
  return svgFrame(id, title, `${title}, binned by token count and weighted frequency.`, content, "0 0 720 230");
}

export function workloadDonut(
  classes: { name: string; priority: string; weight: number }[],
): string {
  if (classes.length === 0) return '<p class="empty">No request classes.</p>';
  const total = classes.reduce((sum, item) => sum + item.weight, 0) || 1;
  let offset = 0;
  const segments = classes
    .map((item, index) => {
      const ratio = item.weight / total;
      const dash = ratio * 100;
      const segment = `<circle class="donut-segment segment-${index % 4}" cx="120" cy="120" r="78" pathLength="100" stroke-dasharray="${dash} ${100 - dash}" stroke-dashoffset="${-offset}" tabindex="0" role="graphics-symbol" aria-label="${escapeHtml(`${item.name}, ${item.priority} priority: ${(ratio * 100).toFixed(1)} percent`)}"/>`;
      offset += dash;
      return segment;
    })
    .join("");
  return svgFrame(
    "workload-donut",
    "Workload request-class mix",
    "Weighted proportions for each request class. Exact proportions are listed beside the chart.",
    `${segments}<circle class="donut-hole" cx="120" cy="120" r="47"/><text class="donut-total" x="120" y="116" text-anchor="middle">${classes.length}</text><text x="120" y="137" text-anchor="middle">classes</text>`,
    "0 0 240 240",
  );
}

export function timelinePlot(
  windows: ControllerWindow[],
  decisions: ControllerDecision[],
  faults: ChaosExecution[],
  ttftLimit: number | undefined,
): string {
  if (windows.length === 0) return '<p class="empty">No replay windows.</p>';
  const maxTime = Math.max(...windows.map(({ window_end_ms }) => window_end_ms), 1);
  const maxTtft = Math.max(
    ...windows.map(({ observed_p95_ttft_ms }) => observed_p95_ttft_ms),
    ttftLimit ?? 0,
    1,
  );
  const maxRate = Math.max(...windows.map(({ arrival_rate_rps }) => arrival_rate_rps), 1);
  const x = linearScale(0, maxTime, 58, 674);
  const yTtft = linearScale(0, maxTtft * 1.08, 244, 24);
  const yRate = linearScale(0, maxRate * 1.08, 244, 24);
  const line = (key: "ttft" | "rate") =>
    windows
      .map((window, index) => {
        const xValue = x((window.window_start_ms + window.window_end_ms) / 2);
        const value =
          key === "ttft" ? window.observed_p95_ttft_ms : window.arrival_rate_rps;
        const yValue = key === "ttft" ? yTtft(value) : yRate(value);
        return `${index === 0 ? "M" : "L"}${xValue},${yValue}`;
      })
      .join(" ");
  const faultBands = faults
    .filter(({ applied }) => applied)
    .map(({ event }) => {
      const start = x(event.start_ms);
      const width = Math.max(x(Math.min(event.start_ms + event.duration_ms, maxTime)) - start, 2);
      return `<rect class="fault-band" x="${start}" y="24" width="${width}" height="220"><title>${escapeHtml(`${event.fault_type}: ${event.duration_ms} ms`)}</title></rect>`;
    })
    .join("");
  const decisionMarks = decisions
    .filter(({ chosen }) => chosen.action_type !== "hold")
    .map((decision) => {
      const cx = x(decision.observed.window_end_ms);
      return `<path class="decision-mark" d="M${cx - 6},16 L${cx + 6},16 L${cx},27 Z"><title>${escapeHtml(`${decision.chosen.action_type}: ${decision.chosen.target_replicas} replicas`)}</title></path>`;
    })
    .join("");
  const threshold =
    ttftLimit === undefined
      ? ""
      : `<line class="slo-line" x1="58" y1="${yTtft(ttftLimit)}" x2="674" y2="${yTtft(ttftLimit)}"><title>p95 TTFT SLO: ${ttftLimit} ms</title></line>`;
  const dots = windows
    .map((window) => {
      const cx = x((window.window_start_ms + window.window_end_ms) / 2);
      const cy = yTtft(window.observed_p95_ttft_ms);
      return `<circle class="timeline-dot ${ttftLimit !== undefined && window.observed_p95_ttft_ms > ttftLimit ? "violation" : ""}" cx="${cx}" cy="${cy}" r="4" tabindex="0" role="graphics-symbol" aria-label="${escapeHtml(`Window ${window.window_index}: ${window.observed_p95_ttft_ms.toFixed(1)} milliseconds p95 TTFT, ${window.arrival_rate_rps.toFixed(1)} requests per second, ${window.replicas} replicas`)}"/>`;
    })
    .join("");
  const content = `${faultBands}${threshold}<path class="timeline-line ttft" d="${line("ttft")}"/><path class="timeline-line rate" d="${line("rate")}"/>${dots}${decisionMarks}
    <line x1="58" y1="244" x2="674" y2="244" class="axis-line"/>
    <text x="58" y="270">0 s</text><text x="674" y="270" text-anchor="end">${(maxTime / 1000).toFixed(1)} s</text>
    <g class="chart-legend" aria-hidden="true"><line x1="482" y1="288" x2="504" y2="288" class="timeline-line ttft"/><text x="510" y="292">p95 TTFT</text><line x1="586" y1="288" x2="608" y2="288" class="timeline-line rate"/><text x="614" y="292">rate</text></g>`;
  return svgFrame(
    "runtime-timeline",
    "Replay and controller timeline",
    "Windowed p95 time to first token and arrival rate. Shaded vertical bands are injected faults, triangles are controller actions, and red points violate the TTFT SLO.",
    content,
  );
}
