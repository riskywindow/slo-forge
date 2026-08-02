import type {
  ControllerDecision,
  ControllerWindow,
  Distribution,
  ParetoCandidate,
  ReportArtifact,
  ServingBaselineEntry,
  ServingBaselineName,
  ServingBaselines,
  ServingRegime,
} from "./types";

export type ChartPoint = { label: string; x: number; y: number };

export const servingRegimeOrder: readonly ServingRegime[] = [
  "steady",
  "bursty",
  "short-prompts",
  "long-prompts",
  "mixed",
];

export const servingBaselineOrder: readonly ServingBaselineName[] = [
  "documented-engine-default",
  "manual-static",
  "sloforge-plan",
];

export function orderedServingBaselines(
  baselines: ServingBaselines,
): ServingBaselineEntry[] {
  return [...baselines.results].sort(
    (left, right) =>
      servingRegimeOrder.indexOf(left.regime) -
        servingRegimeOrder.indexOf(right.regime) ||
      servingBaselineOrder.indexOf(left.baseline) -
        servingBaselineOrder.indexOf(right.baseline),
  );
}

export function servingBaselineFor(
  baselines: ServingBaselines,
  baseline: ServingBaselineName,
  regime: ServingRegime,
): ServingBaselineEntry | undefined {
  return baselines.results.find(
    (entry) => entry.baseline === baseline && entry.regime === regime,
  );
}

export function relativeChange(value: number, reference: number): number | null {
  return reference === 0 ? null : (value - reference) / reference;
}

export function selectedCandidate(report: ReportArtifact): ParetoCandidate | undefined {
  const id = report.metrics.selected_config_id;
  return report.pareto_frontier.find(
    (candidate) => candidate.configuration.config_id === id,
  );
}

export function paretoPoints(report: ReportArtifact): (ChartPoint & { fidelity: string; id: string; selected: boolean; uncertainty: number })[] {
  return report.pareto_frontier
    .filter((candidate) =>
      [
        candidate.predicted["cost_per_million_tokens"],
        candidate.predicted["p95_ttft_ms"],
      ].every((value) => typeof value === "number" && Number.isFinite(value)),
    )
    .map((candidate) => ({
      fidelity: candidate.fidelity,
      id: candidate.configuration.config_id,
      label: candidate.configuration.backend_candidate_id,
      selected: candidate.configuration.config_id === report.metrics.selected_config_id,
      uncertainty: candidate.uncertainty["p95_ttft_ms"] ?? 0,
      x: candidate.predicted["cost_per_million_tokens"] ?? 0,
      y: candidate.predicted["p95_ttft_ms"] ?? 0,
    }));
}

export function measuredPredictionPairs(report: ReportArtifact): {
  candidate: string;
  measured: number;
  metric: "p95_ttft_ms" | "p99_itl_ms" | "cost_per_million_tokens";
  predicted: number;
  relativeError: number;
  uncertainty: number;
}[] {
  const metrics = [
    "p95_ttft_ms",
    "p99_itl_ms",
    "cost_per_million_tokens",
  ] as const;
  return report.pareto_frontier.flatMap((candidate) => {
    if (candidate.measured === null) return [];
    return metrics.flatMap((metric) => {
      const measured = candidate.measured?.[metric];
      const predicted = candidate.predicted[metric];
      if (
        typeof measured !== "number" ||
        typeof predicted !== "number" ||
        !Number.isFinite(measured) ||
        !Number.isFinite(predicted)
      ) {
        return [];
      }
      return [
        {
          candidate: candidate.configuration.config_id,
          measured,
          metric,
          predicted,
          relativeError: measured === 0 ? 0 : Math.abs(predicted - measured) / measured,
          uncertainty: candidate.uncertainty[metric] ?? 0,
        },
      ];
    });
  });
}

export function weightedQuantile(distribution: Distribution, quantile: number): number {
  const samples = distribution.empirical
    .filter(({ value, weight }) => Number.isFinite(value) && weight > 0)
    .slice()
    .sort((a, b) => a.value - b.value);
  if (samples.length === 0) return distribution.fixed_value ?? 0;
  const total = samples.reduce((sum, sample) => sum + sample.weight, 0);
  const threshold = Math.max(0, Math.min(1, quantile)) * total;
  let cumulative = 0;
  for (const sample of samples) {
    cumulative += sample.weight;
    if (cumulative >= threshold) return sample.value;
  }
  return samples.at(-1)?.value ?? 0;
}

export function histogram(
  distribution: Distribution,
  binCount = 10,
): { end: number; start: number; weight: number }[] {
  if (binCount < 1) return [];
  const samples = distribution.empirical.filter(
    ({ value, weight }) => Number.isFinite(value) && weight > 0,
  );
  if (samples.length === 0) return [];
  const values = samples.map(({ value }) => value);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const width = Math.max((maximum - minimum) / binCount, 1);
  const bins = Array.from({ length: binCount }, (_, index) => ({
    end: index === binCount - 1 ? maximum : minimum + (index + 1) * width,
    start: minimum + index * width,
    weight: 0,
  }));
  for (const sample of samples) {
    const index = Math.min(
      binCount - 1,
      Math.floor((sample.value - minimum) / width),
    );
    const bin = bins[index];
    if (bin !== undefined) bin.weight += sample.weight;
  }
  return bins;
}

export function activeDecisions(decisions: ControllerDecision[]): ControllerDecision[] {
  return decisions.filter(
    (decision) =>
      decision.chosen.action_type !== "hold" ||
      decision.canary ||
      decision.rolled_back,
  );
}

export function sloAttainmentByWindow(
  windows: ControllerWindow[],
  ttftLimit: number | undefined,
): { attained: boolean; index: number; p95: number; rate: number }[] {
  return windows.map((window) => ({
    attained:
      ttftLimit === undefined || window.observed_p95_ttft_ms <= ttftLimit,
    index: window.window_index,
    p95: window.observed_p95_ttft_ms,
    rate: window.arrival_rate_rps,
  }));
}

export function formatMetric(value: number, unit = ""): string {
  if (!Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  const rendered =
    absolute >= 1000
      ? new Intl.NumberFormat("en", { maximumFractionDigits: 0 }).format(value)
      : absolute >= 10
        ? value.toFixed(1)
        : value.toFixed(2);
  if (unit === "ratio") return `${(value * 100).toFixed(1)}%`;
  if (unit === "USD/million_tokens") return `$${value.toFixed(2)}/Mt`;
  return unit.length > 0 ? `${rendered} ${unit}` : rendered;
}
