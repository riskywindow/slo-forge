import { describe, expect, it } from "vitest";
import { parseReportArtifact } from "../src/parser";
import {
  activeDecisions,
  histogram,
  measuredPredictionPairs,
  selectedCandidate,
  sloAttainmentByWindow,
  weightedQuantile,
} from "../src/transforms";
import { validArtifact } from "./fixture";

const report = parseReportArtifact(validArtifact);

describe("artifact transforms", () => {
  it("resolves the optimizer selection without relying on array position", () => {
    expect(selectedCandidate(report)?.configuration.config_id).toBe("cfg-selected");
  });

  it("computes weighted empirical quantiles deterministically", () => {
    expect(weightedQuantile(report.plan.workload.prompt_tokens, 0.5)).toBe(64);
    expect(weightedQuantile(report.plan.workload.prompt_tokens, 0.95)).toBe(1024);
  });

  it("preserves distribution mass in histograms", () => {
    const bins = histogram(report.plan.workload.output_tokens, 4);
    expect(bins).toHaveLength(4);
    expect(bins.reduce((sum, bin) => sum + bin.weight, 0)).toBeCloseTo(1);
  });

  it("keeps measured and predicted values distinct", () => {
    const pairs = measuredPredictionPairs(report);
    expect(pairs).toHaveLength(3);
    expect(pairs.find(({ metric }) => metric === "p95_ttft_ms")).toMatchObject({
      measured: 190,
      predicted: 200,
    });
  });

  it("extracts material controller actions and window SLO state", () => {
    expect(activeDecisions(report.controller.predictive_decisions)).toHaveLength(1);
    expect(sloAttainmentByWindow(report.controller.windows, 250)).toEqual([
      { attained: true, index: 0, p95: 90, rate: 5 },
      { attained: false, index: 1, p95: 280, rate: 18 },
    ]);
  });
});
