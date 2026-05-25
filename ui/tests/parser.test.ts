import { describe, expect, it } from "vitest";
import { ArtifactValidationError, parseReportArtifact } from "../src/parser";
import { validArtifact } from "./fixture";

describe("parseReportArtifact", () => {
  it("accepts the explicit artifact fixture", () => {
    const report = parseReportArtifact(validArtifact);
    expect(report.plan.metadata.uid).toBe("plan-fixture-1");
    expect(report.pareto_frontier).toHaveLength(2);
  });

  it("reports structural paths for incomplete artifacts", () => {
    expect(() => parseReportArtifact({ plan: {} })).toThrow(ArtifactValidationError);
    try {
      parseReportArtifact({ plan: {} });
    } catch (error: unknown) {
      expect(error).toBeInstanceOf(ArtifactValidationError);
      expect((error as ArtifactValidationError).problems).toContain(
        "$.pareto_frontier must be an array",
      );
    }
  });

  it("rejects confidence intervals that do not contain their point", () => {
    const artifact = structuredClone(validArtifact) as unknown as {
      plan: { predicted_metrics: { p95_ttft_ms: { lower: number } } };
    };
    artifact.plan.predicted_metrics.p95_ttft_ms.lower = 210;
    expect(() => parseReportArtifact(artifact)).toThrow(/interval must contain its point/);
  });
});
