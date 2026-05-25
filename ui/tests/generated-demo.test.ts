import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { renderDashboard } from "../src/components";
import { parseReportArtifact } from "../src/parser";

const artifactPath = resolve(import.meta.dirname, "../../reports/demo/report-data.json");

describe.skipIf(!existsSync(artifactPath))("generated CPU demo artifact", () => {
  it("validates and renders the actual report contract without fixture fallbacks", () => {
    const raw = JSON.parse(readFileSync(artifactPath, "utf8")) as unknown;
    const report = parseReportArtifact(raw);
    const dashboard = renderDashboard(report);
    expect(report.pareto_frontier.length).toBeGreaterThan(2);
    expect(report.controller.windows.length).toBeGreaterThan(2);
    expect(report.chaos.executions.length).toBeGreaterThanOrEqual(3);
    expect(dashboard).toContain(report.plan.metadata.uid);
    expect(dashboard).not.toContain("undefined");
  });
});
