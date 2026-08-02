import { describe, expect, it } from "vitest";
import { renderDashboard } from "../src/components";
import { parseReportArtifact } from "../src/parser";
import { validArtifact } from "./fixture";

describe("artifact dashboard", () => {
  it("renders every required evidence view and accessible SVG descriptions", () => {
    const host = document.createElement("div");
    host.innerHTML = renderDashboard(parseReportArtifact(validArtifact));

    const headings = [...host.querySelectorAll("h2")].map((node) => node.textContent);
    expect(headings).toEqual([
      "The selected plan",
      "Measured versus predicted",
      "Pareto frontier",
      "Workload composition",
      "Trace, SLO, and control timeline",
      "Backends and health",
      "Injected faults and diagnoses",
    ]);
    expect(host.querySelectorAll('svg[role="img"]')).toHaveLength(6);
    for (const svg of host.querySelectorAll('svg[role="img"]')) {
      expect(svg.querySelector("title")?.textContent).not.toBe("");
      expect(svg.querySelector("desc")?.textContent).not.toBe("");
    }
    expect(host.textContent).toContain("backend slowdown");
    expect(host.textContent).toContain("predictive-00001");
    expect(host.textContent).not.toContain("undefined");
  });

  it("exposes exact chart values through keyboard-focusable graphics symbols", () => {
    const host = document.createElement("div");
    host.innerHTML = renderDashboard(parseReportArtifact(validArtifact));
    const symbols = host.querySelectorAll('[role="graphics-symbol"][tabindex="0"]');
    expect(symbols.length).toBeGreaterThan(10);
    expect([...symbols].every((symbol) => symbol.hasAttribute("aria-label"))).toBe(true);
  });
});
