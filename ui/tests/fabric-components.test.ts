import { describe, expect, it } from "vitest";
import { renderFabricDashboard } from "../src/fabric-components";
import { generatedFabricBundle } from "./fabric-generated";

describe("Fabric dashboard", () => {
  it("renders all physical, causal, recovery, and optimization views from generated artifacts", () => {
    const bundle = generatedFabricBundle();
    const host = document.createElement("div");
    host.innerHTML = renderFabricDashboard(bundle);
    const headings = [...host.querySelectorAll("h2")].map((heading) => heading.textContent);
    expect(headings).toEqual([
      "Topology graph and rank placement",
      "Parallel groups, experts, and affinity",
      "Collectives, KV flows, and overlap",
      "Predicted, observed, degraded, and restored",
      "Fault, diagnosis, and counterfactual repair",
      "Shadow, canary, promotion, and rollback guards",
      "From SLO regression to restoration",
    ]);
    expect(host.textContent).toContain(bundle.physical_plan.plan_id);
    expect(host.textContent).toContain(bundle.counterfactuals.selected_scenario_id);
    expect(host.textContent).toContain(bundle.recovery_execution.state);
    expect(host.textContent).toContain(bundle.manifest.restored.p95_ttft_ms.toFixed(1));
    expect(host.textContent).not.toContain("undefined");
  });

  it("provides accessible descriptions and exact values for interactive graphics", () => {
    const host = document.createElement("div");
    host.innerHTML = renderFabricDashboard(generatedFabricBundle());
    const graphics = host.querySelectorAll('svg[role="img"]');
    expect(graphics.length).toBeGreaterThanOrEqual(3);
    for (const graphic of graphics) {
      expect(graphic.querySelector("title")?.textContent).not.toBe("");
      expect(graphic.querySelector("desc")?.textContent).not.toBe("");
    }
    const symbols = host.querySelectorAll('[role="graphics-symbol"][tabindex="0"]');
    expect(symbols.length).toBeGreaterThan(10);
    expect([...symbols].every((symbol) => symbol.hasAttribute("aria-label"))).toBe(true);
  });
});
