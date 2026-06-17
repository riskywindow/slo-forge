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
    expect(host.textContent).toContain("synthetic calibrated fabric profile");
    expect(host.textContent).toContain("Physical topology edges");
    expect(host.textContent).toContain(
      `${bundle.recovery_plan.rollback_criteria.length} armed`,
    );
    const firstHypothesis = bundle.diagnosis.hypotheses[0];
    const firstRollback = bundle.recovery_plan.rollback_criteria[0];
    if (firstHypothesis === undefined || firstRollback === undefined) {
      throw new Error("generated Fabric evidence lacks diagnosis or rollback data");
    }
    expect(host.textContent).toContain(firstHypothesis.target);
    expect(host.textContent).toContain(firstRollback.metric);
    expect(host.textContent).not.toContain("Representative measured fabric");
    expect(host.textContent).not.toContain("undefined");
    expect(host.querySelectorAll(".fabric-pareto-point.selected")).toHaveLength(1);
    expect(host.querySelectorAll("table tbody tr").length).toBeGreaterThan(
      bundle.physical_plan.rank_placement.bindings.length,
    );
    const firstEdge = bundle.topology.edges[0];
    const firstExpert = bundle.physical_plan.expert_placement.assignments[0];
    const firstRoute = bundle.physical_plan.kv_transfer.routes[0];
    const firstCandidate = bundle.optimizer.pareto_frontier[0];
    const firstResource = [...bundle.simulations.degraded.metrics.resources].sort(
      (left, right) => right.utilization - left.utilization,
    )[0];
    const firstTimeline = bundle.timeline[0];
    if (
      firstEdge === undefined ||
      firstExpert === undefined ||
      firstRoute === undefined ||
      firstCandidate === undefined ||
      firstResource === undefined ||
      firstTimeline === undefined
    ) {
      throw new Error("generated Fabric evidence lacks a required UI view input");
    }
    for (const artifactValue of [
      firstEdge.source_node_id,
      firstEdge.target_node_id,
      firstExpert.expert_id,
      firstRoute.route_id,
      firstResource.resource_id,
      firstTimeline.evidence_uri,
    ]) expect(host.textContent).toContain(artifactValue);
    expect(
      [...host.querySelectorAll(".fabric-pareto-point")].some((point) =>
        point.getAttribute("aria-label")?.startsWith(firstCandidate.candidate_id),
      ),
    ).toBe(true);
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
