import { describe, expect, it } from "vitest";
import {
  counterfactualRanking,
  hostIds,
  kvLinkEvidence,
  placementsByHost,
  requestMetricComparisons,
  resourceHotspots,
  topologyInventory,
  workerPools,
} from "../src/fabric-transforms";
import { generatedFabricBundle } from "./fabric-generated";

describe("Fabric data transforms", () => {
  const bundle = generatedFabricBundle();

  it("preserves every physical binding while grouping by discovered hosts", () => {
    const grouped = placementsByHost(bundle);
    expect([...grouped.values()].flat()).toHaveLength(bundle.physical_plan.rank_placement.bindings.length);
    expect([...grouped.keys()].sort()).toEqual(hostIds(bundle));
    expect(workerPools(bundle).map((pool) => pool.kind)).toEqual(["prefill", "decode"]);
  });

  it("derives topology inventory and request deltas from source artifacts", () => {
    const inventory = topologyInventory(bundle);
    expect(inventory.nodes.reduce((sum, item) => sum + item.count, 0)).toBe(bundle.topology.nodes.length);
    expect(inventory.connections.reduce((sum, item) => sum + item.count, 0)).toBe(bundle.topology.edges.length);
    const ttft = requestMetricComparisons(bundle).find((item) => item.name === "p95_ttft_ms");
    expect(ttft?.healthy).toBe(bundle.manifest.healthy.p95_ttft_ms);
    expect(ttft?.degraded).toBe(bundle.manifest.degraded.p95_ttft_ms);
    expect(ttft?.restored).toBe(bundle.manifest.restored.p95_ttft_ms);
  });

  it("orders counterfactuals and hotspots using recorded outcomes", () => {
    const counterfactuals = counterfactualRanking(bundle);
    expect(counterfactuals[0]?.expected_improvement_ms).toBeGreaterThanOrEqual(
      counterfactuals.at(-1)?.expected_improvement_ms ?? 0,
    );
    const hotspots = resourceHotspots(bundle);
    expect(hotspots[0]?.degraded).toBeGreaterThanOrEqual(hotspots.at(-1)?.degraded ?? 0);
  });

  it("joins selected KV paths to calibrated edges and raw measurements", () => {
    const evidence = kvLinkEvidence(bundle);
    expect(evidence).toHaveLength(bundle.physical_plan.kv_transfer.routes.length);
    expect(evidence[0]?.predictedLinkLatencyUs).toBeGreaterThan(0);
    expect(evidence[0]?.measurementId).toContain("kv_transfer");
    expect(evidence[0]?.rawSampleCount).toBeGreaterThan(0);
  });
});
