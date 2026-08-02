import { describe, expect, it } from "vitest";
import {
  FabricArtifactValidationError,
  parseArtifactDocument,
  parseFabricArtifactBundle,
} from "../src/fabric-parser";
import { generatedFabricBundle } from "./fabric-generated";

describe("Fabric artifact parser", () => {
  it("validates the generated multi-artifact demo contract", () => {
    const bundle = generatedFabricBundle();
    expect(bundle.manifest.physical_plan_id).toBe(bundle.physical_plan.plan_id);
    expect(bundle.topology.nodes.length).toBeGreaterThan(0);
    expect(bundle.fabric_profile.measurements.every((item) => item.samples.length > 0)).toBe(true);
    expect(parseArtifactDocument(bundle).kind).toBe("fabric");
  });

  it("rejects a rank binding with missing physical affinity", () => {
    const bundle = structuredClone(generatedFabricBundle()) as unknown as Record<string, unknown>;
    const plan = bundle["physical_plan"] as Record<string, unknown>;
    const placement = plan["rank_placement"] as Record<string, unknown>;
    const bindings = placement["bindings"] as Record<string, unknown>[];
    delete bindings[0]?.["gpu_id"];
    expect(() => parseFabricArtifactBundle(bundle)).toThrow(FabricArtifactValidationError);
    expect(() => parseFabricArtifactBundle(bundle)).toThrow("gpu_id");
  });

  it("rejects dangling topology, rank, and KV path references", () => {
    const topologyBundle = structuredClone(generatedFabricBundle());
    const firstEdge = topologyBundle.topology.edges[0];
    if (firstEdge === undefined) throw new Error("generated topology has no edges");
    firstEdge.source_node_id = "missing-host";
    expect(() => parseFabricArtifactBundle(topologyBundle)).toThrow(
      "source_node_id must reference a topology node",
    );

    const rankBundle = structuredClone(generatedFabricBundle());
    const firstGroup = rankBundle.physical_plan.parallelism.groups[0];
    if (firstGroup === undefined) throw new Error("generated plan has no parallel group");
    firstGroup.rank_ids.push(99_999);
    expect(() => parseFabricArtifactBundle(rankBundle)).toThrow("must reference a placed rank");

    const kvBundle = structuredClone(generatedFabricBundle());
    const firstRoute = kvBundle.physical_plan.kv_transfer.routes[0];
    if (firstRoute === undefined) throw new Error("generated plan has no KV route");
    firstRoute.edge_path.push("missing-edge");
    expect(() => parseFabricArtifactBundle(kvBundle)).toThrow(
      "must reference a topology edge",
    );
  });

  it("rejects inconsistent diagnosis, counterfactual, recovery, and optimizer identities", () => {
    const diagnosisBundle = structuredClone(generatedFabricBundle());
    diagnosisBundle.manifest.diagnosis_confidence = 0.01;
    expect(() => parseFabricArtifactBundle(diagnosisBundle)).toThrow(
      "diagnosis_confidence must match",
    );

    const counterfactualBundle = structuredClone(generatedFabricBundle());
    counterfactualBundle.manifest.selected_counterfactual = "missing-scenario";
    expect(() => parseFabricArtifactBundle(counterfactualBundle)).toThrow(
      "selected_counterfactual must match",
    );

    const recoveryBundle = structuredClone(generatedFabricBundle());
    recoveryBundle.recovery_plan.physical_plan.uid = "other-plan";
    expect(() => parseFabricArtifactBundle(recoveryBundle)).toThrow(
      "recovery_plan.physical_plan.uid must match",
    );

    const optimizerBundle = structuredClone(generatedFabricBundle());
    optimizerBundle.optimizer.selected.plan_id = "other-plan";
    expect(() => parseFabricArtifactBundle(optimizerBundle)).toThrow(
      "optimizer.selected.plan_id must match",
    );
  });

  it("rejects synthetic evidence mislabeled as hardware measured", () => {
    const bundle = structuredClone(generatedFabricBundle());
    bundle.manifest.synthetic_hardware = false;
    expect(() => parseFabricArtifactBundle(bundle)).toThrow(
      "synthetic_hardware must match the Fabric profile measurement mode",
    );
  });

  it("rejects empty evidence required by the flagship views", () => {
    const bundle = structuredClone(generatedFabricBundle());
    bundle.optimizer.pareto_frontier = [];
    expect(() => parseFabricArtifactBundle(bundle)).toThrow(
      "$.optimizer.pareto_frontier must not be empty",
    );
  });
});
