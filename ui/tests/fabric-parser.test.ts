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
});
