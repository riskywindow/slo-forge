import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseFabricArtifactBundle } from "../src/fabric-parser";
import type { FabricArtifactBundle } from "../src/fabric-types";

const root = resolve(import.meta.dirname, "../../artifacts/fabric-demo");

function json(path: string): unknown {
  return JSON.parse(readFileSync(resolve(root, path), "utf8")) as unknown;
}

export function generatedFabricBundle(): FabricArtifactBundle {
  return parseFabricArtifactBundle({
    artifact_type: "sloforge.fabric.ui-bundle/v1",
    counterfactuals: json("autopsy/counterfactuals.json"),
    diagnosis: json("autopsy/diagnosis.json"),
    fabric_profile: json("fabric-profile.json"),
    manifest: json("manifest.json"),
    optimizer: json("optimizer.json"),
    physical_plan: json("physical-plan.json"),
    recovery_execution: json("recovery/execution.json"),
    recovery_plan: json("recovery/proposal.json"),
    simulations: {
      degraded: json("simulations/degraded.json"),
      healthy: json("simulations/healthy.json"),
      restored: json("simulations/restored.json"),
    },
    timeline: json("timeline.json"),
    topology: json("topology.json"),
  });
}
