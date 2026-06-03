import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchFabricBundleFromManifest } from "../src/fabric-parser";
import { generatedFabricBundle } from "./fabric-generated";

const root = resolve(import.meta.dirname, "../../artifacts/fabric-demo");
const manifestUrl = new URL("http://sloforge.test/artifacts/fabric-demo/manifest.json");

function manifestWithOptimizer(): ReturnType<typeof generatedFabricBundle>["manifest"] {
  const manifest = structuredClone(generatedFabricBundle().manifest);
  if (!manifest.artifacts.some((artifact) => artifact.path === "optimizer.json")) {
    const digest = createHash("sha256")
      .update(readFileSync(resolve(root, "optimizer.json")))
      .digest("hex");
    manifest.artifacts.push({ path: "optimizer.json", sha256: digest });
  }
  return manifest;
}

function artifactFetch(corruptPath?: string): typeof fetch {
  return vi.fn((input: RequestInfo | URL) => {
    const url = new URL(input instanceof Request ? input.url : input.toString());
    const marker = "/artifacts/fabric-demo/";
    const index = url.pathname.indexOf(marker);
    if (index < 0) return Promise.resolve(new Response("not found", { status: 404 }));
    const path = url.pathname.slice(index + marker.length);
    const body = path === corruptPath ? "{}" : readFileSync(resolve(root, path));
    return Promise.resolve(new Response(body, { headers: { "content-type": "application/json" } }));
  }) as typeof fetch;
}

afterEach(() => vi.unstubAllGlobals());

describe("Fabric manifest loader", () => {
  it("loads the required components and verifies their recorded digests", async () => {
    vi.stubGlobal("fetch", artifactFetch());
    const expected = generatedFabricBundle();
    const actual = await fetchFabricBundleFromManifest(
      manifestUrl,
      manifestWithOptimizer(),
      new AbortController().signal,
    );
    expect(actual.physical_plan.plan_id).toBe(expected.physical_plan.plan_id);
    expect(actual.counterfactuals.selected_scenario_id).toBe(expected.counterfactuals.selected_scenario_id);
  });

  it("refuses a component whose bytes do not match the evidence manifest", async () => {
    vi.stubGlobal("fetch", artifactFetch("topology.json"));
    await expect(
      fetchFabricBundleFromManifest(
        manifestUrl,
        manifestWithOptimizer(),
        new AbortController().signal,
      ),
    ).rejects.toThrow("Integrity mismatch");
  });
});
import { createHash } from "node:crypto";
