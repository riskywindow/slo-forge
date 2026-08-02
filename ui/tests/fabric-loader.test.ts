import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchFabricBundleFromManifest,
  MAX_FABRIC_ARTIFACT_BYTES,
} from "../src/fabric-parser";
import { generatedFabricBundle } from "./fabric-generated";

const root = resolve(import.meta.dirname, "../../artifacts/fabric-demo");
const manifestUrl = new URL("http://sloforge.test/artifacts/fabric-demo/manifest.json");

type FetchOptions = {
  bodyOverrides?: Record<string, string>;
  corruptPath?: string;
  missingPath?: string;
  oversizedPath?: string;
};

function artifactFetch(options: FetchOptions = {}): typeof fetch {
  return vi.fn((input: RequestInfo | URL) => {
    const url = new URL(input instanceof Request ? input.url : input.toString());
    const marker = "/artifacts/fabric-demo/";
    const index = url.pathname.indexOf(marker);
    if (index < 0) return Promise.resolve(new Response("not found", { status: 404 }));
    const path = url.pathname.slice(index + marker.length);
    if (path === options.missingPath) {
      return Promise.resolve(new Response("not found", { status: 404 }));
    }
    const body = options.bodyOverrides?.[path] ??
      (path === options.corruptPath ? "{}" : readFileSync(resolve(root, path)));
    const headers: Record<string, string> = { "content-type": "application/json" };
    if (path === options.oversizedPath) {
      headers["content-length"] = String(MAX_FABRIC_ARTIFACT_BYTES + 1);
    }
    return Promise.resolve(new Response(body, { headers }));
  }) as typeof fetch;
}

afterEach(() => vi.unstubAllGlobals());

describe("Fabric manifest loader", () => {
  it("loads the required components and verifies their recorded digests", async () => {
    vi.stubGlobal("fetch", artifactFetch());
    const expected = generatedFabricBundle();
    expect(expected.manifest.artifacts.some((artifact) => artifact.path === "optimizer.json"))
      .toBe(true);
    const actual = await fetchFabricBundleFromManifest(
      manifestUrl,
      expected.manifest,
      new AbortController().signal,
    );
    expect(actual.physical_plan.plan_id).toBe(expected.physical_plan.plan_id);
    expect(actual.counterfactuals.selected_scenario_id).toBe(expected.counterfactuals.selected_scenario_id);
  });

  it("refuses a component whose bytes do not match the evidence manifest", async () => {
    vi.stubGlobal("fetch", artifactFetch({ corruptPath: "topology.json" }));
    await expect(
      fetchFabricBundleFromManifest(
        manifestUrl,
        generatedFabricBundle().manifest,
        new AbortController().signal,
      ),
    ).rejects.toThrow("Integrity mismatch");
  });

  it("rejects missing and duplicate required manifest entries before fetching", async () => {
    const missing = structuredClone(generatedFabricBundle().manifest);
    missing.artifacts = missing.artifacts.filter((artifact) => artifact.path !== "optimizer.json");
    const duplicate = structuredClone(generatedFabricBundle().manifest);
    const firstArtifact = duplicate.artifacts[0];
    if (firstArtifact === undefined) throw new Error("generated manifest has no artifacts");
    duplicate.artifacts.push(structuredClone(firstArtifact));
    const fetch = artifactFetch();
    vi.stubGlobal("fetch", fetch);
    await expect(
      fetchFabricBundleFromManifest(manifestUrl, missing, new AbortController().signal),
    ).rejects.toThrow("required artifact optimizer.json");
    await expect(
      fetchFabricBundleFromManifest(manifestUrl, duplicate, new AbortController().signal),
    ).rejects.toThrow("path must be unique");
    expect(fetch).not.toHaveBeenCalled();
  });

  it("reports malformed, missing, and oversized referenced artifacts", async () => {
    const malformedBytes = "{";
    const malformedManifest = structuredClone(generatedFabricBundle().manifest);
    const optimizer = malformedManifest.artifacts.find(
      (artifact) => artifact.path === "optimizer.json",
    );
    if (optimizer === undefined) throw new Error("generated manifest lacks optimizer.json");
    optimizer.sha256 = createHash("sha256").update(malformedBytes).digest("hex");
    vi.stubGlobal(
      "fetch",
      artifactFetch({ bodyOverrides: { "optimizer.json": malformedBytes } }),
    );
    await expect(
      fetchFabricBundleFromManifest(
        manifestUrl,
        malformedManifest,
        new AbortController().signal,
      ),
    ).rejects.toThrow("is not valid JSON");

    vi.stubGlobal("fetch", artifactFetch({ missingPath: "autopsy/diagnosis.json" }));
    await expect(
      fetchFabricBundleFromManifest(
        manifestUrl,
        generatedFabricBundle().manifest,
        new AbortController().signal,
      ),
    ).rejects.toThrow("HTTP 404");

    vi.stubGlobal("fetch", artifactFetch({ oversizedPath: "fabric-profile.json" }));
    await expect(
      fetchFabricBundleFromManifest(
        manifestUrl,
        generatedFabricBundle().manifest,
        new AbortController().signal,
      ),
    ).rejects.toThrow("exceeds the");
  });
});
