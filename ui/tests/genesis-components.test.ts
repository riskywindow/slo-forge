import { describe, expect, it } from "vitest";
import { renderGenesisDashboard } from "../src/genesis-components";
import { genesisFixture } from "./genesis-fixture";

describe("Genesis artifact views", () => {
  it("renders all required artifact-backed sections", () => {
    const html = renderGenesisDashboard(genesisFixture);
    expect(html).toContain('id="genesis-genome"');
    expect(html).toContain('id="genesis-search"');
    expect(html).toContain('id="genesis-verification"');
    expect(html).toContain('id="genesis-benchmark"');
    expect(html).toContain('id="genesis-evolution"');
    expect(html).toContain('id="genesis-lineage"');
    expect(html).toContain("candidate-fast");
    expect(html).toContain("no committed token is emitted after cancellation");
    expect(html).toContain("fabric degradation");
    expect(html).toContain("transformation-related");
  });

  it("labels synthetic evidence and negative results honestly", () => {
    const html = renderGenesisDashboard(genesisFixture);
    expect(html).toContain("CPU / simulator evidence");
    expect(html).toContain("deterministic simulator result, not hardware timing");
    expect(html).toContain("No accepted performance claim");
    expect(html).toContain("Raw benchmark samples</dt><dd>None");
    expect(html).toContain("not external-production eligible");
    expect(html).toContain("makes no speedup claim");
    expect(html).toContain("SEMANTIC_REJECTED");
  });

  it("escapes artifact-controlled content", () => {
    const hostile = structuredClone(genesisFixture);
    hostile.summary.package_id = '<img src=x onerror="alert(1)">';
    const counterexample = hostile.counterexamples[0];
    if (counterexample === undefined) throw new Error("fixture counterexample is missing");
    counterexample.observed.description = "<script>alert(1)</script>";
    const html = renderGenesisDashboard(hostile);
    expect(html).not.toContain("<script>");
    expect(html).not.toContain("<img src=x");
    expect(html).toContain("&lt;script&gt;");
  });
});
