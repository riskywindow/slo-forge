import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { parseGenesisArtifactBundle } from "../src/genesis-parser";

const generatedBundle = process.env["GENESIS_UI_BUNDLE"];

describe("generated Genesis UI bundle", () => {
  const generated = generatedBundle === undefined ? it.skip : it;

  generated("parses a bundle emitted by the Python flagship demo", async () => {
    if (generatedBundle === undefined) throw new Error("GENESIS_UI_BUNDLE is missing");
    const input = JSON.parse(await readFile(generatedBundle, "utf8")) as unknown;
    const parsed = parseGenesisArtifactBundle(input);
    expect(parsed.artifact_type).toBe("sloforge.genesis.ui-bundle/v1");
    expect(parsed.summary.accepted_candidate_id).toBe(
      parsed.candidates.find(({ candidate }) =>
        candidate.genome_hash.value === parsed.summary.accepted_genome_hash,
      )?.candidate.candidate_id,
    );
    expect(parsed.evolution.audit.some(({ action }) => action === "promote")).toBe(true);
    expect(parsed.lineage.stale_seed_suppressed_after_invalidation).toBe(true);
  });
});
