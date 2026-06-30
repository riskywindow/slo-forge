import { describe, expect, it } from "vitest";
import { parseArtifactDocument } from "../src/fabric-parser";
import {
  GenesisArtifactValidationError,
  parseGenesisArtifactBundle,
} from "../src/genesis-parser";
import { genesisFixture } from "./genesis-fixture";

function copyFixture(): unknown {
  return structuredClone(genesisFixture);
}

describe("Genesis artifact parser", () => {
  it("accepts a cross-referenced artifact bundle", () => {
    const parsed = parseGenesisArtifactBundle(copyFixture());
    expect(parsed.summary.accepted_candidate_id).toBe("candidate-corrected");
    expect(parsed.capsule.benchmarks).toEqual([]);
    expect(parsed.baseline_samples).toBeNull();
    expect(parsed.performance_simulation?.comparison_permitted).toBe(false);
    expect(parseArtifactDocument(copyFixture()).kind).toBe("genesis");
  });

  it("rejects a capsule detached from the accepted genome", () => {
    const input = copyFixture() as {
      capsule: { identity: { candidate_genome_hash: { value: string } } };
    };
    input.capsule.identity.candidate_genome_hash.value = "0".repeat(64);
    expect(() => parseGenesisArtifactBundle(input)).toThrowError(
      GenesisArtifactValidationError,
    );
    expect(() => parseGenesisArtifactBundle(input)).toThrow(
      "capsule identity must reference the accepted genome",
    );
  });

  it("rejects missing counterexample evidence and broken lifecycle chains", () => {
    const input = copyFixture() as {
      candidates: { candidate: { lifecycle: { from_state: string | null }[] } }[];
      counterexamples: unknown[];
    };
    input.counterexamples = [];
    const candidate = input.candidates[0];
    const lifecycle = candidate?.candidate.lifecycle[1];
    if (lifecycle === undefined) throw new Error("fixture lifecycle is missing");
    lifecycle.from_state = "WRONG";
    expect(() => parseGenesisArtifactBundle(input)).toThrow(
      "minimized counterexample counterexample-minimized is missing",
    );
    expect(() => parseGenesisArtifactBundle(input)).toThrow(
      "from_state must match prior state",
    );
  });

  it("rejects simulation identity mismatches", () => {
    const input = copyFixture() as {
      performance_simulation: { candidate_genome_hash: string };
    };
    input.performance_simulation.candidate_genome_hash = "9".repeat(64);
    expect(() => parseGenesisArtifactBundle(input)).toThrow(
      "performance_simulation.candidate_genome_hash must match the accepted genome",
    );
  });

  it("rejects invented samples for an unbenchmarked capsule", () => {
    const input = copyFixture() as {
      candidate_samples: unknown;
    };
    input.candidate_samples = { samples: [] };
    expect(() => parseGenesisArtifactBundle(input)).toThrow(
      "unbenchmarked capsule must not expose benchmark definitions or samples",
    );
  });

  it("rejects external eligibility without local eligibility", () => {
    const input = copyFixture() as {
      summary: {
        capsule_external_production_eligible: boolean;
        capsule_local_evolution_eligible: boolean;
      };
    };
    input.summary.capsule_external_production_eligible = true;
    input.summary.capsule_local_evolution_eligible = false;
    expect(() => parseGenesisArtifactBundle(input)).toThrow(
      "external capsule eligibility requires local evolution eligibility",
    );
  });
});
