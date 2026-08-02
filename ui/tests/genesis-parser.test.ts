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

  it("rejects benchmark provenance mismatches", () => {
    const input = copyFixture() as {
      candidate_samples: { hardware_fingerprint: { value: string } };
    };
    input.candidate_samples.hardware_fingerprint.value = "9".repeat(64);
    expect(() => parseGenesisArtifactBundle(input)).toThrow(
      "baseline and candidate hardware fingerprints must match",
    );
  });

  it("rejects workload provenance mismatches", () => {
    const input = copyFixture() as {
      candidate_samples: { workload_fingerprint: { value: string } };
    };
    input.candidate_samples.workload_fingerprint.value = "9".repeat(64);
    expect(() => parseGenesisArtifactBundle(input)).toThrow(
      "baseline and candidate workload fingerprints must match",
    );
  });
});
