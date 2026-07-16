"""Ancestry commitments included in Merkle-covered capsule evidence."""

from __future__ import annotations

from sloforge.continuum.ir import Digest, VerificationClaim, canonical_hash

from .models import AncestryKind, AncestryProof

ANCESTRY_CLAIM_ID = "continuum-checkpoint-ancestry-v1"


def ancestry_payload(
    *,
    kind: AncestryKind,
    generation: int,
    parent_capsule_id: str | None,
    parent_manifest_id: str | None,
    current_manifest_id: str,
) -> dict[str, object]:
    return {
        "schema": "sloforge.continuum.checkpoint-ancestry/v1",
        "kind": kind.value,
        "generation": generation,
        "parent_capsule_id": parent_capsule_id,
        "parent_manifest_id": parent_manifest_id,
        "current_manifest_id": current_manifest_id,
    }


def make_ancestry_proof(
    *,
    kind: AncestryKind,
    generation: int,
    parent_capsule_id: str | None,
    parent_manifest_id: str | None,
    current_manifest_id: str,
) -> AncestryProof:
    digest = canonical_hash(
        ancestry_payload(
            kind=kind,
            generation=generation,
            parent_capsule_id=parent_capsule_id,
            parent_manifest_id=parent_manifest_id,
            current_manifest_id=current_manifest_id,
        )
    )
    return AncestryProof(
        kind=kind,
        generation=generation,
        parent_capsule_id=parent_capsule_id,
        parent_manifest_id=parent_manifest_id,
        current_manifest_id=current_manifest_id,
        digest=digest,
    )


def ancestry_claim(proof: AncestryProof) -> VerificationClaim:
    return VerificationClaim(
        claim_id=ANCESTRY_CLAIM_ID,
        property="checkpoint parent and CAS publication are integrity-bound",
        scope=f"{proof.kind.value}:generation:{proof.generation}",
        result="pass",
        evidence_digest=Digest(value=proof.digest),
    )
