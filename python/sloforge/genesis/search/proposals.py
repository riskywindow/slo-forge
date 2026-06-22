"""Credential-free deterministic beam, evolutionary, local, and novelty proposals."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Collection, Iterable, Sequence

from pydantic import model_validator

from sloforge.genesis.ir import ArtifactDigest

from .models import (
    CandidateDesign,
    MutationChoice,
    NonNegativeInt,
    PositiveInt,
    Region,
    SearchModel,
)


class ProposalRequest(SearchModel):
    base_genome_hash: ArtifactDigest
    seed: NonNegativeInt
    base_features: tuple[float, ...]
    options: tuple[MutationChoice, ...]
    parents: tuple[CandidateDesign, ...] = ()
    mutable_regions: tuple[Region, ...]
    maximum_proposals: PositiveInt

    @model_validator(mode="after")
    def dimensions_and_bounds(self) -> ProposalRequest:
        if not self.base_features or len(self.base_features) > 64:
            raise ValueError("base feature vector must contain between 1 and 64 values")
        if not self.options:
            raise ValueError("proposal options cannot be empty")
        if any(len(option.feature_delta) != len(self.base_features) for option in self.options):
            raise ValueError("mutation and base feature dimensions must match")
        if len(self.mutable_regions) != len(set(self.mutable_regions)):
            raise ValueError("mutable region whitelist must be unique")
        return self


def _signature(mutations: Sequence[MutationChoice]) -> tuple[str, ...]:
    return tuple(sorted(mutation.transformation_id for mutation in mutations))


def _legal(mutation: MutationChoice, mutable_regions: Collection[Region]) -> bool:
    return set(mutation.regions).issubset(mutable_regions)


def _make_design(
    request: ProposalRequest,
    mutations: Sequence[MutationChoice],
    *,
    engine: str,
    ordinal: int,
    parents: Sequence[str] = (),
) -> CandidateDesign:
    ordered = tuple(sorted(mutations, key=lambda item: item.transformation_id))
    identity = {
        "base_genome_hash": request.base_genome_hash.value,
        "engine": engine,
        "mutations": [
            {
                "family": mutation.family.value,
                "parameters": [(item.key, item.value) for item in mutation.parameters],
                "regions": mutation.regions,
                "transformation_id": mutation.transformation_id,
            }
            for mutation in ordered
        ],
        "ordinal": ordinal,
        "parents": tuple(parents),
        "seed": request.seed,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    features = tuple(
        base + sum(mutation.feature_delta[index] for mutation in ordered)
        for index, base in enumerate(request.base_features)
    )
    seed = int.from_bytes(hashlib.sha256(f"{request.seed}:{digest}".encode()).digest()[:8], "big")
    return CandidateDesign(
        candidate_id=f"candidate-{digest[:24]}",
        seed=seed,
        genome_hash=ArtifactDigest(value=digest),
        parent_candidate_ids=tuple(parents),
        mutations=ordered,
        feature_vector=features,
        proposal_engine=engine,  # type: ignore[arg-type]
    )


class BeamProposalEngine:
    name = "beam"

    def propose(self, request: ProposalRequest) -> tuple[CandidateDesign, ...]:
        mutable = set(request.mutable_regions)
        options = tuple(option for option in request.options if _legal(option, mutable))
        combinations: list[tuple[MutationChoice, ...]] = [(option,) for option in options]
        combinations.extend(
            pair
            for pair in itertools.combinations(options, 2)
            if set(pair[0].regions) != set(pair[1].regions)
        )
        combinations.sort(
            key=lambda choices: (
                -sum(item.expected_upside - item.invalidity_risk for item in choices),
                _signature(choices),
            )
        )
        return tuple(
            _make_design(request, choices, engine=self.name, ordinal=index)
            for index, choices in enumerate(combinations[: request.maximum_proposals])
        )


class EvolutionaryProposalEngine:
    name = "evolutionary"

    def propose(self, request: ProposalRequest) -> tuple[CandidateDesign, ...]:
        if len(request.parents) < 2:
            return ()
        mutable = set(request.mutable_regions)
        generator = random.Random(request.seed)
        parent_pairs = list(itertools.combinations(request.parents, 2))
        generator.shuffle(parent_pairs)
        results: list[CandidateDesign] = []
        for left, right in parent_pairs:
            by_id = {
                mutation.transformation_id: mutation
                for mutation in (*left.mutations, *right.mutations)
                if _legal(mutation, mutable)
            }
            mutations = list(by_id.values())
            generator.shuffle(mutations)
            keep = max(1, (len(mutations) + 1) // 2)
            chosen = mutations[:keep]
            available = [
                option
                for option in request.options
                if option.transformation_id not in by_id and _legal(option, mutable)
            ]
            if available:
                chosen.append(available[generator.randrange(len(available))])
            results.append(
                _make_design(
                    request,
                    chosen,
                    engine=self.name,
                    ordinal=len(results),
                    parents=(left.candidate_id, right.candidate_id),
                )
            )
            if len(results) >= request.maximum_proposals:
                break
        return tuple(results)


class LocalProposalEngine:
    name = "local"

    def propose(self, request: ProposalRequest) -> tuple[CandidateDesign, ...]:
        mutable = set(request.mutable_regions)
        results: list[CandidateDesign] = []
        for parent in sorted(request.parents, key=lambda item: item.candidate_id):
            existing = {mutation.transformation_id for mutation in parent.mutations}
            choices = sorted(
                (
                    option
                    for option in request.options
                    if option.transformation_id not in existing and _legal(option, mutable)
                ),
                key=lambda item: (
                    -(item.expected_upside - item.invalidity_risk),
                    item.transformation_id,
                ),
            )
            if not choices:
                continue
            results.append(
                _make_design(
                    request,
                    (*parent.mutations, choices[0]),
                    engine=self.name,
                    ordinal=len(results),
                    parents=(parent.candidate_id,),
                )
            )
            if len(results) >= request.maximum_proposals:
                break
        return tuple(results)


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else 1.0 - len(left & right) / len(union)


class NoveltyProposalEngine:
    name = "novelty"

    def propose(self, request: ProposalRequest) -> tuple[CandidateDesign, ...]:
        mutable = set(request.mutable_regions)
        prior = [
            {mutation.transformation_id for mutation in parent.mutations}
            for parent in request.parents
        ]
        options = [option for option in request.options if _legal(option, mutable)]
        options.sort(
            key=lambda option: (
                -min(
                    (
                        _jaccard_distance({option.transformation_id}, signature)
                        for signature in prior
                    ),
                    default=1.0,
                ),
                option.transformation_id,
            )
        )
        results: list[CandidateDesign] = []
        for option in options[: request.maximum_proposals]:
            results.append(_make_design(request, (option,), engine=self.name, ordinal=len(results)))
        return tuple(results)


class ProposalPortfolio:
    """Round-robin engines while preserving structurally distinct candidates."""

    def __init__(self) -> None:
        self._engines = (
            BeamProposalEngine(),
            EvolutionaryProposalEngine(),
            LocalProposalEngine(),
            NoveltyProposalEngine(),
        )

    def propose(self, request: ProposalRequest) -> tuple[CandidateDesign, ...]:
        per_engine = max(1, request.maximum_proposals)
        queues = [
            list(engine.propose(request.model_copy(update={"maximum_proposals": per_engine})))
            for engine in self._engines
        ]
        results: list[CandidateDesign] = []
        seen: set[tuple[str, ...]] = set()
        while len(results) < request.maximum_proposals and any(queues):
            for queue in queues:
                if not queue:
                    continue
                candidate = queue.pop(0)
                signature = _signature(candidate.mutations)
                if signature in seen:
                    continue
                seen.add(signature)
                results.append(candidate)
                if len(results) >= request.maximum_proposals:
                    break
        return tuple(results)


def mutation_signatures(candidates: Iterable[CandidateDesign]) -> tuple[tuple[str, ...], ...]:
    return tuple(_signature(candidate.mutations) for candidate in candidates)
