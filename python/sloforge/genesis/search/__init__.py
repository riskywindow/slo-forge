"""Deterministic multi-stage Genesis candidate search."""

from .budget import BudgetExceeded, BudgetManager
from .engine import SearchEngine, StageEvaluator
from .lifecycle import CandidateRepository, CanonicalEventStore
from .models import (
    CandidateDesign,
    CandidateSearchResult,
    FidelityStage,
    MutationChoice,
    ObjectiveVector,
    ParameterValue,
    SearchConfiguration,
    SearchEvent,
    SearchRunResult,
    StageResult,
    StageSpecification,
)
from .pareto import ArchiveEntry, ParetoArchive, dominates
from .proposals import (
    BeamProposalEngine,
    EvolutionaryProposalEngine,
    LocalProposalEngine,
    NoveltyProposalEngine,
    ProposalPortfolio,
    ProposalRequest,
)
from .surrogate import DeterministicSurrogateRanker, ExperimentScore, SurrogateObservation

__all__ = [
    "ArchiveEntry",
    "BeamProposalEngine",
    "BudgetExceeded",
    "BudgetManager",
    "CandidateDesign",
    "CandidateRepository",
    "CandidateSearchResult",
    "CanonicalEventStore",
    "DeterministicSurrogateRanker",
    "EvolutionaryProposalEngine",
    "ExperimentScore",
    "FidelityStage",
    "LocalProposalEngine",
    "MutationChoice",
    "NoveltyProposalEngine",
    "ObjectiveVector",
    "ParameterValue",
    "ParetoArchive",
    "ProposalPortfolio",
    "ProposalRequest",
    "SearchConfiguration",
    "SearchEngine",
    "SearchEvent",
    "SearchRunResult",
    "StageEvaluator",
    "StageResult",
    "StageSpecification",
    "SurrogateObservation",
    "dominates",
]
