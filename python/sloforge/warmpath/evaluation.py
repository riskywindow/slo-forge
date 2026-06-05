"""Reproducible H6 comparison for portable WarmPath startup strategies.

The evaluation bootstraps the checked-in measured local profile. Strategies that
cannot be exercised on the current host are explicitly marked as modeled proxies.
No timing constant is presented as a hardware measurement.
"""

from __future__ import annotations

import argparse
import random
import shutil
import statistics
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, model_validator

from sloforge.util import (
    environment_manifest,
    sha256_file,
    write_json,
)
from sloforge.warmpath.io import load_graph, load_plan
from sloforge.warmpath.models import (
    ArtifactGraph,
    MaterializationMode,
    StartupProfile,
    StartupStage,
    StorageTierSpec,
    WarmPathPlan,
)
from sloforge.warmpath.profiler import load_profile
from sloforge.warmpath.statistics import bootstrap_median_interval, percentile

EVALUATION_SCHEMA_VERSION: Literal["sloforge.warmpath.evaluation/v1"] = (
    "sloforge.warmpath.evaluation/v1"
)

PositiveInt = Annotated[int, Field(gt=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class WarmPathStrategy(StrEnum):
    COLD_UNCACHED = "cold_uncached"
    LOCAL_DISK_CACHE = "local_disk_cache"
    PAGE_CACHE = "page_cache"
    PINNED_HOST_MEMORY_MODELED = "pinned_host_memory_modeled"
    WARMPATH = "warmpath"
    WARM_REPLICA = "warm_replica"


class EvaluationScenario(StrEnum):
    NOMINAL = "nominal"
    EVICTION_PRESSURE = "eviction_pressure"


class WarmPathEvaluationConfig(_EvaluationModel):
    schema_version: Literal["sloforge.warmpath.evaluation-config/v1"] = (
        "sloforge.warmpath.evaluation-config/v1"
    )
    graph_path: str
    profile_path: str
    plan_path: str
    seeds: tuple[int, ...]
    trials_per_seed: Annotated[int, Field(ge=11)]
    bootstrap_rounds: Annotated[int, Field(ge=200)] = 2_000
    cold_tier_id: str = "local-nvme"
    local_disk_tier_id: str = "local-nvme"
    page_cache_tier_id: str = "page-cache"
    host_memory_tier_id: str = "host-memory"
    eviction_probability: Probability = 0.25
    restore_failure_probability: Probability = 0.03
    warm_replica_failure_probability: Probability = 0.01
    warm_replica_hourly_cost_usd: NonNegativeFloat

    @model_validator(mode="after")
    def validate_seeds(self) -> WarmPathEvaluationConfig:
        if len(self.seeds) < 3 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("evaluation requires at least three unique seeds")
        if any(not value.strip() for value in (self.graph_path, self.profile_path, self.plan_path)):
            raise ValueError("evaluation input paths cannot be empty")
        return self


class WarmPathTrial(_EvaluationModel):
    strategy: WarmPathStrategy
    scenario: EvaluationScenario
    seed: int
    trial_index: Annotated[int, Field(ge=0)]
    ready_time_ms: NonNegativeFloat | None
    restore_failed: bool
    eviction_occurred: bool
    transferred_bytes: Annotated[int, Field(ge=0)]
    storage_bytes: Annotated[int, Field(ge=0)]
    hourly_cost_usd: NonNegativeFloat


class WarmPathStrategySummary(_EvaluationModel):
    strategy: WarmPathStrategy
    measurement_basis: str
    seed_count: PositiveInt
    trials_per_scenario: PositiveInt
    successful_nominal_trials: Annotated[int, Field(ge=0)]
    p50_ready_time_ms: NonNegativeFloat
    p95_ready_time_ms: NonNegativeFloat
    seed_median_p95_ready_time_ms: NonNegativeFloat
    p95_confidence_interval_low_ms: NonNegativeFloat
    p95_confidence_interval_high_ms: NonNegativeFloat
    failure_rate: Probability
    failure_rate_confidence_interval_low: Probability
    failure_rate_confidence_interval_high: Probability
    hourly_cost_usd: NonNegativeFloat
    storage_bytes: Annotated[int, Field(ge=0)]
    median_transfer_bytes: NonNegativeFloat
    eviction_p95_ready_time_ms: NonNegativeFloat
    eviction_absolute_penalty_ms: float
    eviction_relative_penalty_percent: float | None

    @model_validator(mode="after")
    def validate_intervals(self) -> WarmPathStrategySummary:
        if not (
            self.p95_confidence_interval_low_ms
            <= self.seed_median_p95_ready_time_ms
            <= self.p95_confidence_interval_high_ms
        ):
            raise ValueError("p95 confidence interval must contain the seed median")
        if not (
            self.failure_rate_confidence_interval_low
            <= self.failure_rate
            <= self.failure_rate_confidence_interval_high
        ):
            raise ValueError("failure interval must contain the observed failure rate")
        return self


class WarmPathEvaluationResult(_EvaluationModel):
    schema_version: Literal["sloforge.warmpath.evaluation/v1"] = EVALUATION_SCHEMA_VERSION
    hypothesis: Literal["H6"] = "H6"
    config_sha256: Sha256
    graph_sha256: Sha256
    profile_sha256: Sha256
    plan_sha256: Sha256
    raw_trials_sha256: Sha256
    strategies: tuple[WarmPathStrategySummary, ...]
    best_non_warm_strategy: WarmPathStrategy
    warmpath_vs_local_disk_p95_improvement_percent: float
    warmpath_vs_best_non_warm_p95_improvement_percent: float
    all_timings_hardware_measured: bool
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_strategies(self) -> WarmPathEvaluationResult:
        expected = set(WarmPathStrategy)
        observed = {item.strategy for item in self.strategies}
        if observed != expected:
            raise ValueError(f"strategy set differs: expected {expected}, got {observed}")
        if self.all_timings_hardware_measured:
            raise ValueError("this fixture comparison includes explicitly modeled strategies")
        return self


class EvaluationArtifact(_EvaluationModel):
    path: str
    sha256: Sha256


class WarmPathEvaluationManifest(_EvaluationModel):
    schema_version: Literal["sloforge.warmpath.evaluation-manifest/v1"] = (
        "sloforge.warmpath.evaluation-manifest/v1"
    )
    command: str
    result_path: str
    raw_trial_count: PositiveInt
    artifacts: tuple[EvaluationArtifact, ...]


_MEASUREMENT_BASIS: dict[WarmPathStrategy, str] = {
    WarmPathStrategy.COLD_UNCACHED: (
        "declared-tier-bandwidth-floor plus bootstrapped measured checksum verification"
    ),
    WarmPathStrategy.LOCAL_DISK_CACHE: "bootstrapped measured local-NVMe fetch and verification",
    WarmPathStrategy.PAGE_CACHE: "bootstrapped measured page-cache fetch and verification",
    WarmPathStrategy.PINNED_HOST_MEMORY_MODELED: (
        "bootstrapped measured host-memory copy used as a pinned-memory proxy"
    ),
    WarmPathStrategy.WARMPATH: "compiled WarmPath placements with bootstrapped measured stages",
    WarmPathStrategy.WARM_REPLICA: "modeled already-ready replica with declared hourly cost",
}


def load_evaluation_config(path: Path) -> WarmPathEvaluationConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("WarmPath evaluation configuration must be a mapping")
    normalized = dict(payload)
    seeds = normalized.get("seeds")
    if isinstance(seeds, list):
        normalized["seeds"] = tuple(seeds)
    return WarmPathEvaluationConfig.model_validate(normalized)


def _profile_inputs(
    config: WarmPathEvaluationConfig, *, repository_root: Path
) -> tuple[ArtifactGraph, StartupProfile, WarmPathPlan]:
    graph = load_graph(repository_root / config.graph_path)
    profile = load_profile(repository_root / config.profile_path)
    plan = load_plan(repository_root / config.plan_path)
    if graph.graph_id != profile.graph_id or graph.graph_id != plan.graph_id:
        raise ValueError("WarmPath graph, profile, and plan identifiers differ")
    if profile.profile_id != plan.profile_id:
        raise ValueError("WarmPath profile and plan identifiers differ")
    tier_ids = {item.tier_id for item in profile.tiers}
    configured = {
        config.cold_tier_id,
        config.local_disk_tier_id,
        config.page_cache_tier_id,
        config.host_memory_tier_id,
    }
    if missing := configured - tier_ids:
        raise ValueError(f"evaluation storage tiers are absent from the profile: {sorted(missing)}")
    return graph, profile, plan


def _measurements(
    profile: StartupProfile,
) -> dict[tuple[str, str, StartupStage], tuple[float, ...]]:
    return {
        (item.artifact_id, item.tier_id, item.stage): tuple(item.raw_samples_ms)
        for item in profile.measurements
    }


def _strategy_tiers(
    strategy: WarmPathStrategy,
    *,
    graph: ArtifactGraph,
    plan: WarmPathPlan,
    config: WarmPathEvaluationConfig,
    eviction: bool,
) -> dict[str, str | None]:
    by_id = {item.artifact_id: item for item in graph.artifacts}
    required = {item.artifact_id for item in graph.artifacts if item.required_for_readiness}
    pending = list(required)
    while pending:
        for dependency in by_id[pending.pop()].dependencies:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    if strategy == WarmPathStrategy.WARM_REPLICA and not eviction:
        return {artifact_id: None for artifact_id in required}
    if eviction and strategy in {
        WarmPathStrategy.PAGE_CACHE,
        WarmPathStrategy.PINNED_HOST_MEMORY_MODELED,
        WarmPathStrategy.WARMPATH,
    }:
        return {artifact_id: config.local_disk_tier_id for artifact_id in required}
    if strategy == WarmPathStrategy.WARMPATH and not eviction:
        return {
            item.artifact_id: item.tier_id
            for item in plan.placements
            if item.artifact_id in required and item.mode != MaterializationMode.LAZY_RESTORE
        }
    selected = {
        WarmPathStrategy.COLD_UNCACHED: config.cold_tier_id,
        WarmPathStrategy.LOCAL_DISK_CACHE: config.local_disk_tier_id,
        WarmPathStrategy.PAGE_CACHE: config.page_cache_tier_id,
        WarmPathStrategy.PINNED_HOST_MEMORY_MODELED: config.host_memory_tier_id,
        WarmPathStrategy.WARMPATH: config.local_disk_tier_id,
        WarmPathStrategy.WARM_REPLICA: config.cold_tier_id,
    }[strategy]
    return {artifact_id: selected for artifact_id in required}


def _sample_duration_ms(
    *,
    artifact_id: str,
    artifact_bytes: int,
    tier: StorageTierSpec,
    measurement_map: dict[tuple[str, str, StartupStage], tuple[float, ...]],
    cold_uncached: bool,
    rng: random.Random,
) -> float:
    fetch = measurement_map.get((artifact_id, tier.tier_id, StartupStage.FETCH))
    verify = measurement_map.get((artifact_id, tier.tier_id, StartupStage.VERIFY))
    if fetch is None or verify is None:
        raise ValueError(f"profile has no measured samples for {artifact_id} on {tier.tier_id}")
    fetch_ms = rng.choice(fetch)
    if cold_uncached:
        theoretical_ms = tier.base_read_latency_ms + (
            1_000.0 * artifact_bytes / tier.read_bandwidth_bytes_per_second
        )
        fetch_ms = max(fetch_ms, theoretical_ms)
    return fetch_ms + rng.choice(verify)


def _hourly_cost(
    tier_by_artifact: dict[str, str | None],
    *,
    graph: ArtifactGraph,
    tiers: dict[str, StorageTierSpec],
    strategy: WarmPathStrategy,
    config: WarmPathEvaluationConfig,
) -> tuple[float, int]:
    if strategy == WarmPathStrategy.WARM_REPLICA and all(
        value is None for value in tier_by_artifact.values()
    ):
        resident_bytes = sum(
            item.size_bytes for item in graph.artifacts if item.artifact_id in tier_by_artifact
        )
        return float(config.warm_replica_hourly_cost_usd), resident_bytes
    artifacts = {item.artifact_id: item for item in graph.artifacts}
    storage_bytes = sum(artifacts[item].size_bytes for item in tier_by_artifact)
    cost = 0.0
    for item, tier in tier_by_artifact.items():
        if tier is not None:
            cost += artifacts[item].size_bytes / (1024.0**3) * tiers[tier].hourly_cost_per_gib
    return cost, storage_bytes


def _trial(
    *,
    strategy: WarmPathStrategy,
    scenario: EvaluationScenario,
    seed: int,
    trial_index: int,
    graph: ArtifactGraph,
    profile: StartupProfile,
    plan: WarmPathPlan,
    config: WarmPathEvaluationConfig,
    measurement_map: dict[tuple[str, str, StartupStage], tuple[float, ...]],
    rng: random.Random,
) -> WarmPathTrial:
    eviction_eligible = strategy in {
        WarmPathStrategy.PAGE_CACHE,
        WarmPathStrategy.PINNED_HOST_MEMORY_MODELED,
        WarmPathStrategy.WARMPATH,
        WarmPathStrategy.WARM_REPLICA,
    }
    eviction = (
        scenario == EvaluationScenario.EVICTION_PRESSURE
        and eviction_eligible
        and rng.random() < config.eviction_probability
    )
    tier_by_artifact = _strategy_tiers(
        strategy,
        graph=graph,
        plan=plan,
        config=config,
        eviction=eviction,
    )
    failure_probability = (
        config.warm_replica_failure_probability
        if strategy == WarmPathStrategy.WARM_REPLICA and not eviction
        else config.restore_failure_probability
    )
    failed = rng.random() < failure_probability
    tiers = {item.tier_id: item for item in profile.tiers}
    hourly_cost, storage_bytes = _hourly_cost(
        tier_by_artifact,
        graph=graph,
        tiers=tiers,
        strategy=strategy,
        config=config,
    )
    if failed:
        return WarmPathTrial(
            strategy=strategy,
            scenario=scenario,
            seed=seed,
            trial_index=trial_index,
            ready_time_ms=None,
            restore_failed=True,
            eviction_occurred=eviction,
            transferred_bytes=0,
            storage_bytes=storage_bytes,
            hourly_cost_usd=hourly_cost,
        )

    finishes: dict[str, float] = {}
    transferred = 0
    for artifact in graph.topological_order():
        if artifact.artifact_id not in tier_by_artifact:
            continue
        dependency_finish = max(
            (finishes[item] for item in artifact.dependencies if item in finishes), default=0.0
        )
        tier_id = tier_by_artifact[artifact.artifact_id]
        if tier_id is None:
            duration = 0.0
        else:
            transferred += artifact.size_bytes
            duration = _sample_duration_ms(
                artifact_id=artifact.artifact_id,
                artifact_bytes=artifact.size_bytes,
                tier=tiers[tier_id],
                measurement_map=measurement_map,
                cold_uncached=strategy == WarmPathStrategy.COLD_UNCACHED
                or (strategy == WarmPathStrategy.WARM_REPLICA and eviction),
                rng=rng,
            )
        finishes[artifact.artifact_id] = dependency_finish + duration
    readiness = max((finishes[item] for item in tier_by_artifact), default=0.0)
    return WarmPathTrial(
        strategy=strategy,
        scenario=scenario,
        seed=seed,
        trial_index=trial_index,
        ready_time_ms=readiness,
        restore_failed=False,
        eviction_occurred=eviction,
        transferred_bytes=transferred,
        storage_bytes=storage_bytes,
        hourly_cost_usd=hourly_cost,
    )


def _bootstrap_rate_interval(
    values: tuple[float, ...], *, seed: int, rounds: int
) -> tuple[float, float]:
    if len(values) < 3:
        raise ValueError("rate interval requires at least three seed aggregates")
    rng = random.Random(seed)
    estimates = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(rounds))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def _summary(
    strategy: WarmPathStrategy,
    *,
    trials: tuple[WarmPathTrial, ...],
    config: WarmPathEvaluationConfig,
) -> WarmPathStrategySummary:
    nominal = tuple(item for item in trials if item.scenario == EvaluationScenario.NOMINAL)
    eviction = tuple(
        item for item in trials if item.scenario == EvaluationScenario.EVICTION_PRESSURE
    )
    successful = tuple(
        cast(float, item.ready_time_ms) for item in nominal if not item.restore_failed
    )
    successful_eviction = tuple(
        cast(float, item.ready_time_ms) for item in eviction if not item.restore_failed
    )
    if not successful or not successful_eviction:
        raise RuntimeError(f"all {strategy} trials failed")
    seed_p95 = tuple(
        percentile(
            tuple(
                cast(float, item.ready_time_ms)
                for item in nominal
                if item.seed == seed and not item.restore_failed
            ),
            0.95,
        )
        for seed in config.seeds
    )
    p95_low, p95_high = bootstrap_median_interval(
        seed_p95,
        seed=sum(config.seeds) + list(WarmPathStrategy).index(strategy),
        rounds=config.bootstrap_rounds,
    )
    failures_by_seed = tuple(
        statistics.mean(float(item.restore_failed) for item in nominal if item.seed == seed)
        for seed in config.seeds
    )
    failure_rate = statistics.mean(float(item.restore_failed) for item in nominal)
    failure_low, failure_high = _bootstrap_rate_interval(
        failures_by_seed,
        seed=sum(config.seeds) + 101 + list(WarmPathStrategy).index(strategy),
        rounds=config.bootstrap_rounds,
    )
    nominal_p95 = percentile(successful, 0.95)
    eviction_p95 = percentile(successful_eviction, 0.95)
    penalty = eviction_p95 - nominal_p95
    relative = None if nominal_p95 == 0.0 else 100.0 * penalty / nominal_p95
    return WarmPathStrategySummary(
        strategy=strategy,
        measurement_basis=_MEASUREMENT_BASIS[strategy],
        seed_count=len(config.seeds),
        trials_per_scenario=len(nominal),
        successful_nominal_trials=len(successful),
        p50_ready_time_ms=percentile(successful, 0.5),
        p95_ready_time_ms=nominal_p95,
        seed_median_p95_ready_time_ms=statistics.median(seed_p95),
        p95_confidence_interval_low_ms=p95_low,
        p95_confidence_interval_high_ms=p95_high,
        failure_rate=failure_rate,
        failure_rate_confidence_interval_low=failure_low,
        failure_rate_confidence_interval_high=failure_high,
        hourly_cost_usd=statistics.median(item.hourly_cost_usd for item in nominal),
        storage_bytes=int(statistics.median(item.storage_bytes for item in nominal)),
        median_transfer_bytes=statistics.median(item.transferred_bytes for item in nominal),
        eviction_p95_ready_time_ms=eviction_p95,
        eviction_absolute_penalty_ms=penalty,
        eviction_relative_penalty_percent=relative,
    )


def _all_trials(
    *,
    graph: ArtifactGraph,
    profile: StartupProfile,
    plan: WarmPathPlan,
    config: WarmPathEvaluationConfig,
) -> tuple[WarmPathTrial, ...]:
    measurements = _measurements(profile)
    records: list[WarmPathTrial] = []
    for strategy_index, strategy in enumerate(WarmPathStrategy):
        for scenario_index, scenario in enumerate(EvaluationScenario):
            for seed in config.seeds:
                rng = random.Random(seed + strategy_index * 1_000_003 + scenario_index * 10_000_019)
                records.extend(
                    _trial(
                        strategy=strategy,
                        scenario=scenario,
                        seed=seed,
                        trial_index=trial_index,
                        graph=graph,
                        profile=profile,
                        plan=plan,
                        config=config,
                        measurement_map=measurements,
                        rng=rng,
                    )
                    for trial_index in range(config.trials_per_seed)
                )
    return tuple(records)


def _write_jsonl(path: Path, values: Iterable[_EvaluationModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{item.model_dump_json()}\n" for item in values)
    path.write_text(payload, encoding="utf-8")


def _bar_plot(path: Path, summaries: tuple[WarmPathStrategySummary, ...]) -> None:
    width = 920
    height = 420
    margin = 90
    chart_height = 260
    maximum = max(item.p95_ready_time_ms for item in summaries) or 1.0
    bar_width = 90
    gap = 45
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<text x="32" y="38" fill="#f3f4f6" font-family="sans-serif" font-size="22">'
        "WarmPath H6 p95 readiness (lower is better)</text>",
        f'<line x1="{margin}" y1="{margin + chart_height}" x2="{width - 30}" '
        f'y2="{margin + chart_height}" stroke="#94a3b8"/>',
    ]
    labels = {
        WarmPathStrategy.COLD_UNCACHED: "cold",
        WarmPathStrategy.LOCAL_DISK_CACHE: "disk",
        WarmPathStrategy.PAGE_CACHE: "page",
        WarmPathStrategy.PINNED_HOST_MEMORY_MODELED: "host proxy",
        WarmPathStrategy.WARMPATH: "WarmPath",
        WarmPathStrategy.WARM_REPLICA: "warm",
    }
    for index, item in enumerate(summaries):
        x = margin + index * (bar_width + gap)
        bar_height = chart_height * item.p95_ready_time_ms / maximum
        y = margin + chart_height - bar_height
        color = "#2dd4bf" if item.strategy == WarmPathStrategy.WARMPATH else "#60a5fa"
        elements.extend(
            (
                f'<rect x="{x}" y="{y:.3f}" width="{bar_width}" height="{bar_height:.3f}" '
                f'fill="{color}"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{max(75.0, y - 7):.3f}" '
                f'fill="#e5e7eb" text-anchor="middle" font-family="monospace" font-size="12">'
                f"{item.p95_ready_time_ms:.3f} ms</text>",
                f'<text x="{x + bar_width / 2:.1f}" y="{margin + chart_height + 24}" '
                f'fill="#cbd5e1" text-anchor="middle" font-family="sans-serif" font-size="12">'
                f"{labels[item.strategy]}</text>",
            )
        )
    elements.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def _report(path: Path, result: WarmPathEvaluationResult) -> None:
    by_strategy = {item.strategy: item for item in result.strategies}
    warm = by_strategy[WarmPathStrategy.WARMPATH]
    lines = [
        "# WarmPath H6 evaluation",
        "",
        "This CPU-only comparison resamples measured local startup stages across "
        f"{warm.seed_count} deterministic seeds. It evaluates "
        f"{warm.trials_per_scenario} trials per strategy and scenario. P95 intervals are "
        "bootstrap intervals over per-seed p95 values.",
        "",
        "| Strategy | p50 ready (ms) | p95 ready (ms) | 95% CI of seed p95 (ms) | "
        "Hourly cost (USD) | Failure rate | Eviction p95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result.strategies:
        lines.append(
            f"| {item.strategy.value} | {item.p50_ready_time_ms:.4f} | "
            f"{item.p95_ready_time_ms:.4f} | [{item.p95_confidence_interval_low_ms:.4f}, "
            f"{item.p95_confidence_interval_high_ms:.4f}] | {item.hourly_cost_usd:.6f} | "
            f"{100.0 * item.failure_rate:.2f}% | {item.eviction_p95_ready_time_ms:.4f} |"
        )
    lines.extend(
        (
            "",
            "## Findings",
            "",
            f"- WarmPath changed p95 readiness by "
            f"{result.warmpath_vs_local_disk_p95_improvement_percent:+.2f}% relative to the "
            "local-disk baseline (positive means faster).",
            f"- The best non-warm strategy was `{result.best_non_warm_strategy.value}`; "
            f"WarmPath changed p95 by {result.warmpath_vs_best_non_warm_p95_improvement_percent:+.2f}% "
            "relative to it.",
            f"- Under the configured eviction pressure, WarmPath p95 changed by "
            f"{warm.eviction_absolute_penalty_ms:+.4f} ms.",
            "",
            "## Measurement basis and limitations",
            "",
        )
    )
    lines.extend(
        f"- `{item.strategy.value}`: {item.measurement_basis}." for item in result.strategies
    )
    lines.extend(("", *(f"- {item}" for item in result.limitations)))
    lines.extend(
        (
            "",
            "![H6 p95 readiness](warmpath-evaluation-h6.svg)",
            "",
            "Every table value is loaded from `artifacts/warmpath/evaluation/result.json`; "
            "individual trial outcomes are preserved in "
            "`artifacts/warmpath/evaluation/raw/trials.jsonl`.",
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_warmpath_evaluation(
    *,
    config_path: Path,
    output_directory: Path,
    report_path: Path,
    repository_root: Path,
    reset: bool = False,
) -> WarmPathEvaluationResult:
    """Run H6 from measured profile evidence and preserve every generated trial."""

    if reset and output_directory.exists():
        resolved = output_directory.resolve()
        if resolved in {Path("/").resolve(), Path.home().resolve(), repository_root.resolve()}:
            raise ValueError("refusing to reset a broad directory")
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    config = load_evaluation_config(config_path)
    graph, profile, plan = _profile_inputs(config, repository_root=repository_root)
    trials = _all_trials(graph=graph, profile=profile, plan=plan, config=config)
    raw_path = output_directory / "raw" / "trials.jsonl"
    _write_jsonl(raw_path, trials)
    summaries = tuple(
        _summary(
            strategy,
            trials=tuple(item for item in trials if item.strategy == strategy),
            config=config,
        )
        for strategy in WarmPathStrategy
    )
    by_strategy = {item.strategy: item for item in summaries}
    non_warm = tuple(
        item
        for item in summaries
        if item.strategy not in {WarmPathStrategy.WARM_REPLICA, WarmPathStrategy.WARMPATH}
    )
    best = min(non_warm, key=lambda item: (item.p95_ready_time_ms, item.strategy.value))
    warmpath = by_strategy[WarmPathStrategy.WARMPATH]
    local = by_strategy[WarmPathStrategy.LOCAL_DISK_CACHE]

    def improvement(reference: float) -> float:
        return (
            0.0
            if reference == 0.0
            else 100.0 * (reference - warmpath.p95_ready_time_ms) / reference
        )

    result = WarmPathEvaluationResult(
        config_sha256=sha256_file(config_path),
        graph_sha256=sha256_file(repository_root / config.graph_path),
        profile_sha256=sha256_file(repository_root / config.profile_path),
        plan_sha256=sha256_file(repository_root / config.plan_path),
        raw_trials_sha256=sha256_file(raw_path),
        strategies=summaries,
        best_non_warm_strategy=best.strategy,
        warmpath_vs_local_disk_p95_improvement_percent=improvement(local.p95_ready_time_ms),
        warmpath_vs_best_non_warm_p95_improvement_percent=improvement(best.p95_ready_time_ms),
        all_timings_hardware_measured=False,
        limitations=(
            "The workload uses a small deterministic synthetic snapshot, not production model weights.",
            "Pinned host memory is a modeled proxy backed by measured ordinary host-memory copies.",
            "Cold-cache behavior is a conservative bandwidth-floor model because unprivileged portable cache eviction is unavailable.",
            "Failure and eviction probabilities are controlled sensitivity scenarios, not observed host failure rates.",
            "Warm-replica readiness and cost are modeled; no cloud or GPU resource was created.",
        ),
    )
    result_path = output_directory / "result.json"
    write_json(result_path, result.model_dump(mode="json"))
    environment_path = output_directory / "environment.json"
    write_json(environment_path, environment_manifest(include_packages=True))
    plot_path = report_path.with_name("warmpath-evaluation-h6.svg")
    _bar_plot(plot_path, summaries)
    _report(report_path, result)
    tracked = (result_path, raw_path, environment_path, plot_path, report_path, config_path)
    manifest = WarmPathEvaluationManifest(
        command=(
            "python -m sloforge.warmpath.evaluation "
            f"--config {config_path} --output {output_directory} --report {report_path} --reset"
        ),
        result_path=str(result_path),
        raw_trial_count=len(trials),
        artifacts=tuple(
            EvaluationArtifact(path=str(path), sha256=sha256_file(path)) for path in tracked
        ),
    )
    write_json(output_directory / "manifest.json", manifest.model_dump(mode="json"))
    reloaded = WarmPathEvaluationResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    if f"{reloaded.warmpath_vs_local_disk_p95_improvement_percent:+.2f}%" not in report:
        raise RuntimeError("WarmPath report is not derived from the machine-readable result")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("benchmarks/warmpath/h6-evaluation.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/warmpath/evaluation"))
    parser.add_argument("--report", type=Path, default=Path("reports/warmpath-evaluation.md"))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    result = run_warmpath_evaluation(
        config_path=args.config,
        output_directory=args.output,
        report_path=args.report,
        repository_root=args.repository_root.resolve(),
        reset=args.reset,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
