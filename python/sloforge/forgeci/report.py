"""Upstream issue report generation from ForgeCI evidence."""

from __future__ import annotations

from sloforge.forgeci.models import BisectResult, ComparisonRecord, MinimalReproducer


def render_upstream_issue(
    *,
    title: str,
    comparison: ComparisonRecord,
    bisection: BisectResult,
    reproducer: MinimalReproducer,
) -> str:
    """Render a factual Markdown issue body without unsupported attribution claims."""

    metrics = "\n".join(
        (
            f"- `{metric.metric}`: {metric.baseline_median:.6g} → "
            f"{metric.candidate_median:.6g} {metric.unit} "
            f"({metric.degradation_percent:+.2f}%, "
            f"CI [{metric.degradation_ci_low_percent:+.2f}%, "
            f"{metric.degradation_ci_high_percent:+.2f}%], "
            f"Cliff's δ {metric.cliffs_delta:+.3f}; {metric.classification.value})"
        )
        for metric in comparison.metrics
    )
    commands = "\n".join(f"    {command}" for command in reproducer.reproduction_commands)
    artifacts = "\n".join(f"- `{path}`" for path in reproducer.artifact_references)
    suspected = bisection.first_regressing_commit or "not isolated"
    caveats = "\n".join(f"- {item}" for item in bisection.caveats) or "- None recorded"
    return f"""# {title}

## Observed regression

ForgeCI classification: **{comparison.classification.value}**.

{metrics}

Warmups were excluded from the measured trials. Intervals use deterministic bootstrap
resampling with multiple-metric correction; practical significance and the measured
noise floor are applied in addition to statistical significance.

## Suspected range

- Known good: `{bisection.good_commit}`
- Known bad: `{bisection.bad_commit}`
- First likely regressing commit: `{suspected}`
- Bisection confidence: {bisection.confidence:.1%}

This identifies a change point, not a proven source-code cause.

## Minimal reproducer

- Model/workload: `{reproducer.benchmark.input.model}` / `{reproducer.benchmark.input.workload}`
- Shape: prompt={reproducer.benchmark.input.prompt_tokens}, output={reproducer.benchmark.input.output_tokens}
- Concurrency: {reproducer.benchmark.input.concurrency}
- Expected: {reproducer.expected_regression}
- Confidence interval: {reproducer.confidence_interval}

```console
{commands}
```

## Environment requirement

- Architecture: `{reproducer.hardware.architecture}`
- CPUs: at least {reproducer.hardware.minimum_cpu_cores}
- Memory: at least {reproducer.hardware.minimum_memory_gib:g} GiB
- GPUs: {reproducer.hardware.gpu_count}

## Evidence artifacts

{artifacts}

## Caveats

{caveats}
"""
