from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCUMENTS = (
    "docs/fabric/ARCHITECTURE.md",
    "docs/fabric/PHYSICAL_EXECUTION_PLAN.md",
    "docs/fabric/TOPOLOGY_DISCOVERY.md",
    "docs/fabric/FABRIC_PROFILER.md",
    "docs/fabric/DIGITAL_TWIN.md",
    "docs/fabric/PHYSICAL_COMPILER.md",
    "docs/fabric/RUNTIME_ADAPTERS.md",
    "docs/autopsy/ARCHITECTURE.md",
    "docs/autopsy/EVIDENCE_MODEL.md",
    "docs/autopsy/TIME_ALIGNMENT.md",
    "docs/autopsy/DIAGNOSIS.md",
    "docs/autopsy/COUNTERFACTUAL_REPLAY.md",
    "docs/autopsy/MINIMIZATION.md",
    "docs/recovery/RECOVERY_PLANNER.md",
    "docs/recovery/STATE_MACHINE.md",
    "docs/recovery/SAFETY.md",
    "docs/forgeci/ARCHITECTURE.md",
    "docs/forgeci/STATISTICS.md",
    "docs/forgeci/BISECTION.md",
    "docs/warmpath/ARCHITECTURE.md",
    "docs/warmpath/ARTIFACT_GRAPH.md",
    "docs/warmpath/PLANNER.md",
    "docs/FABRIC_RELATED_WORK.md",
    "docs/FABRIC_SECURITY.md",
    "docs/FABRIC_REPRODUCIBILITY.md",
    "docs/FABRIC_LIMITATIONS.md",
    "docs/FABRIC_DEMO_SCRIPT.md",
    "docs/FABRIC_INTERVIEW_DEEP_DIVE.md",
    "docs/FABRIC_RESUME_BULLETS.md",
    "paper/fabric_extension/README.md",
    "paper/fabric_extension/SLOFORGE_FABRIC.md",
)

REQUIRED_ADRS = tuple(f"docs/adr/{number:04d}" for number in range(12, 23))
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def test_required_fabric_documentation_is_substantive() -> None:
    for relative in REQUIRED_DOCUMENTS:
        path = ROOT / relative
        assert path.is_file(), relative
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# "), relative
        minimum_lines = 4 if relative.endswith("fabric_extension/README.md") else 15
        assert len(content.splitlines()) >= minimum_lines, relative

    adr_names = [path.relative_to(ROOT).as_posix() for path in (ROOT / "docs/adr").glob("*.md")]
    for prefix in REQUIRED_ADRS:
        assert sum(name.startswith(prefix) for name in adr_names) == 1, prefix


def test_relative_markdown_links_resolve() -> None:
    for relative in REQUIRED_DOCUMENTS:
        path = ROOT / relative
        for target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
            clean = target.split("#", maxsplit=1)[0]
            if not clean or clean.startswith(("http://", "https://", "mailto:")):
                continue
            assert (path.parent / clean).resolve().exists(), f"{relative}: {target}"


def test_resume_metrics_have_artifact_source_comments() -> None:
    content = (ROOT / "docs/FABRIC_RESUME_BULLETS.md").read_text(encoding="utf-8")
    bullets = [line for line in content.splitlines() if line.startswith("- ")]
    comments = [line for line in content.splitlines() if line.startswith("<!-- source:")]
    assert len(bullets) == len(comments) == 5
    assert "explicitly labeled synthetic" in content


def test_recruiter_documents_match_current_flagship_artifact() -> None:
    manifest = json.loads(
        (ROOT / "artifacts/fabric-demo/manifest.json").read_text(encoding="utf-8")
    )
    healthy = manifest["healthy"]["p95_ttft_ms"]
    degraded = manifest["degraded"]["p95_ttft_ms"]
    expected_values = (f"{healthy:.3f}", f"{degraded:.3f}")
    current_documents = (
        ROOT / "docs/FABRIC_INTERVIEW_DEEP_DIVE.md",
        ROOT / "docs/FABRIC_RESUME_BULLETS.md",
        ROOT / "paper/fabric_extension/SLOFORGE_FABRIC.md",
    )
    for path in current_documents:
        content = path.read_text(encoding="utf-8")
        assert all(value in content for value in expected_values), path
        assert "zero simulator calls" not in content, path


def test_autopsy_architecture_matches_current_flagship_evidence() -> None:
    root = ROOT / "artifacts/fabric-demo"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    degraded = json.loads((root / "autopsy/degraded-run.json").read_text(encoding="utf-8"))
    content = (ROOT / "docs/autopsy/ARCHITECTURE.md").read_text(encoding="utf-8")
    assert f"{len(degraded['events']):,} canonical events" in content
    assert f"{manifest['diagnosis_confidence']:.3f}" in content
    assert f"`{manifest['diagnosis']}` first" in content


def test_public_evaluation_claims_match_current_result() -> None:
    result = json.loads(
        (ROOT / "artifacts/fabric/evaluation/result.json").read_text(encoding="utf-8")
    )
    methods = {item["method"]: item for item in result["method_summaries"]}
    twin = result["twin_summary"]
    expert = next(
        item
        for item in result["twin_group_summaries"]
        if item["dimension"] == "workload" and item["value"] == "expert_skewed"
    )
    common_values = (
        f"{methods['hierarchical_compiler']['p95_ttft_ms']['median']:.3f}",
        f"{twin['rank_correlation']:.3f}",
        f"{100.0 * twin['interval_coverage']:.2f}%",
        f"{100.0 * expert['interval_coverage']:.0f}%",
    )
    current_documents = (
        ROOT / "README.md",
        ROOT / "docs/FABRIC_INTERVIEW_DEEP_DIVE.md",
        ROOT / "docs/FABRIC_RESUME_BULLETS.md",
        ROOT / "docs/FABRIC_LIMITATIONS.md",
        ROOT / "paper/fabric_extension/SLOFORGE_FABRIC.md",
        ROOT / "FABRIC_FINAL_ADVERSARIAL_REVIEW.md",
    )
    for path in current_documents:
        content = path.read_text(encoding="utf-8")
        assert all(value in content for value in common_values), path
    relative_error = f"{100.0 * twin['median_relative_error']:.4f}%"
    for path in current_documents:
        if path.name != "FABRIC_RESUME_BULLETS.md":
            assert relative_error in path.read_text(encoding="utf-8"), path
