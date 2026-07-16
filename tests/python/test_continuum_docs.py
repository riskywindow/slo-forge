from __future__ import annotations

import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
CONTINUUM_DOCS = REPOSITORY / "docs" / "continuum"

REQUIRED_DOCUMENTS = (
    "ARCHITECTURE.md",
    "LOGICAL_STATE.md",
    "PHYSICAL_STATE.md",
    "EXECUTION_STATE_CAPSULE.md",
    "RUNTIME_ADAPTER_SDK.md",
    "COMPATIBILITY.md",
    "STATE_TRANSFORMATION_IR.md",
    "CONVERSION_COMPILER.md",
    "CONTENT_STORE.md",
    "TRANSPORT.md",
    "DIRTY_TRACKING.md",
    "MIGRATION_PLANNER.md",
    "TRANSACTION_PROTOCOL.md",
    "TOKEN_COMMIT_PROTOCOL.md",
    "TRUST_MODEL.md",
    "MODEL_CHECKING.md",
    "FORK_AND_CLONE.md",
    "CROSS_MODEL_RULES.md",
    "FABRIC_INTEGRATION.md",
    "WARMPATH_INTEGRATION.md",
    "GENESIS_INTEGRATION.md",
    "SECURITY.md",
    "THREAT_MODEL.md",
    "REPRODUCIBILITY.md",
    "LIMITATIONS.md",
    "DEMO_SCRIPT.md",
    "INTERVIEW_DEEP_DIVE.md",
    "RESUME_BULLETS.md",
    "README_SECTION.md",
)

REQUIRED_ADRS = (
    "0001-logical-physical-separation.md",
    "0002-runtime-adapter-boundary.md",
    "0003-canonical-state-representation.md",
    "0004-direct-conversion.md",
    "0005-transaction-coordinator.md",
    "0006-token-commit-semantics.md",
    "0007-rollback-windows.md",
    "0008-dirty-tracking.md",
    "0009-content-addressing.md",
    "0010-cross-tenant-deduplication.md",
    "0011-exactness-classes.md",
    "0012-cross-model-compatibility.md",
    "0013-rust-python-boundary.md",
    "0014-model-checking-scope.md",
    "0015-transport-abstraction.md",
    "0016-optional-bifrost-integration.md",
)

PAPER_SECTIONS = (
    "Abstract",
    "Motivation",
    "State model",
    "Portable ABI",
    "Compatibility system",
    "Conversion compiler",
    "Migration planner",
    "Transaction protocol",
    "Runtime adapters",
    "Implementation",
    "Evaluation",
    "Fault tolerance",
    "Security",
    "Related work",
    "Limitations",
    "Future work",
)


def test_required_continuum_documents_are_substantive() -> None:
    for relative in REQUIRED_DOCUMENTS:
        path = CONTINUUM_DOCS / relative
        assert path.is_file(), f"missing Continuum document: {relative}"
        text = path.read_text(encoding="utf-8")
        expected_heading = "## " if relative == "README_SECTION.md" else "# "
        assert text.startswith(expected_heading), f"{relative} needs the expected heading"
        assert len(text.split()) >= 45, f"{relative} is not substantive"

    related_work = REPOSITORY / "docs" / "CONTINUUM_RELATED_WORK.md"
    assert related_work.is_file()
    assert len(related_work.read_text(encoding="utf-8").split()) >= 200


def test_architecture_decisions_and_paper_cover_required_scope() -> None:
    adr_root = CONTINUUM_DOCS / "adrs"
    assert tuple(path.name for path in sorted(adr_root.glob("*.md"))) == REQUIRED_ADRS
    for relative in REQUIRED_ADRS:
        text = (adr_root / relative).read_text(encoding="utf-8")
        for heading in ("## Context", "## Decision", "## Consequences"):
            assert heading in text
        assert "Status: Accepted" in text

    paper = (REPOSITORY / "paper" / "continuum" / "REPORT.md").read_text(encoding="utf-8")
    for section in PAPER_SECTIONS:
        assert f"## {section}" in paper
    assert len(paper.split()) >= 1_500


def test_documentation_scopes_claims_and_resume_metrics() -> None:
    documents = {
        path.relative_to(REPOSITORY).as_posix(): path.read_text(encoding="utf-8")
        for path in (
            *CONTINUUM_DOCS.rglob("*.md"),
            REPOSITORY / "docs" / "CONTINUUM_RELATED_WORK.md",
            *(REPOSITORY / "paper" / "continuum").glob("*.md"),
        )
    }
    for relative, text in documents.items():
        lowered = text.lower()
        assert "todo" not in lowered, relative
        assert "fixme" not in lowered, relative
        assert "universal proof of correctness" not in lowered, relative
        assert "fully supports vllm-to-sglang" not in lowered, relative
        if not relative.endswith("RESUME_BULLETS.md"):
            assert re.search(r"\[[A-Z][A-Z0-9_]+\]", text) is None, relative

    resume = documents["docs/continuum/RESUME_BULLETS.md"]
    placeholders = set(re.findall(r"\[([A-Z][A-Z0-9_]+)\]", resume))
    assert not placeholders
    assert "direct CPU path won 0/5" in resume
    assert "bounded exploration, not a universal proof" in resume
    assert "artifacts/continuum/evaluation/evaluation-summary.json" in resume

    limitations = documents["docs/continuum/LIMITATIONS.md"]
    assert "No NVIDIA GPU" in limitations
    assert "exactly-once gateway acceptance" in limitations
    model_checking = documents["docs/continuum/MODEL_CHECKING.md"]
    assert "bounded exploration results, not a universal proof" in model_checking
