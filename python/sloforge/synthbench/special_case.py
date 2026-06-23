"""Static detection of task/seed/hidden-case special casing in submitted systems."""

from __future__ import annotations

import ast
from pathlib import Path

from .models import HiddenCase, SpecialCaseAudit, SpecialCaseFinding, TaskDescriptor

MAXIMUM_SOURCE_BYTES = 1024 * 1024
MAXIMUM_SOURCE_FILES = 256


def _sources(path: Path) -> tuple[Path, ...]:
    candidates: tuple[Path, ...]
    if path.is_file():
        candidates = (path,)
    elif path.is_dir():
        candidates = tuple(sorted(path.rglob("*.py")))
    else:
        raise FileNotFoundError(f"implementation source does not exist: {path}")
    if len(candidates) > MAXIMUM_SOURCE_FILES:
        raise ValueError("implementation exceeds bounded source-file count")
    return candidates


def audit_special_casing(
    implementation_path: Path,
    descriptor: TaskDescriptor,
    hidden_cases: tuple[HiddenCase, ...],
) -> SpecialCaseAudit:
    findings: list[SpecialCaseFinding] = []
    hidden_lengths = {len(case.request.prompt_tokens) for case in hidden_cases}
    compared_hidden_lengths: dict[int, set[int]] = {}
    for source_path in _sources(implementation_path):
        payload = source_path.read_bytes()
        if len(payload) > MAXIMUM_SOURCE_BYTES:
            raise ValueError(f"implementation source exceeds bounded size: {source_path}")
        tree = ast.parse(payload.decode("utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                category: str | None = None
                message: str | None = None
                if node.value == descriptor.task_id:
                    category = "task_identity"
                    message = "implementation embeds the held-out task identifier"
                elif node.value == descriptor.seed:
                    category = "seed_identity"
                    message = "implementation embeds the held-out generation seed"
                elif node.value == descriptor.hidden_commitment:
                    category = "hidden_commitment"
                    message = "implementation embeds the hidden-case commitment"
                if category is not None and message is not None:
                    findings.append(
                        SpecialCaseFinding(
                            finding_id=f"literal-{len(findings):04d}",
                            severity="reject",
                            category=category,  # type: ignore[arg-type]
                            message=message,
                            line=max(1, node.lineno),
                        )
                    )
            if isinstance(node, ast.Compare):
                values = {
                    child.value
                    for child in ast.walk(node)
                    if isinstance(child, ast.Constant)
                    and isinstance(child.value, int)
                    and not isinstance(child.value, bool)
                    and child.value in hidden_lengths
                    and child.value not in {0, 1}
                }
                if values:
                    compared_hidden_lengths.setdefault(node.lineno, set()).update(values)
    all_compared = (
        set().union(*compared_hidden_lengths.values()) if compared_hidden_lengths else set()
    )
    if len(all_compared) >= 2:
        for line, values in sorted(compared_hidden_lengths.items()):
            if values:
                findings.append(
                    SpecialCaseFinding(
                        finding_id=f"hidden-shape-{len(findings):04d}",
                        severity="reject",
                        category="hidden_shape",
                        message=f"implementation branches on hidden prompt lengths {sorted(values)}",
                        line=line,
                    )
                )
    return SpecialCaseAudit(
        passed=not any(finding.severity == "reject" for finding in findings),
        findings=tuple(findings),
    )
