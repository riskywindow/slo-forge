"""Deterministic dependency-version matching used by lineage invalidation."""

from __future__ import annotations

import re

_TERM = re.compile(r"^(<=|>=|<|>)?\s*([0-9]+(?:\.(?:[0-9]+|[xX*])){0,2})$")


def _numeric_version(value: str) -> tuple[int, int, int]:
    core = value.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    if not 1 <= len(parts) <= 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"dependency version is not numeric semantic version: {value}")
    values = [int(item) for item in parts]
    values.extend(0 for _ in range(3 - len(values)))
    return values[0], values[1], values[2]


def _match_term(version: tuple[int, int, int], term: str) -> bool:
    match = _TERM.fullmatch(term.strip())
    if match is None:
        raise ValueError(f"unsupported dependency version term: {term}")
    operator, expected_text = match.groups()
    expected_parts = expected_text.split(".")
    wildcard = next(
        (index for index, item in enumerate(expected_parts) if item.lower() in {"x", "*"}),
        None,
    )
    if wildcard is not None:
        prefix = tuple(int(item) for item in expected_parts[:wildcard])
        current_prefix = version[:wildcard]
        if operator == "<":
            return current_prefix < prefix
        if operator == "<=":
            return current_prefix <= prefix
        if operator == ">":
            return current_prefix > prefix
        if operator == ">=":
            return current_prefix >= prefix
        return current_prefix == prefix
    expected = _numeric_version(expected_text)
    if operator == "<":
        return version < expected
    if operator == "<=":
        return version <= expected
    if operator == ">":
        return version > expected
    if operator == ">=":
        return version >= expected
    return version == expected


def version_matches(version: str, version_range: str) -> bool:
    if version_range == "*":
        return True
    parsed = _numeric_version(version)
    return all(_match_term(parsed, term) for term in version_range.split(","))
