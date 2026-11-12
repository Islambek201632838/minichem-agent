"""Exact, LLM-free checks for the golden suite: tool-sequence matching and answer checks."""

from __future__ import annotations

import re
from typing import Any

from minichem import molecules as mol
from minichem.tools import chem
from minichem.tools.models import Constraints

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
REGEX_FLAGS = re.I | re.S


def collapse_repeats(tools: list[str]) -> list[str]:
    """Drop immediate repeats (a retry after an error is not a different step)."""
    return [t for i, t in enumerate(tools) if i == 0 or t != tools[i - 1]]


def is_subsequence(expected: list[str], observed: list[str]) -> bool:
    """True if `expected` appears in `observed` in order (extra calls in between are allowed)."""
    remaining = iter(observed)
    return all(tool in remaining for tool in expected)


def match_tools(observed: list[str], expected: dict[str, list[list[str]]]) -> str:
    """'complete', 'partial' or 'none' against the accepted sequences from the golden file.

    An empty accepted sequence means "no tool should be called" and only matches when none was.
    """
    seen = collapse_repeats(observed)
    for sequence in expected.get("complete", []):
        if (not sequence and not seen) or (sequence and is_subsequence(sequence, seen)):
            return "complete"
    for sequence in expected.get("partial", []):
        if (not sequence and not seen) or (sequence and is_subsequence(sequence, seen)):
            return "partial"
    return "none"


def _answer_smiles(answer: str) -> list[str]:
    return [t for t in mol.smiles_in_text(answer) if chem.parse(t) is not None]


def _canon_set(smiles: list[str]) -> set[str]:
    return {c for s in smiles if (c := chem.canonical(s))}


def check_smiles_present(answer: str, expected: list[str]) -> list[str]:
    found = _canon_set(_answer_smiles(answer))
    return [f"missing SMILES {s}" for s in expected if chem.canonical(s) not in found]


def check_numbers(answer: str, expected: list[dict[str, float]]) -> list[str]:
    values = [float(x) for x in NUMBER_RE.findall(answer)]
    return [
        f"number {e['value']} (+/-{e.get('tol', 0.05)}) not found"
        for e in expected
        if not any(abs(v - e["value"]) <= e.get("tol", 0.05) for v in values)
    ]


def check_regex(answer: str, patterns: list[str]) -> list[str]:
    return [f"pattern not matched: {p}" for p in patterns if not re.search(p, answer, REGEX_FLAGS)]


def check_regex_any(answer: str, patterns: list[str]) -> list[str]:
    if any(re.search(p, answer, REGEX_FLAGS) for p in patterns):
        return []
    return [f"none of the patterns matched: {patterns}"]


def check_regex_none(answer: str, patterns: list[str]) -> list[str]:
    return [
        f"forbidden pattern matched: {p}" for p in patterns if re.search(p, answer, REGEX_FLAGS)
    ]


def check_no_smiles(answer: str, _: Any = True) -> list[str]:
    return [f"unexpected SMILES: {s}" for s in _answer_smiles(answer)]


def _presented_smiles(answer: str) -> list[str]:
    """SMILES the answer presents as results: those in markdown table rows if there are any.

    Prose may legitimately mention molecules that were rejected ("X was filtered out"), so
    when the answer has a table only the table counts; otherwise every SMILES counts.
    """
    rows = "\n".join(line for line in answer.splitlines() if line.lstrip().startswith("|"))
    return _answer_smiles(rows) or _answer_smiles(answer)


def check_smiles_property(answer: str, spec: dict[str, Any]) -> list[str]:
    """Every presented molecule (except `exclude`) must satisfy the constraints.

    Properties are recomputed with RDKit, so this does not trust the model's numbers.
    """
    excluded = _canon_set(spec.get("exclude", []))
    constraints = Constraints(**spec["constraints"])
    molecules = [s for s in _presented_smiles(answer) if chem.canonical(s) not in excluded]
    unique = sorted(_canon_set(molecules))
    problems = []
    if len(unique) < spec.get("min_count", 1):
        problems.append(f"only {len(unique)} molecules in the answer, need {spec['min_count']}")
    for smiles in unique:
        props = chem.calculate_properties([smiles]).results[0]
        if not chem.satisfies(props, constraints):
            problems.append(f"{smiles} violates constraints {spec['constraints']}")
    return problems


CHECKS = {
    "smiles_present": check_smiles_present,
    "numbers": check_numbers,
    "regex": check_regex,
    "regex_any": check_regex_any,
    "regex_none": check_regex_none,
    "no_smiles": check_no_smiles,
    "smiles_property": check_smiles_property,
}


def run_checks(answer: str, checks: dict[str, Any] | None) -> list[str] | None:
    """Failures for every check in the golden entry; None if the entry has no exact check."""
    if not checks:
        return None
    unknown = set(checks) - set(CHECKS)
    if unknown:
        raise ValueError(f"Unknown check types: {sorted(unknown)}")
    return [problem for name, spec in checks.items() for problem in CHECKS[name](answer, spec)]


def classify(tool_match: str, answer_failures: list[str] | None, crashed: bool) -> str:
    """complete / partial / incorrect.

    A wrong answer is incorrect even when the tool sequence was right: the user gets a wrong result.
    """
    if crashed or tool_match == "none" or answer_failures:
        return "incorrect"
    return "complete" if tool_match == "complete" else "partial"
