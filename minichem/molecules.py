"""Verbatim SMILES handling.

The LLM never has to retype a SMILES: molecules live in a session registry, the LLM refers to
them by handle (M1, M2, ...), and code substitutes the exact strings. Tool results are the
only source of new SMILES, and a guard flags any SMILES in an answer that no tool produced.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import ToolMessage
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from minichem.tools import chem

HANDLE_RE = re.compile(r"\bM(\d+)\b")
_EDGE_CHARS = "\"'`*,;:.!?"
_STRUCTURAL = set("=#()[]@/\\0123456789")
MIN_TOKEN_LEN = 3


def handle(index: int) -> str:
    """Handle text for a zero-based registry index."""
    return f"M{index + 1}"


def describe(smiles: str) -> str:
    """Short label the supervisor can see instead of the SMILES: the molecular formula."""
    parsed = chem.parse(smiles)
    return CalcMolFormula(parsed) if parsed is not None else "unparsed"


def find(registry: list[str], smiles: str) -> int | None:
    """Index of `smiles` in the registry (matching on canonical form), or None."""
    target = chem.canonical(smiles)
    for i, known in enumerate(registry):
        if known == smiles or (target is not None and chem.canonical(known) == target):
            return i
    return None


def register(registry: list[str], new_smiles: list[str]) -> list[str]:
    """Return the registry extended with any molecules it does not already hold.

    Valid SMILES are stored in canonical form, so a handle always expands to the same string
    no matter how the user happened to write the molecule.
    """
    updated = list(registry)
    for smiles in new_smiles:
        if find(updated, smiles) is None:
            updated.append(chem.canonical(smiles) or smiles)
    return updated


def expand_handles(text: str, registry: list[str], *, quote: bool = False) -> str:
    """Replace handles that exist in the registry with their exact SMILES."""

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        if not 0 <= index < len(registry):
            return match.group(0)
        return f"`{registry[index]}`" if quote else registry[index]

    return HANDLE_RE.sub(replace, text)


_SMILES_CHARS = r"\w@+\-\[\]=#/\\%"


def _replace_smiles(text: str, smiles: str, replacement: str) -> str:
    pattern = rf"(?<![{_SMILES_CHARS}]){re.escape(smiles)}(?![{_SMILES_CHARS}])"
    return re.sub(pattern, replacement, text)


def mask_smiles(text: str, registry: list[str]) -> str:
    """Replace registered molecules in `text` by handles: what the supervisor is allowed to see.

    Matches both the registry's canonical strings and other spellings of the same molecule
    (e.g. the user typed 'OCC'). Longest first, so a short SMILES never matches inside a
    longer registered one.
    """
    spellings: dict[str, str] = {s: handle(i) for i, s in enumerate(registry)}
    for token in smiles_in_text(text):
        index = find(registry, token)
        if index is not None:
            spellings.setdefault(token, handle(index))
    for smiles in sorted(spellings, key=len, reverse=True):
        text = _replace_smiles(text, smiles, spellings[smiles])
    return text


def collect_smiles(payload: Any) -> list[str]:
    """All values stored under a `smiles` key in a tool result, in order, without duplicates."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            value = node.get("smiles")
            if isinstance(value, str) and value not in found:
                found.append(value)
            for key, child in node.items():
                if key != "smiles":
                    walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(payload)
    return found


MAX_VALUE_CHARS = 400


def _index_of(registry: list[str]) -> dict[str, int]:
    """Canonical SMILES -> position, built once so a payload is not re-parsed per string."""
    index: dict[str, int] = {}
    for i, known in enumerate(registry):
        index.setdefault(chem.canonical(known) or known, i)
    return index


def _hide(node: Any, index: dict[str, int]) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "smiles" and isinstance(value, str):
                position = index.get(chem.canonical(value) or value)
                out["id"] = handle(position) if position is not None else "unregistered"
            else:
                out[key] = _hide(value, index)
        return out
    if isinstance(node, list):
        return [_hide(item, index) for item in node]
    if isinstance(node, str) and " " not in node and len(node) <= MAX_VALUE_CHARS:
        position = index.get(chem.canonical(node) or "")
        return handle(position) if position is not None else node
    return node


def with_handles(payload: Any, registry: list[str]) -> Any:
    """Copy of a tool result with every registered molecule replaced by its handle.

    A `smiles` key becomes `id`. Other string values (`input`, `query`, `unresolved` items)
    are replaced when the whole value is a registered molecule, in any spelling.
    """
    return _hide(payload, _index_of(registry))


def payload_of(message: ToolMessage) -> dict | None:
    """Decode a ToolMessage (local tool: JSON string; MCP tool: list of text blocks)."""
    content = message.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    try:
        decoded = json.loads(content)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _clean(token: str) -> str:
    """Strip prose punctuation and wrapping parentheses around a candidate SMILES."""
    previous = None
    while token != previous:
        previous = token
        token = token.strip(_EDGE_CHARS)
        if token.endswith(")") and token.count(")") > token.count("("):
            token = token[:-1]
        if token.startswith("(") and token.count("(") > token.count(")"):
            token = token[1:]
        if token.startswith("(") and token.endswith(")") and Chem.MolFromSmiles(token) is None:
            token = token[1:-1]  # '(c1ccccc1)' in prose is a parenthesised SMILES
    return token


def _tokens(text: str) -> list[str]:
    return [_clean(raw) for raw in text.split()]


def _is_molecule(token: str) -> bool:
    return len(token) >= MIN_TOKEN_LEN and Chem.MolFromSmiles(token) is not None


def smiles_in_text(text: str) -> list[str]:
    """SMILES typed by a user. Accepts structural characters or all-caps atom strings (CCO)."""
    found = []
    for token in _tokens(text):
        structural = any(ch in _STRUCTURAL for ch in token)
        all_caps = token.isalpha() and token.isupper()
        if (structural or all_caps) and _is_molecule(token) and token not in found:
            found.append(token)
    return found


def unverified_smiles(answer: str, registry: list[str]) -> list[str]:
    """SMILES-looking tokens in `answer` that are not in the registry (possible retyping)."""
    suspicious = []
    for token in _tokens(answer):
        if not any(ch in _STRUCTURAL for ch in token) or not _is_molecule(token):
            continue
        if find(registry, token) is None and token not in suspicious:
            suspicious.append(token)
    return suspicious
