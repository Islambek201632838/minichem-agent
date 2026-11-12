"""LangChain tool definitions for the agents.

Every tool has a Pydantic argument schema, a description that says when to use it,
and returns a plain dict. Failures are returned as `{"error": ..., "hint": ...}` so the
agent can read them and correct itself; they are never raised into the agent loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from minichem.tools import analogues, chem, synthesis
from minichem.tools.models import Constraints, SmilesList, ToolError
from minichem.tools.pubchem import ChemLookup


class NameInput(BaseModel):
    name: str = Field(description="Compound name or common synonym, e.g. 'ibuprofen'.")


class SmilesInput(BaseModel):
    smiles: str = Field(description="Handle of one molecule, e.g. 'M1'.")


class SmilesListInput(BaseModel):
    smiles_list: SmilesList


class AnalogueInput(BaseModel):
    smiles: str = Field(description="Handle of the query molecule, e.g. 'M1'.")
    n: int = Field(default=10, ge=1, le=30, description="How many analogues to return.")
    min_similarity: float = Field(
        default=analogues.DEFAULT_MIN_SIMILARITY,
        ge=0,
        le=1,
        description="Minimum Tanimoto similarity to the query.",
    )
    constraints: Constraints | None = Field(
        default=None, description="Property windows the analogues must satisfy."
    )


class FilterInput(BaseModel):
    smiles_list: SmilesList
    constraints: Constraints = Field(description="Property windows each molecule must satisfy.")


def _safe(fn: Callable[..., BaseModel]) -> Callable[..., dict[str, Any]]:
    """Turn a typed result into a dict and any unexpected exception into an error dict.

    This is the one deliberate catch-all: it is the boundary between code and the LLM,
    and the tracer records the error either way.
    """

    def wrapper(**kwargs: Any) -> dict[str, Any]:
        try:
            return fn(**kwargs).model_dump(exclude_none=True)
        except Exception as exc:  # noqa: BLE001
            error = ToolError(error=f"{type(exc).__name__}: {exc}", hint="Check the arguments.")
            return error.model_dump()

    return wrapper


def _tool(
    fn: Callable[..., BaseModel], name: str, description: str, schema: type[BaseModel]
) -> BaseTool:
    return StructuredTool.from_function(
        func=_safe(fn), name=name, description=description, args_schema=schema
    )


def utility_tools(lookup: ChemLookup) -> list[BaseTool]:
    """Name <-> SMILES lookup and SMILES canonicalisation."""
    return [
        _tool(
            lambda name: lookup.name_to_smiles(name),
            "name_to_smiles",
            "Look up the canonical SMILES of a compound by its name or synonym (PubChem). "
            "Use whenever the user names a molecule instead of giving a SMILES.",
            NameInput,
        ),
        _tool(
            lambda smiles: lookup.smiles_to_name(smiles),
            "smiles_to_name",
            "Find the common name and IUPAC name of a molecule from its SMILES (PubChem). "
            "Use when the user asks what a SMILES is called.",
            SmilesInput,
        ),
        _tool(
            lambda smiles_list: chem.validate_smiles(smiles_list),
            "canonicalize_smiles",
            "Validate SMILES strings and return their canonical form and molecular formula. "
            "Use to check that a SMILES is valid or to normalise it.",
            SmilesListInput,
        ),
    ]


def design_local_tools() -> list[BaseTool]:
    """Analogue generation and constraint filtering (the RDKit property tools come via MCP)."""
    return [
        _tool(
            lambda smiles, n, min_similarity, constraints: analogues.generate_analogues(
                smiles, n, min_similarity, constraints
            ),
            "generate_analogues",
            "Propose structural analogues of one molecule, ranked by Tanimoto similarity, "
            "optionally restricted by property constraints (MW, logP, HBD, ...). "
            "Use when the user wants analogues, variants or similar designs of a molecule.",
            AnalogueInput,
        ),
        _tool(
            lambda smiles_list, constraints: chem.filter_molecules(smiles_list, constraints),
            "filter_molecules",
            "Split a list of molecules into those that meet property constraints and those "
            "that do not. Use for requests like 'keep molecules with HBD <= 2'.",
            FilterInput,
        ),
    ]


def synthesis_tools() -> list[BaseTool]:
    """SA score and structural flags (retrosynthesis comes via its own MCP server, if enabled)."""
    return [
        _tool(
            lambda smiles_list: synthesis.sa_scores(smiles_list),
            "sa_score",
            "Compute the synthetic accessibility score (1 easy .. 10 hard) of molecules. "
            "Use when asked how easy or hard a molecule is to make.",
            SmilesListInput,
        ),
        _tool(
            lambda smiles_list: synthesis.structural_flags(smiles_list),
            "structural_flags",
            "Report ring systems, stereocentres and structural alerts that make a molecule "
            "harder to make or less stable. Use alongside sa_score for synthesis concerns.",
            SmilesListInput,
        ),
    ]
