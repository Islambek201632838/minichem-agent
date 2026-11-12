"""MCP server exposing the RDKit tools (properties, SMILES validation, similarity).

Run over stdio:  python -m mcp_server.server
The functions are thin wrappers over `minichem.tools.chem`, so the chemistry lives in one place
and any MCP-capable client (another agent, another team) can reuse it.
"""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from minichem.tools import chem
from minichem.tools.models import (
    MAX_BATCH,
    PropertiesResponse,
    SimilarityResponse,
    ToolError,
    ValidationResponse,
)

mcp = FastMCP("minichem-rdkit")

Smiles = Annotated[str, Field(description="One SMILES string, copied exactly.")]
SmilesBatch = Annotated[
    list[str],
    Field(min_length=1, max_length=MAX_BATCH, description="SMILES strings, copied exactly."),
]


@mcp.tool()
def calculate_properties(smiles_list: SmilesBatch) -> PropertiesResponse:
    """Compute MW, logP, HBD, HBA, rotatable bonds, TPSA and QED for molecules.

    Use for any question about physicochemical properties. Invalid SMILES are reported
    per item in `errors`; valid ones are still returned.
    """
    return chem.calculate_properties(smiles_list)


@mcp.tool()
def validate_smiles(smiles_list: SmilesBatch) -> ValidationResponse:
    """Check whether SMILES strings are valid and return canonical SMILES and formula.

    Use before trusting a SMILES that did not come from a tool.
    """
    return chem.validate_smiles(smiles_list)


@mcp.tool()
def tanimoto_similarity(query: Smiles, candidates: SmilesBatch) -> SimilarityResponse | ToolError:
    """Tanimoto similarity (Morgan radius 2, 2048 bits) of each candidate to the query.

    Use when asked how similar molecules are, or to rank molecules against a reference.
    """
    return chem.tanimoto_similarity(query, candidates)


if __name__ == "__main__":
    mcp.run(transport="stdio")
