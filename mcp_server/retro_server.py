"""MCP server for AiZynthFinder: route summaries and route steps (reactions, no conditions).

AiZynthFinder needs rdkit<2024 and numpy<2, which conflict with this project's pins, so this
server runs with the Python of a separate virtualenv (settings: AIZYNTH_PYTHON) and the agent
talks to it over stdio, exactly like the RDKit server. It reads AIZYNTH_CONFIG from the
environment because the project's settings package is not installed in that environment.

Run:  AIZYNTH_CONFIG=/path/to/config.yml <aizynth-python> -m mcp_server.retro_server
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from rdkit import Chem

from minichem.tools.models import ItemError, ToolError
from minichem.tools.retrosynthesis import (
    MAX_ROUTES,
    AiZynthRetrosynthesis,
    RetroResponse,
    RetroResult,
    RoutesResponse,
)

CONFIG_ENV = "AIZYNTH_CONFIG"
MAX_RETRO_BATCH = 5  # each search takes seconds to a minute, so batches stay small

mcp = FastMCP("minichem-retrosynthesis")


@cache
def _finder() -> AiZynthRetrosynthesis:
    """Load the models on first use so the MCP handshake stays fast."""
    config = os.environ.get(CONFIG_ENV)
    if not config:
        raise RuntimeError(f"{CONFIG_ENV} is not set; point it at AiZynthFinder's config.yml.")
    return AiZynthRetrosynthesis(Path(config))


@mcp.tool()
def retrosynthesis_check(
    smiles_list: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=MAX_RETRO_BATCH,
            description="SMILES strings, copied exactly.",
        ),
    ],
) -> RetroResponse:
    """Run a retrosynthesis search and report whether routes to purchasable stock exist.

    Returns per molecule: solved, number of routes, steps in the shortest route and the top
    route score. It never returns reagents, conditions or step lists. Use only when asked
    about synthetic routes. Searches are slow, so send few molecules.
    """
    results: list[RetroResult] = []
    errors: list[ItemError] = []
    for smiles in smiles_list:
        if Chem.MolFromSmiles(smiles) is None:
            errors.append(ItemError(input=smiles, error="Invalid SMILES."))
        else:
            results.append(_finder().analyse(smiles))
    return RetroResponse(results=results, errors=errors)


@mcp.tool()
def retrosynthesis_routes(
    smiles: Annotated[str, Field(description="One SMILES string, copied exactly.")],
    max_routes: Annotated[int, Field(ge=1, le=MAX_ROUTES, description="Routes to return.")] = 2,
) -> RoutesResponse | ToolError:
    """Return the best retrosynthetic routes for ONE molecule as a list of disconnections.

    Each step gives the product, its reactants (with in_stock = purchasable), the template
    model's probability and a forward check that the reversed template reproduces the product.
    If no route reaches purchasable molecules (solved = false), the best partial routes are
    returned instead, with `unresolved` listing the molecules the search could not split or
    buy: those steps would have to be designed by a chemist. It predicts no temperatures,
    catalysts, solvents or amounts. Use when asked for routes, precursors or reactions, not
    just whether a route exists. Searches take up to two minutes for large molecules.
    """
    if Chem.MolFromSmiles(smiles) is None:
        return ToolError(error="Invalid SMILES.", hint="Pass a SMILES taken from a tool result.")
    return _finder().routes(smiles, max_routes)


if __name__ == "__main__":
    mcp.run(transport="stdio")
