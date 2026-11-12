"""MCP server for reaction precedents: similar reactions from US patents (text-mined).

Two backends, chosen by the environment variable PRECEDENTS_BACKEND:
- `ord`: the hosted Open Reaction Database API (ORD_API_URL, ORD_SEARCH_TIMEOUT_S). Needs only the
  project environment and the internet; the product SMILES of the step leaves the machine.
- `local` (default when unset): the PrecedentFinder dataset. Needs rxnmapper, reaction_utils,
  pandas, scipy and paretoset, which the project environment does not have
  (settings: PRECEDENTS_PYTHON), and the data (PRECEDENTS_DATA_DIR; 1.8 million reactions, about
  3.5 GB in memory, loaded on the first call).
The agent talks to the server over stdio, like the other servers. Settings arrive as environment
variables because the local backend's environment does not have the project's settings package.

Run:  PRECEDENTS_BACKEND=ord python -m mcp_server.precedent_server
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from minichem.tools.models import ToolError
from minichem.tools.precedents import (
    MAX_PRECEDENTS,
    PrecedentSearch,
    PrecedentSetupError,
    PrecedentsResponse,
)

DATA_DIR_ENV = "PRECEDENTS_DATA_DIR"
BACKEND_ENV = "PRECEDENTS_BACKEND"
ORD_URL_ENV = "ORD_API_URL"
ORD_TIMEOUT_ENV = "ORD_SEARCH_TIMEOUT_S"
DEFAULT_ORD_URL = "https://open-reaction-database.org/api"
DEFAULT_ORD_TIMEOUT_S = 45.0
MAX_REACTANTS = 5

mcp = FastMCP("minichem-precedents")


@cache
def _search() -> PrecedentSearch:
    """Build the backend on first use so the MCP handshake stays fast."""
    if os.environ.get(BACKEND_ENV) == "ord":
        from minichem.tools.ord_client import OrdClient
        from minichem.tools.ord_precedents import OrdPrecedents

        return OrdPrecedents(
            OrdClient(
                os.environ.get(ORD_URL_ENV, DEFAULT_ORD_URL),
                float(os.environ.get(ORD_TIMEOUT_ENV, DEFAULT_ORD_TIMEOUT_S)),
            )
        )
    from minichem.tools.precedent_search import PatentPrecedents

    data_dir = os.environ.get(DATA_DIR_ENV)
    if not data_dir:
        raise PrecedentSetupError(
            f"{DATA_DIR_ENV} is not set; point it at the PrecedentFinder data."
        )
    return PatentPrecedents(Path(data_dir))


@mcp.tool()
def find_reaction_precedents(
    product: Annotated[str, Field(description="SMILES of the product of ONE reaction step.")],
    reactants: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=MAX_REACTANTS,
            description="SMILES of the reactants of that step.",
        ),
    ],
    max_precedents: Annotated[
        int, Field(ge=1, le=MAX_PRECEDENTS, description="Precedents to return.")
    ] = 3,
) -> PrecedentsResponse | ToolError:
    """Find similar reactions published in US patents (1976-2016) for ONE reaction step.

    Returns, for each precedent, the patent number, the year, the text-mined yield, and the
    reagents and solvents exactly as the patent record lists them. The records are extracted
    from patent text automatically and are NOT verified; this tool gives no temperatures,
    times or amounts. The search can take up to a minute when the hosted database is busy.
    Use after a retrosynthesis route has been found, when asked what reagents or solvents were
    used for a step, or whether a step is known.
    """
    try:
        return _search().find(product, reactants, max_precedents)
    except PrecedentSetupError as error:
        return ToolError(error=str(error), hint="The precedent database is not set up.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
