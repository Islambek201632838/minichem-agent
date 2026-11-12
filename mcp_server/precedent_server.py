"""MCP server for reaction precedents: similar reactions from US patents (text-mined).

It needs rxnmapper, reaction_utils, pandas, scipy and paretoset, which the project environment
does not have (settings: PRECEDENTS_PYTHON), and the PrecedentFinder dataset (settings:
PRECEDENTS_DATA_DIR). The agent talks to it over stdio, like the other servers. It reads
PRECEDENTS_DATA_DIR from the environment because the project's settings package is not
installed in that environment. The dataset (1.8 million reactions, about 3.5 GB in memory) is
loaded on the first call, not at startup.

Run:  PRECEDENTS_DATA_DIR=/path/to/data <precedents-python> -m mcp_server.precedent_server
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
MAX_REACTANTS = 5

mcp = FastMCP("minichem-precedents")


@cache
def _search() -> PrecedentSearch:
    """Load the dataset on first use so the MCP handshake stays fast."""
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
    from patent text automatically and are NOT verified; the data has no temperatures, times
    or amounts, and this tool gives none. Use after a retrosynthesis route has been found,
    when asked what reagents or solvents were used for a step, or whether a step is known.
    """
    try:
        return _search().find(product, reactants, max_precedents)
    except PrecedentSetupError as error:
        return ToolError(error=str(error), hint="The precedent database is not set up.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
