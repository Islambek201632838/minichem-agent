"""MCP server for REINVENT 4 Mol2Mol analogue generation.

REINVENT 4 pulls in torch, polars, pandas 3 and mmpdb, which do not belong in this project's
lockfile for an optional feature, so the server runs with the Python of a separate virtualenv
(settings: REINVENT_PYTHON) and the agent talks to it over stdio, like the retrosynthesis server.
It reads REINVENT_PRIOR_DIR and REINVENT_DEVICE from the environment because the project's
settings package is not installed in that environment.

Run:  REINVENT_PRIOR_DIR=/path/to/priors <reinvent-python> -m mcp_server.reinvent_server
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from minichem.tools.models import Constraints, ToolError
from minichem.tools.mol2mol import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_SEED,
    MAX_MOLECULES,
    GenerationResponse,
    Mol2MolGenerator,
    ReinventMol2Mol,
    generate,
)

PRIOR_DIR_ENV = "REINVENT_PRIOR_DIR"
DEVICE_ENV = "REINVENT_DEVICE"
FALLBACK_HINT = "Use the deterministic generate_analogues tool instead."

mcp = FastMCP("minichem-reinvent")


@cache
def _generator() -> Mol2MolGenerator:
    """Create the sampler on first use; the prior itself is loaded on its first sample."""
    prior_dir = os.environ.get(PRIOR_DIR_ENV)
    if not prior_dir:
        raise RuntimeError(f"{PRIOR_DIR_ENV} is not set; point it at the folder with the priors.")
    return ReinventMol2Mol(Path(prior_dir), os.environ.get(DEVICE_ENV, "mps"))


@mcp.tool()
def mol2mol_analogues(
    smiles: Annotated[str, Field(description="One SMILES string, copied exactly.")],
    n: Annotated[int, Field(ge=1, le=MAX_MOLECULES, description="How many to return.")] = 10,
    min_similarity: Annotated[
        float, Field(ge=0, le=1, description="Minimum Tanimoto similarity to the query.")
    ] = DEFAULT_MIN_SIMILARITY,
    seed: Annotated[
        int, Field(ge=0, description="Random seed; the same seed gives the same list.")
    ] = DEFAULT_SEED,
    constraints: Annotated[
        Constraints | None, Field(description="Property windows the molecules must satisfy.")
    ] = None,
) -> GenerationResponse | ToolError:
    """Propose analogues of ONE molecule with REINVENT 4 Mol2Mol, a trained generative model.

    It samples new molecules that look like the query (stochastic; the seed is returned and
    makes a run repeatable), then keeps those above `min_similarity` that meet `constraints`,
    ranked by Tanimoto similarity. It does not optimise activity or any property, so say so if
    asked. The first call loads the model and takes longer. On an error use generate_analogues.
    """
    try:
        return generate(_generator(), smiles, n, min_similarity, seed, constraints)
    except (ImportError, RuntimeError, ValueError, KeyError, OSError) as exc:
        return ToolError(error=f"REINVENT could not sample: {exc}", hint=FALLBACK_HINT)


if __name__ == "__main__":
    mcp.run(transport="stdio")
