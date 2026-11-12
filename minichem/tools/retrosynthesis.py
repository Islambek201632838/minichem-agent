"""Optional retrosynthesis, OFF by default.

The interface returns only counts and a score: no reagents, conditions or step lists.
`MockRetrosynthesis` is used in tests. `AiZynthRetrosynthesis` needs a local
AiZynthFinder install with downloaded models (see README); nothing is fetched here.
It is used only by `mcp_server/retro_server.py`, which runs in AiZynthFinder's own environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from minichem.tools.models import ItemError

DEFAULT_TIME_LIMIT_S = 60


class RetroResult(BaseModel):
    """Summary of a retrosynthesis search."""

    smiles: str
    solved: bool
    n_routes: int
    n_steps_shortest: int | None = None
    top_score: float | None = None


class RetroResponse(BaseModel):
    """Batch result: one summary per molecule that could be searched, errors for the rest."""

    results: list[RetroResult]
    errors: list[ItemError] = []


class Retrosynthesis(Protocol):
    """Anything that can summarise whether a molecule has a plausible route."""

    def analyse(self, smiles: str) -> RetroResult: ...


class MockRetrosynthesis:
    """Deterministic stand-in: molecules with at most `max_atoms` heavy atoms are 'solved'."""

    def __init__(self, max_atoms: int = 20) -> None:
        self.max_atoms = max_atoms

    def analyse(self, smiles: str) -> RetroResult:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        solved = mol.GetNumHeavyAtoms() <= self.max_atoms
        return RetroResult(
            smiles=smiles,
            solved=solved,
            n_routes=2 if solved else 0,
            n_steps_shortest=3 if solved else None,
            top_score=0.9 if solved else 0.1,
        )


class AiZynthRetrosynthesis:
    """Adapter over AiZynthFinder's documented Python interface.

    Needs aizynthfinder in the running interpreter, which cannot share this project's
    environment (it requires rdkit<2024 and numpy<2). Smoke-tested by hand only.
    The statistics keys are read with `.get` so a missing key degrades to a default.
    """

    def __init__(
        self,
        config_file: Path,
        stock: str = "zinc",
        policy: str = "uspto",
        time_limit_s: int = DEFAULT_TIME_LIMIT_S,
    ) -> None:
        from aizynthfinder.aizynthfinder import AiZynthFinder

        self._finder = AiZynthFinder(configfile=str(config_file))
        self._finder.config.search.time_limit = time_limit_s
        self._finder.stock.select(stock)
        self._finder.expansion_policy.select(policy)

    def analyse(self, smiles: str) -> RetroResult:
        self._finder.target_smiles = smiles
        self._finder.tree_search()
        self._finder.build_routes()
        stats = self._finder.extract_statistics()
        n_routes = int(stats.get("number_of_solved_routes", 0))
        return RetroResult(
            smiles=smiles,
            solved=bool(stats.get("is_solved", False)),
            n_routes=n_routes,
            n_steps_shortest=stats.get("number_of_steps") if n_routes else None,
            top_score=stats.get("top_score"),
        )
