"""Optional retrosynthesis, OFF by default.

`analyse` returns counts and a score; `routes` returns disconnections (product <- reactants).
Neither predicts conditions: no temperatures, catalysts, solvents or amounts.
`MockRetrosynthesis` is used in tests. `AiZynthRetrosynthesis` needs a local
AiZynthFinder install with downloaded models (see README); nothing is fetched here.
It is used only by `mcp_server/retro_server.py`, which runs in AiZynthFinder's own environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from minichem.tools.models import ItemError
from minichem.tools.route_check import forward_check

DEFAULT_TIME_LIMIT_S = 60
MAX_ROUTES = 3


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


class RouteMolecule(BaseModel):
    """A molecule in a route. `in_stock` is true for purchasable starting materials."""

    smiles: str
    in_stock: bool


class RouteStep(BaseModel):
    """One disconnection: `product` is made from `reactants`. No conditions are predicted.

    `forward_check` says whether the template, applied forward to the reactants, regenerates
    the product. `policy_probability` is the expansion model's confidence in the template and
    `template_support` is how many training reactions the template was extracted from.
    """

    product: RouteMolecule
    reactants: list[RouteMolecule]
    policy_probability: float | None = None
    template_support: int | None = None
    forward_check: bool


class Route(BaseModel):
    """A route from the target (first step) down to purchasable molecules."""

    score: float
    all_in_stock: bool
    steps: list[RouteStep]


class RoutesResponse(BaseModel):
    """The best routes for one molecule, empty if the search found none."""

    smiles: str
    solved: bool
    routes: list[Route]


def steps_from_tree(node: dict) -> list[RouteStep]:
    """Flatten an AiZynthFinder tree dict (`ReactionTree.to_dict()`) into steps, target first."""
    steps: list[RouteStep] = []
    for reaction in node.get("children", []):
        meta = reaction.get("metadata", {})
        children = reaction.get("children", [])
        reactants = [RouteMolecule(smiles=c["smiles"], in_stock=c["in_stock"]) for c in children]
        probability = meta.get("policy_probability")
        support = meta.get("library_occurence")
        steps.append(
            RouteStep(
                product=RouteMolecule(smiles=node["smiles"], in_stock=node["in_stock"]),
                reactants=reactants,
                policy_probability=None if probability is None else round(float(probability), 3),
                template_support=None if support is None else int(support),
                forward_check=forward_check(
                    meta.get("template", ""), [r.smiles for r in reactants], node["smiles"]
                ),
            )
        )
        for child in children:
            steps.extend(steps_from_tree(child))
    return steps


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

    def routes(self, smiles: str, max_routes: int = 2) -> RoutesResponse:
        """Best routes by AiZynthFinder's score, each as a list of disconnections."""
        self._finder.target_smiles = smiles
        self._finder.tree_search()
        self._finder.build_routes()
        collection = self._finder.routes
        found = [
            Route(
                score=round(float(score["state score"]), 3),
                all_in_stock=tree.is_solved,
                steps=steps_from_tree(tree.to_dict()),
            )
            for tree, score in zip(collection.reaction_trees, collection.scores, strict=True)
        ]
        solved = [r for r in found if r.all_in_stock][:max_routes]
        return RoutesResponse(smiles=smiles, solved=bool(solved), routes=solved)
