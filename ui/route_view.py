"""Draw retrosynthesis routes: the target, the starting materials, then one scheme per step.

The route data comes from `retrosynthesis_routes` (product <- reactants per step). Nothing here
predicts chemistry: it lays out what the tool returned. Molecules are labelled by their role in
the route (Target, Start, Intermediate, Unresolved), not by session handles, which mean nothing
to a reader. The tool gives no temperatures, solvents or amounts, so none are shown; the
reagent line is a class-level hint from `reaction_types.py`.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st
from PIL.Image import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions

from minichem.tools.retrosynthesis import Route, RouteMolecule, RoutesResponse, RouteStep

UNKNOWN_LABEL = "molecule"
LONG_ROUTE_STEPS = 4  # longer routes start with every step collapsed
STEP_IMAGE_SIZE = (260, 200)
GRID_IMAGE_SIZE = (220, 170)
LEGEND = (
    "Pictures are the molecules. **Target** is what you asked about, **Start** molecules are "
    "purchasable, **Intermediate** molecules are made in one step and used in the next, "
    "**Unresolved** molecules have no purchasable source and no reaction was found. Steps run "
    "from the starting materials to the target. Reaction names and reagent classes come from a "
    "small built-in table; solvents, temperatures and amounts are not predicted."
)


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def forward_steps(route: Route) -> list[RouteStep]:
    """Steps in the order a chemist would run them: precursors first, target last.

    The tool lists steps target-first, parents before their children, so reversing the list puts
    every step after the steps that make its reactants.
    """
    return list(reversed(route.steps))


def starting_materials(route: Route) -> list[RouteMolecule]:
    """Purchasable molecules the route begins from, in the order the steps use them."""
    seen: dict[str, RouteMolecule] = {}
    for step in forward_steps(route):
        for reactant in step.reactants:
            if reactant.in_stock:
                seen.setdefault(_canonical(reactant.smiles), reactant)
    return list(seen.values())


def route_labels(route: Route, target: str) -> Callable[[str], str]:
    """SMILES -> role label within one route.

    A molecule gets its first matching role: Target, Intermediate (made by a step),
    Unresolved, Start. Numbers count within each role.
    """
    labels = {_canonical(target): "Target"}
    counts = {"Intermediate": 0, "Unresolved": 0, "Start": 0}

    def add(smiles: str, role: str) -> None:
        key = _canonical(smiles)
        if key not in labels:
            counts[role] += 1
            labels[key] = f"{role} {counts[role]}"

    for step in forward_steps(route):
        add(step.product.smiles, "Intermediate")
    for smiles in route.unresolved:
        add(smiles, "Unresolved")
    for molecule in starting_materials(route):
        add(molecule.smiles, "Start")
    return lambda smiles: labels.get(_canonical(smiles), UNKNOWN_LABEL)


def step_title(step: RouteStep, label: Callable[[str], str]) -> str:
    """'Start 1 + Start 2 -> Intermediate 1' for one step."""
    reactants = " + ".join(label(r.smiles) for r in step.reactants)
    return f"{reactants} -> {label(step.product.smiles)}"


def reaction_image(step: RouteStep) -> Image:
    """Reactants -> product drawn as a reaction scheme."""
    smarts = ".".join(r.smiles for r in step.reactants) + ">>" + step.product.smiles
    reaction = rdChemReactions.ReactionFromSmarts(smarts, useSmiles=True)
    return Draw.ReactionToImage(reaction, subImgSize=STEP_IMAGE_SIZE)


def _grid(smiles: list[str], label: Callable[[str], str]) -> None:
    st.image(
        Draw.MolsToGridImage(
            [Chem.MolFromSmiles(s) for s in smiles],
            legends=[label(s) for s in smiles],
            molsPerRow=4,
            subImgSize=GRID_IMAGE_SIZE,
        )
    )


def _show_overview(route: Route, target: str, label: Callable[[str], str]) -> None:
    st.markdown("**Target**")
    _grid([target], label)
    starts = [m.smiles for m in starting_materials(route)]
    if starts:
        st.markdown("**Starting materials** (purchasable in the stock database)")
        _grid(starts, label)
    if route.unresolved:
        st.markdown(
            ":red[**Unresolved:**] not purchasable and no reaction was found to make them. "
            "A chemist would have to design these steps."
        )
        _grid(route.unresolved, label)


def _show_step(
    number: int,
    step: RouteStep,
    label: Callable[[str], str],
    unresolved: set[str],
    expanded: bool,
) -> None:
    name = step.reaction_name or "Reaction type not in the built-in table"
    open_ends = [label(r.smiles) for r in step.reactants if _canonical(r.smiles) in unresolved]
    flag = " (has unresolved reactant)" if open_ends else ""
    with st.expander(f"Step {number}: {name}: {step_title(step, label)}{flag}", expanded=expanded):
        scheme, facts = st.columns([3, 1])
        scheme.image(reaction_image(step))
        if step.reagent_hint:
            facts.markdown(f"Typical reagent class: {step.reagent_hint}")
        check = ":green[passed]" if step.forward_check else ":red[failed]"
        facts.markdown(
            f"Forward check: {check}  \n"
            f"Template confidence: {step.policy_probability}  \n"
            f"Template seen in {step.template_support} known reactions"
        )
        if open_ends:
            facts.markdown(f":red[Unresolved: {', '.join(open_ends)}]")


def show_routes(routes: list[dict]) -> None:
    """Render every route of every retrosynthesis call this turn (dicts from the API)."""
    for response in map(RoutesResponse.model_validate, routes):
        if not response.routes:
            st.info("The search found no reaction template for this molecule.")
        elif not response.solved:
            st.warning(
                "No complete route to purchasable molecules was found. Below are the best "
                "partial routes: what the search could build, and where it got stuck."
            )
        for number, route in enumerate(response.routes, 1):
            label = route_labels(route, response.smiles)
            kind = "" if route.all_in_stock else " (partial)"
            st.markdown(
                f"#### Route {number}{kind}: "
                f"{route.n_steps} step{'s' if route.n_steps != 1 else ''}, score {route.score}"
            )
            st.caption(LEGEND)
            _show_overview(route, response.smiles, label)
            open_ends = {_canonical(s) for s in route.unresolved}
            for index, step in enumerate(forward_steps(route), 1):
                _show_step(index, step, label, open_ends, route.n_steps <= LONG_ROUTE_STEPS)
