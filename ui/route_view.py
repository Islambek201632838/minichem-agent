"""Draw retrosynthesis routes: starting materials, then one reaction scheme per step.

The route data comes from `retrosynthesis_routes` (product <- reactants per step). Nothing here
predicts chemistry: it only lays out what the tool returned. The tool gives no temperatures,
catalysts, solvents or amounts, so none are shown.
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
START_IMAGE_SIZE = (220, 170)
NO_CONDITIONS_NOTE = (
    "Read from the starting materials down to the target. Steps are predicted by a reaction "
    "template model, not looked up: reagents, solvent, temperature, catalyst and amounts are "
    "not predicted."
)


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def make_labeller(molecules: list[dict]) -> Callable[[str], str]:
    """SMILES -> handle (M1, ...) for molecules known this turn, else a generic label."""
    by_smiles = {_canonical(m["smiles"]): m["id"] for m in molecules}
    return lambda smiles: by_smiles.get(_canonical(smiles), UNKNOWN_LABEL)


def forward_steps(route: Route) -> list[RouteStep]:
    """Steps in the order a chemist would run them: precursors first, target last.

    The tool lists steps target-first, parents before their children, so reversing the list puts
    every step after the steps that make its reactants.
    """
    return list(reversed(route.steps))


def starting_materials(route: Route) -> list[RouteMolecule]:
    """Purchasable molecules the route begins from, without duplicates."""
    seen: dict[str, RouteMolecule] = {}
    for step in route.steps:
        for reactant in step.reactants:
            if reactant.in_stock:
                seen.setdefault(_canonical(reactant.smiles), reactant)
    return list(seen.values())


def step_title(step: RouteStep, label: Callable[[str], str]) -> str:
    """'M3 + M4 -> M2' for one step."""
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
            subImgSize=START_IMAGE_SIZE,
        )
    )


def _show_start(route: Route, label: Callable[[str], str]) -> None:
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
    open_ends = [label(r.smiles) for r in step.reactants if _canonical(r.smiles) in unresolved]
    flag = " (has unresolved reactant)" if open_ends else ""
    with st.expander(f"Step {number}: {step_title(step, label)}{flag}", expanded=expanded):
        scheme, facts = st.columns([3, 1])
        scheme.image(reaction_image(step))
        check = ":green[passed]" if step.forward_check else ":red[failed]"
        facts.markdown(
            f"Forward check: {check}  \n"
            f"Template confidence: {step.policy_probability}  \n"
            f"Template seen in {step.template_support} known reactions"
        )
        bought = [label(r.smiles) for r in step.reactants if r.in_stock]
        if bought:
            facts.markdown(f"Purchasable: {', '.join(bought)}")
        if open_ends:
            facts.markdown(f":red[Unresolved: {', '.join(open_ends)}]")


def show_routes(routes: list[dict], molecules: list[dict]) -> None:
    """Render every route of every retrosynthesis call this turn (dicts from the API)."""
    label = make_labeller(molecules)
    for response in map(RoutesResponse.model_validate, routes):
        if not response.routes:
            st.info(f"The search found no reaction template for {label(response.smiles)}.")
        elif not response.solved:
            st.warning(
                f"No complete route to purchasable molecules was found for "
                f"{label(response.smiles)}. Below are the best partial routes: what the search "
                f"could build, and where it got stuck."
            )
        for number, route in enumerate(response.routes, 1):
            kind = "" if route.all_in_stock else " (partial)"
            st.markdown(
                f"#### Route {number} to {label(response.smiles)}{kind}: "
                f"{route.n_steps} step{'s' if route.n_steps != 1 else ''}, score {route.score}"
            )
            st.caption(NO_CONDITIONS_NOTE)
            _show_start(route, label)
            open_ends = {_canonical(s) for s in route.unresolved}
            for index, step in enumerate(forward_steps(route), 1):
                _show_step(index, step, label, open_ends, route.n_steps <= LONG_ROUTE_STEPS)
