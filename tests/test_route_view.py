"""Layout helpers for drawing routes (no Streamlit session needed)."""

import copy

from minichem.tools.retrosynthesis import Route, RouteMolecule, RouteStep, route_from_tree
from tests.conftest import ASPIRIN
from tests.test_route_check import ANHYDRIDE, ASPIRIN_TREE, SALICYLIC_ACID
from ui.route_view import (
    UNKNOWN_LABEL,
    forward_steps,
    reaction_image,
    route_labels,
    starting_materials,
    step_title,
)

TWO_STEP = Route(
    score=0.9,
    all_in_stock=True,
    n_steps=2,
    steps=[
        RouteStep(
            product=RouteMolecule(smiles="CCOC(C)=O", in_stock=False),
            reactants=[
                RouteMolecule(smiles="CC(=O)O", in_stock=True),
                RouteMolecule(smiles="CCO", in_stock=True),
            ],
            forward_check=True,
        ),
        RouteStep(
            product=RouteMolecule(smiles="CC(=O)O", in_stock=True),
            reactants=[RouteMolecule(smiles="CC=O", in_stock=True)],
            forward_check=True,
        ),
    ],
)


UNRESOLVED_TREE = copy.deepcopy(ASPIRIN_TREE)
UNRESOLVED_TREE["children"][0]["children"][1]["in_stock"] = False


def _aspirin_route() -> Route:
    return route_from_tree(ASPIRIN_TREE, True, 0.99)


def test_forward_steps_put_precursor_steps_before_the_target_step():
    steps = forward_steps(TWO_STEP)
    assert [s.product.smiles for s in steps] == ["CC(=O)O", "CCOC(C)=O"]


def test_starting_materials_are_the_purchasable_reactants_without_duplicates():
    assert [m.smiles for m in starting_materials(_aspirin_route())] == [ANHYDRIDE, SALICYLIC_ACID]
    assert len(starting_materials(TWO_STEP)) == 3


def test_labels_are_roles_within_the_route_not_session_handles():
    label = route_labels(TWO_STEP, "CCOC(C)=O")
    assert label("OCC") == "Start 2"
    assert label("CCOC(C)=O") == "Target"
    assert label("CC(=O)O") == "Intermediate 1"
    assert label("CC=O") == "Start 1"
    assert label("c1ccccc1") == UNKNOWN_LABEL


def test_unresolved_molecules_get_their_own_label():
    partial = route_from_tree(UNRESOLVED_TREE, False, 0.5)
    label = route_labels(partial, ASPIRIN)
    assert label(SALICYLIC_ACID) == "Unresolved 1"
    assert label(ANHYDRIDE) == "Start 1"


def test_step_title_reads_reactants_then_product():
    label = route_labels(_aspirin_route(), ASPIRIN)
    (step,) = _aspirin_route().steps
    assert step_title(step, label) == "Start 1 + Start 2 -> Target"


def test_reaction_image_is_drawn_for_a_real_step():
    (step,) = _aspirin_route().steps
    image = reaction_image(step)
    assert image.width > 0
    assert image.height > 0


def test_routes_of_keeps_only_successful_route_results():
    from minichem.assistant import _routes_of
    from minichem.tracing import Tracer

    tracer = Tracer(None)
    ok = {"smiles": ASPIRIN, "solved": True, "routes": [_aspirin_route().model_dump()]}
    tracer.results = [
        ("retrosynthesis_routes", ok),
        ("retrosynthesis_routes", {"error": "Invalid SMILES.", "hint": "x"}),
        ("sa_score", {"results": []}),
    ]
    (response,) = _routes_of(tracer)
    assert response.solved
    assert len(response.routes[0].steps) == 1
