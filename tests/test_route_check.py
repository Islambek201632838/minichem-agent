"""Route parsing and the forward check, using a real AiZynthFinder tree captured for aspirin."""

from minichem.tools.retrosynthesis import Route, RoutesResponse, steps_from_tree
from minichem.tools.route_check import forward_check
from tests.conftest import ASPIRIN

TEMPLATE = (
    "[C;D1;H3:2]-[C;H0;D3;+0:1](=[O;D1;H0:3])-[O;H0;D2;+0:4]-[c:5]>>"
    "C-C(=O)-O-[C;H0;D3;+0:1](-[C;D1;H3:2])=[O;D1;H0:3].[OH;D1;+0:4]-[c:5]"
)
ANHYDRIDE = "CC(=O)OC(C)=O"
SALICYLIC_ACID = "O=C(O)c1ccccc1O"
ASPIRIN_TREE = {
    "type": "mol",
    "smiles": ASPIRIN,
    "in_stock": False,
    "children": [
        {
            "type": "reaction",
            "metadata": {
                "library_occurence": 1196,
                "policy_probability": 0.7261999845504761,
                "template": TEMPLATE,
            },
            "children": [
                {"type": "mol", "smiles": ANHYDRIDE, "in_stock": True, "children": []},
                {"type": "mol", "smiles": SALICYLIC_ACID, "in_stock": True, "children": []},
            ],
        }
    ],
}


def test_forward_check_reproduces_aspirin_from_its_precursors():
    assert forward_check(TEMPLATE, [ANHYDRIDE, SALICYLIC_ACID], ASPIRIN)
    assert forward_check(TEMPLATE, [SALICYLIC_ACID, ANHYDRIDE], ASPIRIN)


def test_forward_check_rejects_wrong_reactants_and_bad_input():
    assert not forward_check(TEMPLATE, ["CCO", SALICYLIC_ACID], ASPIRIN)
    assert not forward_check(TEMPLATE, [ANHYDRIDE], ASPIRIN)
    assert not forward_check(TEMPLATE, [ANHYDRIDE, "bad"], ASPIRIN)
    assert not forward_check("no separator", [ANHYDRIDE, SALICYLIC_ACID], ASPIRIN)


def test_steps_come_from_the_tree_with_numbers_and_no_conditions():
    (step,) = steps_from_tree(ASPIRIN_TREE)
    assert step.product.smiles == ASPIRIN
    assert [r.smiles for r in step.reactants] == [ANHYDRIDE, SALICYLIC_ACID]
    assert all(r.in_stock for r in step.reactants)
    assert step.policy_probability == 0.726
    assert step.template_support == 1196
    assert step.forward_check
    assert set(step.model_dump()) == {
        "product",
        "reactants",
        "policy_probability",
        "template_support",
        "forward_check",
    }


def test_multi_step_trees_are_listed_target_first():
    deeper = {
        "type": "mol",
        "smiles": SALICYLIC_ACID,
        "in_stock": False,
        "children": [
            {
                "type": "reaction",
                "metadata": {"template": "", "policy_probability": 0.5},
                "children": [ASPIRIN_TREE],
            }
        ],
    }
    steps = steps_from_tree(deeper)
    assert [s.product.smiles for s in steps] == [SALICYLIC_ACID, ASPIRIN]
    assert steps[1].forward_check
    assert not steps[0].forward_check


def test_routes_response_shape():
    route = Route(score=0.998, all_in_stock=True, steps=steps_from_tree(ASPIRIN_TREE))
    response = RoutesResponse(smiles=ASPIRIN, solved=True, routes=[route])
    assert response.model_dump()["routes"][0]["steps"][0]["reactants"][0]["in_stock"]
