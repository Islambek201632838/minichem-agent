"""Reaction naming: a name is given only when the named template reproduces the product."""

import pytest

from minichem.tools.reaction_types import REACTION_TYPES, classify_step


@pytest.mark.parametrize(
    ("reactants", "product", "name"),
    [
        (["CC(=O)OC(C)=O", "O=C(O)c1ccccc1O"], "CC(=O)Oc1ccccc1C(=O)O", "Acylation of an alcohol"),
        (["CC(=O)O", "Nc1ccccc1"], "CC(=O)Nc1ccccc1", "Amide coupling"),
        (["CC(=O)Cl", "Nc1ccccc1"], "CC(=O)Nc1ccccc1", "Amide formation from an acid chloride"),
        (["CC(=O)O", "CCO"], "CC(=O)OCC", "Esterification"),
        (["CC(=O)OCC"], "CC(=O)O", "Ester hydrolysis"),
        (["Brc1ccccc1", "OB(O)c1ccccc1"], "c1ccc(-c2ccccc2)cc1", "Suzuki coupling"),
        (["CC(=O)CC"], "CC(O)CC", "Reduction of a carbonyl to an alcohol"),
        (["O=[N+]([O-])c1ccccc1"], "Nc1ccccc1", "Reduction of a nitro group"),
        (["CC(C)C(=O)c1ccc(F)cc1", "CCOC(=O)CC"], "c1ccccc1", None),
    ],
)
def test_named_only_when_the_template_reproduces_the_product(reactants, product, name):
    found = classify_step(reactants, product)
    if name is None:
        assert found is None
    else:
        assert found is not None
        assert found.name.startswith(name)


def test_wrong_product_or_bad_input_is_never_named():
    assert classify_step(["CC(=O)O", "CCO"], "CCCCCC") is None
    assert classify_step(["CC(=O)O", "bad smiles"], "CC(=O)OCC") is None
    assert classify_step(["CC(=O)O"], "not a molecule") is None


def test_every_template_parses_and_every_hint_is_class_level():
    forbidden = ("mol", " g", "°", "reflux", "equiv")
    for reaction_type in REACTION_TYPES:
        assert reaction_type.name
        assert not any(word in reaction_type.reagent_hint for word in forbidden)
