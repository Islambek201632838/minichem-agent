import pytest

from minichem.tools import chem
from minichem.tools.models import Constraints, ToolError
from tests.conftest import ASPIRIN, ETHANOL, IBUPROFEN

# (smiles, MW, HBD, HBA, rotatable bonds). RDKit's HBA/HBD definitions (Lipinski-style).
KNOWN = [
    (ETHANOL, 46.07, 1, 1, 0),
    (ASPIRIN, 180.16, 1, 3, 2),
    (IBUPROFEN, 206.28, 1, 1, 4),
]


@pytest.mark.parametrize(("smiles", "mw", "hbd", "hba", "rotb"), KNOWN)
def test_known_molecule_properties(smiles, mw, hbd, hba, rotb):
    result = chem.calculate_properties([smiles])
    assert not result.errors
    props = result.results[0]
    assert props.mw == pytest.approx(mw, abs=0.01)
    assert (props.hbd, props.hba, props.rotatable_bonds) == (hbd, hba, rotb)
    assert 0 <= props.qed <= 1


def test_properties_are_reported_for_canonical_smiles():
    assert chem.calculate_properties([IBUPROFEN]).results[0].smiles == chem.canonical(IBUPROFEN)


def test_ethanol_logp_is_not_negative_zero():
    assert str(chem.calculate_properties([ETHANOL]).results[0].logp) == "0.0"


def test_bad_smiles_is_reported_per_item_and_good_ones_still_computed():
    result = chem.calculate_properties(["not_a_smiles", ETHANOL, ""])
    assert [p.smiles for p in result.results] == ["CCO"]
    assert {e.input for e in result.errors} == {"not_a_smiles", ""}


def test_validate_smiles_returns_canonical_and_formula():
    items = chem.validate_smiles(["OCC", "C1CC", ASPIRIN]).results
    assert (items[0].valid, items[0].smiles, items[0].formula) == (True, "CCO", "C2H6O")
    assert items[1].valid is False
    assert items[1].error
    assert items[2].formula == "C9H8O4"


def test_similarity_bounds_and_identity():
    result = chem.tanimoto_similarity(ETHANOL, [ETHANOL, "CCC", "c1ccccc1", IBUPROFEN])
    scores = {h.smiles: h.tanimoto for h in result.results}
    assert scores["CCO"] == 1.0
    assert all(0.0 <= s <= 1.0 for s in scores.values())
    assert scores["c1ccccc1"] == 0.0


def test_similarity_is_symmetric():
    ab = chem.tanimoto_similarity(ASPIRIN, [IBUPROFEN]).results[0].tanimoto
    ba = chem.tanimoto_similarity(IBUPROFEN, [ASPIRIN]).results[0].tanimoto
    assert ab == ba


def test_similarity_with_invalid_query_returns_error_object():
    result = chem.tanimoto_similarity("bad", [ETHANOL])
    assert isinstance(result, ToolError)
    assert result.hint


def test_similarity_reports_invalid_candidates():
    result = chem.tanimoto_similarity(ETHANOL, ["CCC", "bad"])
    assert len(result.results) == 1
    assert result.errors[0].input == "bad"


def test_filter_hbd_max():
    result = chem.filter_molecules([ETHANOL, "OCC(O)C(O)CO"], Constraints(hbd_max=1))
    assert [p.smiles for p in result.passed] == ["CCO"]
    assert result.failed[0].hbd == 4


def test_filter_with_no_constraints_passes_everything():
    result = chem.filter_molecules([ETHANOL, ASPIRIN], Constraints())
    assert len(result.passed) == 2
    assert not result.failed


def test_filter_mw_window_and_invalid_input():
    result = chem.filter_molecules([ETHANOL, ASPIRIN, "bad"], Constraints(mw_min=100, mw_max=200))
    assert [p.mw for p in result.passed] == [180.16]
    assert result.errors[0].input == "bad"
