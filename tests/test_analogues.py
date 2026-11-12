from minichem.tools import chem
from minichem.tools.analogues import generate_analogues
from minichem.tools.models import Constraints, ToolError
from tests.conftest import ETHANOL, IBUPROFEN


def test_analogues_are_valid_unique_and_exclude_the_query():
    result = generate_analogues(IBUPROFEN, n=10)
    smiles = [a.smiles for a in result.analogues]
    assert result.returned == len(smiles) > 0
    assert len(set(smiles)) == len(smiles)
    assert chem.canonical(IBUPROFEN) not in smiles
    assert all(chem.parse(s) is not None for s in smiles)


def test_analogues_respect_similarity_floor_and_ordering():
    result = generate_analogues(IBUPROFEN, n=10, min_similarity=0.5)
    scores = [a.tanimoto for a in result.analogues]
    assert all(s >= 0.5 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_analogues_respect_constraints():
    constraints = Constraints(mw_min=180, mw_max=230, hbd_max=1)
    result = generate_analogues(IBUPROFEN, n=10, constraints=constraints)
    assert result.analogues
    assert all(180 <= a.mw <= 230 and a.hbd <= 1 for a in result.analogues)


def test_analogues_never_exceed_n():
    assert generate_analogues(IBUPROFEN, n=3).returned == 3


def test_analogues_are_deterministic():
    first = [a.smiles for a in generate_analogues(IBUPROFEN, n=8).analogues]
    second = [a.smiles for a in generate_analogues(IBUPROFEN, n=8).analogues]
    assert first == second


def test_impossible_constraints_give_empty_result_with_explanation():
    result = generate_analogues(IBUPROFEN, constraints=Constraints(mw_min=900))
    assert result.returned == 0
    assert "relax" in result.note


def test_small_molecule_with_high_similarity_floor_can_be_empty():
    assert generate_analogues(ETHANOL, min_similarity=0.9).returned == 0


def test_invalid_query_returns_error_object():
    assert isinstance(generate_analogues("bad"), ToolError)
