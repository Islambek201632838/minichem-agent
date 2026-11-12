import pytest

from minichem.tools.retrosynthesis import MockRetrosynthesis, Retrosynthesis
from tests.conftest import ASPIRIN


def test_mock_satisfies_the_interface_and_returns_counts_only():
    retro: Retrosynthesis = MockRetrosynthesis()
    result = retro.analyse(ASPIRIN)
    assert result.solved
    assert set(result.model_dump()) == {
        "smiles",
        "solved",
        "n_routes",
        "n_steps_shortest",
        "top_score",
    }


def test_mock_marks_large_molecules_unsolved():
    assert not MockRetrosynthesis(max_atoms=5).analyse(ASPIRIN).solved


def test_mock_rejects_invalid_smiles():
    with pytest.raises(ValueError, match="Invalid SMILES"):
        MockRetrosynthesis().analyse("bad")
