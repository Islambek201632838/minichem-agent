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


def test_retro_server_batches_results_and_reports_invalid_smiles(monkeypatch):
    from mcp_server import retro_server

    monkeypatch.setattr(retro_server, "_finder", lambda: MockRetrosynthesis())
    response = retro_server.retrosynthesis_check([ASPIRIN, "bad"])
    assert [r.smiles for r in response.results] == [ASPIRIN]
    assert [e.input for e in response.errors] == ["bad"]


def test_retro_server_without_config_fails_with_a_clear_message(monkeypatch):
    from mcp_server import retro_server

    monkeypatch.delenv(retro_server.CONFIG_ENV, raising=False)
    retro_server._finder.cache_clear()
    with pytest.raises(RuntimeError, match="AIZYNTH_CONFIG"):
        retro_server._finder()


def test_assistant_requires_both_aizynth_paths(tmp_path):
    from minichem.assistant import _retro_server_config
    from minichem.config import Settings

    settings = Settings(
        _env_file=None,
        enable_retrosynthesis=True,
        aizynth_config=tmp_path / "c",
        aizynth_python=None,  # explicit: config.py loads .env into os.environ on import
    )
    with pytest.raises(RuntimeError, match="AIZYNTH_PYTHON"):
        _retro_server_config(settings)


def test_routes_tool_rejects_invalid_smiles_without_raising():
    from mcp_server import retro_server

    result = retro_server.retrosynthesis_routes("bad")
    assert result.error == "Invalid SMILES."
