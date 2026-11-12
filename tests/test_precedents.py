"""Precedents: models, helpers, the MCP tool with a mock, and the settings (no data, no network)."""

from __future__ import annotations

import pytest

from minichem.config import Settings
from minichem.handles import RunHandles
from minichem.tools.precedents import (
    PROVENANCE,
    MockPrecedents,
    PrecedentSetupError,
    clean_mapped,
    parse_record,
    strip_maps,
)
from tests.conftest import ASPIRIN

SALICYLIC = "O=C(O)c1ccccc1O"
ANHYDRIDE = "CC(=O)OC(C)=O"
MAPPED = (
    "[CH3:1][C:2](=[O:3])Cl.[OH:4][c:5]1[cH:6][cH:7][cH:8][cH:9][c:10]1[C:11](=[O:12])[OH:13]"
    ">>[CH3:1][C:2](=[O:3])[O:4][c:5]1[cH:6][cH:7][cH:8][cH:9][c:10]1[C:11](=[O:12])[OH:13]"
)


def test_settings_default_off():
    settings = Settings(_env_file=None, enable_precedents=False)
    assert settings.enable_precedents is False


def test_strip_maps_gives_plain_canonical_smiles():
    assert strip_maps("[CH3:1][OH:2]") == "CO"
    assert strip_maps("not a smiles [C:1]") == "not a smiles [C]"


def test_clean_mapped_moves_unmapped_reactants_to_reagents_and_drops_extra_maps():
    mapped = "[CH3:1][OH:2].[Na+].[OH-]>>[CH3:1][OH:2]"
    reactants, reagents, product = clean_mapped(mapped).split(">")
    assert reactants == "[CH3:1][OH:2]"
    assert reagents == "[Na+].[OH-]"
    assert product == "[CH3:1][OH:2]"


def test_parse_record_strips_maps_and_dedupes_reagents():
    record = (
        "[CH3:1][OH:2].[CH3:3][C:4](=[O:5])Cl>CCO.CCO.O>[CH3:1][O:2][C:4]([CH3:3])=[O:5] |f:1.2|"
    )
    reactants, reagents, product = parse_record(record)
    assert reactants == ["CO", "CC(=O)Cl"]
    assert reagents == ["CCO", "O"]
    assert product == "COC(C)=O"


def test_mock_returns_provenance_and_no_conditions():
    response = MockPrecedents().find(ASPIRIN, [SALICYLIC, ANHYDRIDE], 3)
    assert response.provenance == PROVENANCE
    assert "not verified" in response.provenance
    fields = set(response.precedents[0].model_dump())
    assert not fields & {"temperature", "time", "amount", "solvent_volume"}
    assert response.precedents[0].reagents_and_solvents


def test_tool_returns_precedents_and_passes_arguments(monkeypatch):
    from mcp_server import precedent_server

    monkeypatch.setattr(precedent_server, "_search", lambda: MockPrecedents())
    response = precedent_server.find_reaction_precedents(ASPIRIN, [SALICYLIC, ANHYDRIDE], 1)
    assert response.precedents[0].product.smiles == ASPIRIN
    assert [m.smiles for m in response.precedents[0].reactants] == [SALICYLIC, ANHYDRIDE]


def test_tool_reports_missing_setup_as_a_value_not_an_exception(monkeypatch):
    from mcp_server import precedent_server

    monkeypatch.delenv(precedent_server.DATA_DIR_ENV, raising=False)
    precedent_server._search.cache_clear()
    result = precedent_server.find_reaction_precedents(ASPIRIN, [SALICYLIC])
    assert "PRECEDENTS_DATA_DIR" in result.error
    assert result.hint


def test_search_reports_incomplete_data_directory(tmp_path):
    from minichem.tools.precedent_search import PatentPrecedents

    with pytest.raises(PrecedentSetupError, match="metadata_cleaned.csv.gz"):
        PatentPrecedents(tmp_path)


def test_assistant_requires_both_precedent_settings(tmp_path):
    from minichem.assistant import _precedent_server_config

    settings = Settings(
        _env_file=None,
        enable_precedents=True,
        precedents_data_dir=tmp_path,
        precedents_python=None,  # explicit: config.py loads .env into os.environ on import
    )
    with pytest.raises(RuntimeError, match="PRECEDENTS_PYTHON"):
        _precedent_server_config(settings)


def test_assistant_precedent_server_config(tmp_path):
    from minichem.assistant import _precedent_server_config

    settings = Settings(
        _env_file=None,
        enable_precedents=True,
        precedents_data_dir=tmp_path / "data",
        precedents_python=tmp_path / "python",
    )
    config = _precedent_server_config(settings)
    assert config["args"] == ["-m", "mcp_server.precedent_server"]
    assert config["env"]["PRECEDENTS_DATA_DIR"] == str(tmp_path / "data")


def test_handles_resolve_product_and_reactants():
    handles = RunHandles(["CCO", ASPIRIN, SALICYLIC])
    resolved = handles.resolve({"product": "M2", "reactants": ["M3", "M1"], "max_precedents": 2})
    assert resolved == {"product": ASPIRIN, "reactants": [SALICYLIC, "CCO"], "max_precedents": 2}


def test_handles_show_patent_molecules_as_handles_and_register_new_ones():
    handles = RunHandles([ASPIRIN])
    response = MockPrecedents().find(ASPIRIN, [SALICYLIC], 1)
    text = handles.present(response.model_dump())
    assert ASPIRIN not in text
    assert "M1" in text
    assert SALICYLIC in handles.registry
    assert "CCO" in handles.registry


def test_prompts_allow_solvents_only_from_the_precedent_tool():
    from minichem.prompts import load_prompts

    prompts = load_prompts()
    assert "find_reaction_precedents" in prompts["common"]
    assert "not verified" in prompts["common"]
    assert "only source of reagents and solvents" in prompts["synthesis"]
    assert "no temperatures, times, amounts" in prompts["synthesis"].replace("\n  ", " ")
    assert "not verified" in prompts["supervisor"]
