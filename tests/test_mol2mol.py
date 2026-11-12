"""Mol2Mol analogue generation: selection logic, server wrapper and settings, without REINVENT."""

import sys
from contextlib import asynccontextmanager

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from rdkit import Chem

from minichem.assistant import _reinvent_server_config
from minichem.config import PROJECT_ROOT, Settings
from minichem.molecules import payload_of
from minichem.tools import chem
from minichem.tools.models import Constraints, ToolError
from minichem.tools.mol2mol import (
    MIN_SAMPLES,
    MockMol2Mol,
    RawSample,
    generate,
    samples_needed,
    select_candidates,
)
from tests.conftest import ASPIRIN, IBUPROFEN

# Hand-picked ibuprofen neighbours. Similarities are recomputed by the code under test.
FLURBIPROFEN = "CC(C(=O)O)c1ccc(-c2ccccc2)c(F)c1"
IBUPROFEN_AMIDE = "CC(C)Cc1ccc(C(C)C(N)=O)cc1"
IBUPROFEN_METHYL_ESTER = "CC(C)Cc1ccc(C(C)C(=O)OC)cc1"
ETHANOL = "CCO"


def raw(*pairs: tuple[str, float]) -> list[RawSample]:
    return [RawSample(smiles, nll) for smiles, nll in pairs]


def query_mol() -> Chem.Mol:
    mol = chem.parse(IBUPROFEN)
    assert mol is not None
    return mol


def test_select_drops_invalid_query_itself_and_low_similarity():
    samples = raw(
        ("not a smiles", 1.0),
        (IBUPROFEN, 1.0),
        (ETHANOL, 1.0),
        (IBUPROFEN_AMIDE, 2.0),
    )
    chosen = select_candidates(query_mol(), samples, 10, 0.4, Constraints())
    assert [m.smiles for m in chosen] == [chem.canonical(IBUPROFEN_AMIDE)]


def test_select_recomputes_similarity_with_the_project_fingerprint():
    chosen = select_candidates(query_mol(), raw((IBUPROFEN_AMIDE, 2.0)), 5, 0.0, Constraints())
    expected = chem.similarity_value(query_mol(), chem.parse(IBUPROFEN_AMIDE))
    assert chosen[0].tanimoto == pytest.approx(expected, abs=0.001)


def test_select_dedupes_spellings_and_keeps_the_likelier_one():
    other_spelling = "NC(=O)C(C)c1ccc(CC(C)C)cc1"
    chosen = select_candidates(
        query_mol(), raw((IBUPROFEN_AMIDE, 5.0), (other_spelling, 3.0)), 5, 0.4, Constraints()
    )
    assert len(chosen) == 1
    assert chosen[0].nll == 3.0


def test_select_ranks_by_similarity_then_smiles_and_caps_at_n():
    samples = raw((FLURBIPROFEN, 1.0), (IBUPROFEN_METHYL_ESTER, 1.0), (IBUPROFEN_AMIDE, 1.0))
    chosen = select_candidates(query_mol(), samples, 2, 0.0, Constraints())
    assert len(chosen) == 2
    assert chosen[0].tanimoto >= chosen[1].tanimoto
    assert chem.canonical(FLURBIPROFEN) not in {m.smiles for m in chosen}  # least similar


def test_select_applies_constraints_with_rdkit_values():
    samples = raw((IBUPROFEN_AMIDE, 1.0), (IBUPROFEN_METHYL_ESTER, 1.0))
    chosen = select_candidates(query_mol(), samples, 5, 0.0, Constraints(hbd_max=0))
    assert [m.smiles for m in chosen] == [chem.canonical(IBUPROFEN_METHYL_ESTER)]
    assert chosen[0].hbd == 0


def test_generate_passes_canonical_query_seed_and_sample_count_to_the_sampler():
    sampler = MockMol2Mol(raw((IBUPROFEN_AMIDE, 2.0)))
    result = generate(sampler, "OC(=O)C(C)c1ccc(CC(C)C)cc1", n=3, seed=7)
    assert sampler.calls == [(chem.canonical(IBUPROFEN), samples_needed(3), 7)]
    assert result.seed == 7
    assert result.returned == 1
    assert "Only 1 of 1 sampled" in result.note
    assert result.molecules[0].smiles == chem.canonical(IBUPROFEN_AMIDE)


def test_generate_reports_an_invalid_query_as_a_tool_error_without_sampling():
    sampler = MockMol2Mol([])
    result = generate(sampler, "bad")
    assert isinstance(result, ToolError)
    assert sampler.calls == []


def test_samples_needed_has_a_floor_and_grows_with_n():
    assert samples_needed(1) == MIN_SAMPLES
    assert samples_needed(30) > MIN_SAMPLES


def test_server_tool_uses_the_generator_and_never_raises(monkeypatch):
    from mcp_server import reinvent_server

    monkeypatch.setattr(
        reinvent_server, "_generator", lambda: MockMol2Mol(raw((IBUPROFEN_AMIDE, 2.0)))
    )
    ok = reinvent_server.mol2mol_analogues(IBUPROFEN, n=2, seed=1)
    assert ok.returned == 1

    class Broken:
        def sample(self, smiles, num_samples, seed):
            raise RuntimeError("device lost")

    monkeypatch.setattr(reinvent_server, "_generator", lambda: Broken())
    failed = reinvent_server.mol2mol_analogues(IBUPROFEN)
    assert isinstance(failed, ToolError)
    assert "device lost" in failed.error
    assert "generate_analogues" in failed.hint


def test_server_without_prior_dir_returns_an_error_value(monkeypatch):
    from mcp_server import reinvent_server

    monkeypatch.delenv(reinvent_server.PRIOR_DIR_ENV, raising=False)
    reinvent_server._generator.cache_clear()
    failed = reinvent_server.mol2mol_analogues(IBUPROFEN)
    assert isinstance(failed, ToolError)
    assert "REINVENT_PRIOR_DIR" in failed.error


@asynccontextmanager
async def reinvent_tools(prior_dir):
    client = MultiServerMCPClient(
        {
            "reinvent": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_server.reinvent_server"],
                "cwd": str(PROJECT_ROOT),
                "env": {"REINVENT_PRIOR_DIR": str(prior_dir), "PYTHONPATH": str(PROJECT_ROOT)},
            }
        }
    )
    async with client.session("reinvent") as session:
        yield {t.name: t for t in await load_mcp_tools(session)}


async def call(tool, **args):
    message = await tool.ainvoke({"type": "tool_call", "id": "1", "name": tool.name, "args": args})
    return payload_of(message)


async def test_real_stdio_server_lists_the_tool_and_reports_a_missing_prior(tmp_path):
    async with reinvent_tools(tmp_path) as tools:
        assert set(tools) == {"mol2mol_analogues"}
        missing = await call(tools["mol2mol_analogues"], smiles=ASPIRIN)
        assert "prior not found" in missing["error"]
        assert "generate_analogues" in missing["hint"]
        invalid = await call(tools["mol2mol_analogues"], smiles="bad")
        assert "not a valid SMILES" in invalid["error"]


def test_assistant_requires_both_reinvent_paths(tmp_path):
    settings = Settings(
        _env_file=None,
        enable_reinvent=True,
        reinvent_prior_dir=tmp_path,
        reinvent_python=None,  # explicit: config.py loads .env into os.environ on import
    )
    with pytest.raises(RuntimeError, match="REINVENT_PYTHON"):
        _reinvent_server_config(settings)


def test_reinvent_server_config_passes_paths_and_device(tmp_path):
    settings = Settings(
        _env_file=None,
        enable_reinvent=True,
        reinvent_python=tmp_path / "py",
        reinvent_prior_dir=tmp_path / "priors",
        reinvent_device="cpu",
    )
    config = _reinvent_server_config(settings)
    assert config["command"] == str(tmp_path / "py")
    assert config["args"] == ["-m", "mcp_server.reinvent_server"]
    assert config["env"]["REINVENT_PRIOR_DIR"] == str(tmp_path / "priors")
    assert config["env"]["REINVENT_DEVICE"] == "cpu"
