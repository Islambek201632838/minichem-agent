"""eval/complex.yaml: file shape, offline entities, and expected values recomputed with RDKit."""

from pathlib import Path

import pytest
from rdkit.Chem import Descriptors, rdMolDescriptors

from eval.checks import run_checks
from eval.run_eval import load_golden
from minichem.tools import chem, synthesis
from minichem.tools.pubchem import OFFLINE_COMPOUNDS

COMPLEX_PATH = Path(__file__).resolve().parents[1] / "eval" / "complex.yaml"

# Formula and average mass (g/mol) as PubChem reported them on 2026-11-12 (compound property
# service), kept here to prove that each offline SMILES is the molecule its name says.
PUBCHEM_REFERENCE = {
    "imatinib": ("C29H31N7O", 493.6),
    "nilotinib": ("C28H22F3N7O", 529.5),
    "dasatinib": ("C22H26ClN7O2S", 488.0),
    "osimertinib": ("C28H33N7O2", 499.6),
    "sildenafil": ("C22H30N6O4S", 474.6),
    "tofacitinib": ("C16H20N6O", 312.37),
    "ruxolitinib": ("C17H18N6", 306.4),
    "baricitinib": ("C16H17N7O2S", 371.4),
    "paclitaxel": ("C47H51NO14", 853.9),
    "strychnine": ("C21H22N2O2", 334.4),
}


def _smiles(name: str) -> str:
    return chem.canonical(OFFLINE_COMPOUNDS[name][0])


def _props(name: str):
    return chem.calculate_properties([_smiles(name)]).results[0]


def _sa(name: str) -> float:
    return synthesis.sa_scores([_smiles(name)]).results[0].sa_score


def _tanimoto(a: str, b: str) -> float:
    return chem.tanimoto_similarity(_smiles(a), [_smiles(b)]).results[0].tanimoto


@pytest.mark.parametrize("name", sorted(PUBCHEM_REFERENCE))
def test_offline_smiles_match_the_pubchem_formula_and_mass(name):
    formula, mass = PUBCHEM_REFERENCE[name]
    mol = chem.parse(OFFLINE_COMPOUNDS[name][0])
    assert rdMolDescriptors.CalcMolFormula(mol) == formula
    assert abs(Descriptors.MolWt(mol) - mass) < 0.06


def test_complex_file_is_well_formed_and_uses_offline_entities():
    entries = load_golden(COMPLEX_PATH)
    assert len({e["id"] for e in entries}) == len(entries) >= 10
    assert {e["category"] for e in entries} == {"complex"}
    assert any(len(e["turns"]) > 1 for e in entries)
    for entry in entries:
        assert entry["expected_tools"]["complete"], entry["id"]
        run_checks("probe", entry.get("checks"))  # raises on unknown check types


def _yaml_numbers(entry_id: str) -> list[tuple[float, float]]:
    entry = {e["id"]: e for e in load_golden(COMPLEX_PATH)}[entry_id]
    return [(c["value"], c["tol"]) for c in entry["checks"]["numbers"]]


def _assert_values(entry_id: str, computed: list[float]) -> None:
    yaml_values = _yaml_numbers(entry_id)
    assert len(yaml_values) == len(computed), entry_id
    for (value, tol), actual in zip(yaml_values, computed, strict=True):
        assert abs(value - actual) <= tol, (entry_id, value, actual)


def test_expected_numbers_equal_the_rdkit_values():
    _assert_values("c01", [_props("imatinib").mw, _props("imatinib").logp])
    _assert_values(
        "c02",
        [
            _props("osimertinib").tpsa,
            _props("sildenafil").tpsa,
            _props("osimertinib").qed,
            _props("sildenafil").qed,
        ],
    )
    _assert_values("c03", [_sa("paclitaxel"), _sa("dasatinib")])
    _assert_values(
        "c04",
        [
            _tanimoto("imatinib", "nilotinib"),
            _tanimoto("imatinib", "dasatinib"),
            _tanimoto("nilotinib", "dasatinib"),
            _props("dasatinib").qed,
        ],
    )
    _assert_values(
        "c05",
        [
            _props("tofacitinib").mw,
            _props("ruxolitinib").mw,
            _props("tofacitinib").tpsa,
            _props("ruxolitinib").tpsa,
            _sa("tofacitinib"),
            _sa("ruxolitinib"),
        ],
    )
    _assert_values("c06", [_sa("strychnine"), _sa("paclitaxel")])
    _assert_values(
        "c07", [_props("paclitaxel").mw, _props("paclitaxel").logp, _props("paclitaxel").hba]
    )
    _assert_values("c12", [_props("dasatinib").qed, _props("tofacitinib").qed])


def test_c05_removes_baricitinib_and_c14_passes_exactly_two_molecules():
    assert (
        _props("baricitinib").tpsa
        > 100
        >= max(_props("tofacitinib").tpsa, _props("ruxolitinib").tpsa)
    )
    assert f"{_sa('baricitinib'):.2f}" == "3.25"
    passing = [
        n
        for n in ("imatinib", "nilotinib", "dasatinib", "osimertinib", "sildenafil")
        if _props(n).mw <= 495 and _props(n).hbd <= 2
    ]
    assert passing == ["imatinib", "sildenafil"]


def test_c06_stereocentres_are_fully_specified_and_osimertinib_has_one_alert():
    flags = synthesis.structural_flags([_smiles("strychnine"), _smiles("paclitaxel")]).results
    assert [f.n_unspecified_stereocenters for f in flags] == [0, 0]
    assert flags[0].n_spiro_atoms == 1
    alerts = synthesis.structural_flags([_smiles("osimertinib")]).results[0].alerts
    assert alerts == ["Michael_acceptor_1"]


def test_answer_checks_catch_a_wrong_number_and_accept_the_right_one():
    entry = {e["id"]: e for e in load_golden(COMPLEX_PATH)}["c01"]
    assert run_checks("Imatinib: MW 493.62, logP 4.59.", entry["checks"]) == []
    assert run_checks("Imatinib: MW 493.62, logP 3.1.", entry["checks"])
