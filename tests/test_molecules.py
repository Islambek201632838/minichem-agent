from minichem import molecules as mol
from minichem.tools import chem
from tests.conftest import ASPIRIN, ETHANOL, IBUPROFEN


def test_register_dedupes_on_canonical_form_and_stores_canonical_smiles():
    registry = mol.register([], ["OCC", ETHANOL, IBUPROFEN])
    assert registry == [ETHANOL, chem.canonical(IBUPROFEN)]
    assert registry[1] != IBUPROFEN  # the user's spelling was not the canonical one


def test_expand_handles_substitutes_exact_smiles_and_ignores_unknown():
    registry = [ASPIRIN, ETHANOL]
    assert mol.expand_handles("Compare M1 with M2, not M7", registry) == (
        f"Compare {ASPIRIN} with {ETHANOL}, not M7"
    )
    assert mol.expand_handles("M2", registry, quote=True) == f"`{ETHANOL}`"


def test_collect_smiles_reads_nested_tool_results_in_order():
    payload = {"results": [{"smiles": "CCO", "mw": 46}, {"smiles": "CCC"}], "query": "ignored"}
    assert mol.collect_smiles(payload) == ["CCO", "CCC"]


def test_with_handles_replaces_smiles_by_ids():
    out = mol.with_handles({"results": [{"smiles": ETHANOL, "mw": 46.07}]}, [ASPIRIN, ETHANOL])
    assert out == {"results": [{"id": "M2", "mw": 46.07}]}


def test_user_smiles_are_detected_in_prose():
    text = f"How many HBD does {ASPIRIN} have? Compare with CCO, and MW of (c1ccccc1)."
    assert mol.smiles_in_text(text) == [ASPIRIN, "CCO", "c1ccccc1"]


def test_ordinary_words_and_numbers_are_not_smiles():
    assert mol.smiles_in_text("What is the MW between 180 and 230 for HBD <= 2?") == []


def test_unverified_smiles_flags_only_unknown_molecules():
    registry = [ASPIRIN]
    answer = f"Aspirin is `{ASPIRIN}` but {IBUPROFEN} was typed by the model. MW 180-230."
    assert mol.unverified_smiles(answer, registry) == [IBUPROFEN]


def test_describe_gives_formula():
    assert mol.describe(ASPIRIN) == "C9H8O4"


def test_mask_smiles_hides_registered_molecules_but_not_substrings():
    registry = ["CC(CCO)C", ETHANOL]
    masked = mol.mask_smiles("Compare CC(CCO)C with CCO, then CCOC and (CCO).", registry)
    assert masked == "Compare M1 with M2, then CCOC and (M2)."


def test_mask_smiles_also_hides_other_spellings_of_a_registered_molecule():
    registry = mol.register([], ["OCC"])
    assert mol.mask_smiles("Canonicalise OCC. Result: CCO.", registry) == (
        "Canonicalise M1. Result: M1."
    )


def test_parenthesised_backticked_smiles_are_found_by_detector_and_guard():
    text = f"Ibuprofen (`{IBUPROFEN}`) is known; also see (`{ASPIRIN}`)."
    assert mol.smiles_in_text(text) == [IBUPROFEN, ASPIRIN]
    assert mol.unverified_smiles(text, [ASPIRIN]) == [IBUPROFEN]
