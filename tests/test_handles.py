from minichem.handles import RunHandles, is_handle
from tests.conftest import ASPIRIN, ETHANOL, IBUPROFEN


def test_is_handle_accepts_quoted_handles_only():
    assert all(is_handle(text) for text in ("M12", " `M3` ", "'M1'"))
    assert not any(is_handle(text) for text in ("M", "CCO", "M1 M2"))


def test_resolve_replaces_handles_in_molecule_arguments_only():
    handles = RunHandles([ASPIRIN, ETHANOL])
    args = {"smiles_list": ["M2", "`M1`", "CCC", "M9"], "name": "M1", "n": 5}
    assert handles.resolve(args) == {
        "smiles_list": [ETHANOL, ASPIRIN, "CCC", "M9"],  # raw and unknown values pass through
        "name": "M1",
        "n": 5,
    }
    assert handles.resolve({"smiles": "M1"}) == {"smiles": ASPIRIN}


def test_present_registers_new_molecules_and_shows_handles():
    handles = RunHandles([ASPIRIN])
    text = handles.present({"results": [{"smiles": ETHANOL, "mw": 46.07}, {"smiles": ASPIRIN}]})
    assert text == '{"results":[{"id":"M2","mw":46.07},{"id":"M1"}]}'
    assert handles.registry == [ASPIRIN, ETHANOL]
    assert handles.produced == [ETHANOL, ASPIRIN]  # touched this run, in order


def test_present_keeps_the_handle_of_a_molecule_given_in_another_spelling():
    handles = RunHandles([])
    first = handles.present({"smiles": "OCC"})
    again = handles.present({"smiles": IBUPROFEN})
    assert (first, again) == ('{"id":"M1"}', '{"id":"M2"}')
    assert handles.registry[0] == ETHANOL
