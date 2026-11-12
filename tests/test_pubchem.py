import json

import requests

from minichem.tools import chem
from minichem.tools.models import NameLookup, ToolError
from minichem.tools.pubchem import OFFLINE_COMPOUNDS, OfflinePubChem, PubChemClient
from tests.conftest import ASPIRIN

ASPIRIN_PAYLOAD = {
    "PropertyTable": {
        "Properties": [
            {
                "CID": 2244,
                "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "Title": "Aspirin",
                "IUPACName": "2-acetyloxybenzoic acid",
                "MolecularFormula": "C9H8O4",
            }
        ]
    }
}


class FakeResponse:
    def __init__(self, status: int = 200, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Stands in for requests.Session; records calls so tests can assert on network use."""

    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def client(session: FakeSession, cache_path) -> PubChemClient:
    return PubChemClient(cache_path, timeout_s=1, session=session)


def test_name_lookup_returns_canonical_smiles(cache_path):
    session = FakeSession(FakeResponse(payload=ASPIRIN_PAYLOAD))
    result = client(session, cache_path).name_to_smiles("aspirin")
    assert isinstance(result, NameLookup)
    assert result.smiles == chem.canonical(ASPIRIN)
    assert result.title == "Aspirin"
    assert "name/aspirin/property" in session.calls[0][1]


def test_second_lookup_is_served_from_cache_even_across_instances(cache_path):
    session = FakeSession(FakeResponse(payload=ASPIRIN_PAYLOAD))
    client(session, cache_path).name_to_smiles("Aspirin")
    offline = FakeSession(requests.ConnectionError("network down"))
    result = client(offline, cache_path).name_to_smiles("aspirin")
    assert isinstance(result, NameLookup)
    assert not offline.calls
    assert json.loads(cache_path.read_text())


def test_not_found_is_an_error_object_with_hint(cache_path):
    result = client(FakeSession(FakeResponse(404)), cache_path).name_to_smiles("zzz")
    assert isinstance(result, ToolError)
    assert "synonym" in result.hint


def test_timeout_and_connection_errors_do_not_raise(cache_path):
    for exc in (requests.Timeout(), requests.ConnectionError("down")):
        result = client(FakeSession(exc), cache_path).name_to_smiles("aspirin")
        assert isinstance(result, ToolError)


def test_server_error_and_malformed_payload(cache_path):
    assert isinstance(
        client(FakeSession(FakeResponse(503)), cache_path).name_to_smiles("a"), ToolError
    )
    bad = FakeSession(FakeResponse(200, {"unexpected": 1}))
    assert isinstance(client(bad, cache_path).name_to_smiles("a"), ToolError)


def test_failed_lookups_are_not_cached(cache_path):
    client(FakeSession(FakeResponse(404)), cache_path).name_to_smiles("zzz")
    assert not cache_path.exists()


def test_corrupt_cache_file_is_ignored(cache_path):
    cache_path.write_text("{not json")
    session = FakeSession(FakeResponse(payload=ASPIRIN_PAYLOAD))
    assert isinstance(client(session, cache_path).name_to_smiles("aspirin"), NameLookup)


def test_smiles_to_name_posts_the_smiles(cache_path):
    session = FakeSession(FakeResponse(payload=ASPIRIN_PAYLOAD))
    result = client(session, cache_path).smiles_to_name(ASPIRIN)
    assert result.title == "Aspirin"
    assert session.calls[0][0] == "POST"


def test_smiles_to_name_rejects_invalid_smiles_without_network(cache_path):
    session = FakeSession(FakeResponse(payload=ASPIRIN_PAYLOAD))
    assert isinstance(client(session, cache_path).smiles_to_name("bad"), ToolError)
    assert not session.calls


def test_offline_fixtures_are_valid_and_round_trip():
    backend = OfflinePubChem()
    for name, (smiles, _) in OFFLINE_COMPOUNDS.items():
        assert chem.parse(smiles) is not None, name
        assert backend.name_to_smiles(name).smiles == chem.canonical(smiles)
    assert backend.smiles_to_name(ASPIRIN).title == "Aspirin"
    assert isinstance(backend.name_to_smiles("unobtainium"), ToolError)
