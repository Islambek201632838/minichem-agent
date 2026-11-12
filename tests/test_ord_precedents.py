"""ORD precedents: record parsing, safety (no conditions read), client retries, ranking.

No network: the client talks to a fake session, records are built with the ORD protobuf classes.
"""

from __future__ import annotations

import base64

import pytest
import requests

from minichem.assistant import _precedent_server_config
from minichem.config import Settings
from minichem.ord_proto import reaction_pb2
from minichem.tools.models import ToolError
from minichem.tools.ord_client import OrdApiError, OrdClient
from minichem.tools.ord_precedents import OrdPrecedents, parse_ord_record
from tests.conftest import ASPIRIN

YEARS = {"ds-1976": 1976, "ds-1990": 1990}
ACETYLATION = (
    "[CH3:1][C:2](=[O:3])Cl.[OH:4][c:5]1[cH:6][cH:7][cH:8][cH:9][c:10]1[C:11](=[O:12])[OH:13]"
    ".CCOCC>C(N(CC)CC)C>"
    "[CH3:1][C:2](=[O:3])[O:4][c:5]1[cH:6][cH:7][cH:8][cH:9][c:10]1[C:11](=[O:12])[OH:13]"
)
SECRET_PROCEDURE = "The mixture was heated to 80 C for 2 hours with 5.5 g of reagent"


def make_record(
    reaction_id: str = "ord-1",
    smiles: str = ACETYLATION,
    patent: str | None = "US03930836",
    yield_percent: float | None = 81.0,
    dataset: str = "ds-1976",
) -> dict:
    """An API record whose protobuf also carries conditions the code must never read."""
    reaction = reaction_pb2.Reaction(reaction_id=reaction_id)
    identifier = reaction.identifiers.add()
    identifier.type = reaction_pb2.ReactionIdentifier.REACTION_CXSMILES
    identifier.value = smiles
    if patent:
        reaction.provenance.patent = patent
    reaction.notes.procedure_details = SECRET_PROCEDURE
    reaction.conditions.temperature.setpoint.value = 80.0
    outcome = reaction.outcomes.add()
    outcome.reaction_time.value = 2.0
    measurement = outcome.products.add().measurements.add()
    if yield_percent is not None:
        measurement.type = reaction_pb2.ProductMeasurement.YIELD
        measurement.percentage.value = yield_percent
    binpb = base64.b64encode(reaction.SerializeToString()).decode()
    return {"reaction_id": reaction_id, "dataset_id": dataset, "binpb": binpb, "molblocks": {}}


class FakeResponse:
    def __init__(self, status: int = 200, body=None) -> None:
        self.status_code = status
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    """Plays back scripted responses (or exceptions) and records the calls."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(*responses) -> tuple[OrdClient, FakeSession]:
    session = FakeSession(*responses)
    return OrdClient(
        "https://ord.example/api/", 5.0, session=session, sleep=lambda _s: None
    ), session


def test_parse_reads_patent_year_yield_and_splits_reagents():
    record = parse_ord_record(make_record(), YEARS)
    assert record is not None
    assert (record.patent, record.year, record.yield_percent) == ("US03930836", 1976, 81.0)
    assert record.reactants == ["CC(=O)Cl", "O=C(O)c1ccccc1O"]
    assert sorted(record.reagents) == ["CCN(CC)CC", "CCOCC"]  # unmapped ether moved here
    assert record.product == ASPIRIN


def test_parsed_record_holds_no_conditions():
    record = parse_ord_record(make_record(), YEARS)
    text = repr(record)
    assert "80" not in text.replace("1976", "").replace("US03930836", "")
    assert "heated" not in text
    assert set(vars(record)) == {
        "reaction_id", "patent", "year", "yield_percent", "reactants", "reagents", "product"
    }  # fmt: skip


@pytest.mark.parametrize(
    "record",
    [
        make_record(patent=None),
        make_record(dataset="unknown-dataset"),
        make_record(smiles="CCO"),
        {"reaction_id": "x", "dataset_id": "ds-1976", "binpb": "!!not base64!!"},
        {
            "reaction_id": "x",
            "dataset_id": "ds-1976",
            "binpb": base64.b64encode(b"\xff\xff").decode(),
        },
    ],
)
def test_unusable_records_are_skipped(record):
    assert parse_ord_record(record, YEARS) is None


def test_record_without_yield_has_none():
    assert parse_ord_record(make_record(yield_percent=None), YEARS).yield_percent is None


def test_client_retries_a_busy_search_then_returns_ids():
    ord_client, session = client(FakeResponse(503), FakeResponse(200, {"reaction_ids": ["a"]}))
    assert ord_client.search_ids({"query": {}}) == ["a"]
    assert len(session.calls) == 2
    assert session.calls[0][1] == "https://ord.example/api/search"


def test_client_gives_up_after_the_busy_retries():
    ord_client, session = client(*[FakeResponse(503)] * 3)
    with pytest.raises(OrdApiError, match="busy"):
        ord_client.search_ids({})
    assert len(session.calls) == 3


def test_client_does_not_retry_a_server_side_timeout():
    ord_client, session = client(FakeResponse(504), FakeResponse(200, {"reaction_ids": []}))
    with pytest.raises(OrdApiError, match="timed out"):
        ord_client.search_ids({})
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (requests.Timeout(), "did not answer"),
        (requests.ConnectionError(), "could not be reached"),
        (FakeResponse(500), "HTTP 500"),
        (FakeResponse(200), "not JSON"),
    ],
)
def test_client_failures_are_api_errors(outcome, message):
    ord_client, _ = client(outcome)
    with pytest.raises(OrdApiError, match=message):
        ord_client.search_ids({})


def test_client_fetches_in_batches_and_caches_dataset_years():
    ids = [f"ord-{i}" for i in range(150)]
    ord_client, session = client(
        FakeResponse(200, [{"reaction_id": "a"}]),
        FakeResponse(200, [{"reaction_id": "b"}]),
        FakeResponse(200, [{"dataset_id": "d1", "name": "uspto-grants-1976"}, {"name": "HTE"}]),
    )
    assert [r["reaction_id"] for r in ord_client.reactions(ids)] == ["a", "b"]
    assert len(session.calls[0][2]["params"]["id"]) == 100
    assert ord_client.dataset_years() == {"d1": 1976}
    assert ord_client.dataset_years() == {"d1": 1976}
    assert len(session.calls) == 3


class FakeOrdClient:
    """Stands in for OrdClient: one id list per search call."""

    def __init__(self, searches, records, fail_search=None) -> None:
        self.searches = list(searches)
        self.records = records
        self.fail_search = fail_search
        self.search_queries: list[dict] = []

    def search_ids(self, query):
        self.search_queries.append(query)
        if self.fail_search is not None and len(self.search_queries) > self.fail_search:
            raise OrdApiError("busy")
        return self.searches.pop(0)

    def reactions(self, ids):
        return [r for r in self.records if r["reaction_id"] in ids]

    def dataset_years(self):
        return YEARS


SALICYLIC = "O=C(O)c1ccccc1O"
ANHYDRIDE = "CC(=O)OC(C)=O"
OTHER = "[CH3:1][CH2:2][OH:3].[CH3:4][C:5](=[O:6])Cl>>[CH3:1][CH2:2][O:3][C:5]([CH3:4])=[O:6]"


def finder(**kwargs) -> tuple[OrdPrecedents, FakeOrdClient]:
    records = [
        make_record("ord-weak", OTHER, yield_percent=95.0, dataset="ds-1990"),
        make_record("ord-good", ACETYLATION, yield_percent=70.0),
    ]
    fake = FakeOrdClient([["ord-weak", "ord-good"], ["ord-good"]], records, **kwargs)
    return OrdPrecedents(fake), fake


def test_find_ranks_by_similarity_and_labels_the_source():
    ord_finder, fake = finder()
    response = ord_finder.find(ASPIRIN, [SALICYLIC, ANHYDRIDE], 3)
    assert [p.reaction_id for p in response.precedents] == ["ord-good", "ord-weak"]
    best = response.precedents[0]
    assert (best.patent, best.year, best.yield_percent) == ("US03930836", 1976, 70.0)
    assert best.centre_similarity is None
    assert response.mapping_confidence is None
    assert "Open Reaction Database" in response.provenance
    assert "not verified" in response.provenance
    assert best.similarity > response.precedents[1].similarity
    assert len(fake.search_queries) == 2  # exact product, then similar products


def test_find_searches_patent_records_only():
    ord_finder, fake = finder()
    ord_finder.find(ASPIRIN, [SALICYLIC], 1)
    clauses = fake.search_queries[0]["query"]["where"]["clauses"]
    assert {"op": "not_null", "path": "provenance.patent"} in clauses


def test_find_output_has_no_conditions():
    ord_finder, _ = finder()
    dumped = ord_finder.find(ASPIRIN, [SALICYLIC], 3).model_dump_json()
    assert "heated" not in dumped
    assert not {"temperature", "time", "amount"} & set(dumped.lower().split('"'))


def test_find_limits_results_and_caches_the_search():
    ord_finder, fake = finder()
    assert len(ord_finder.find(ASPIRIN, [SALICYLIC], 1).precedents) == 1
    assert len(ord_finder.find(ASPIRIN, [SALICYLIC], 2).precedents) == 2
    assert len(fake.search_queries) == 2


def test_find_keeps_exact_hits_when_the_similar_search_fails():
    ord_finder, _ = finder(fail_search=1)
    response = ord_finder.find(ASPIRIN, [SALICYLIC], 3)
    assert response.precedents


def test_find_reports_api_failure_as_a_value():
    ord_finder, _ = finder(fail_search=0)
    result = ord_finder.find(ASPIRIN, [SALICYLIC], 3)
    assert isinstance(result, ToolError)
    assert "busy" in result.error
    assert "do not guess" in result.hint


def test_find_reports_no_hits_and_bad_input():
    empty = OrdPrecedents(FakeOrdClient([[], []], []))
    assert "No patent reaction" in empty.find(ASPIRIN, [SALICYLIC], 3).error
    assert "Invalid SMILES" in empty.find("not a smiles", [SALICYLIC], 3).error
    assert "At most" in empty.find(ASPIRIN, [SALICYLIC] * 6, 3).error


def test_server_uses_ord_backend_from_the_environment(monkeypatch):
    from mcp_server import precedent_server

    monkeypatch.setenv(precedent_server.BACKEND_ENV, "ord")
    precedent_server._search.cache_clear()
    try:
        assert isinstance(precedent_server._search(), OrdPrecedents)
    finally:
        precedent_server._search.cache_clear()


def test_assistant_config_for_ord_needs_no_local_data():
    settings = Settings(_env_file=None, enable_precedents=True, precedents_backend="ord")
    config = _precedent_server_config(settings)
    assert config["env"]["PRECEDENTS_BACKEND"] == "ord"
    assert "PRECEDENTS_DATA_DIR" not in config["env"]
