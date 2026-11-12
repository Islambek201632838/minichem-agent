from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from google.genai.errors import APIError

from app import api
from minichem.assistant import ChatResult, MoleculeRef
from minichem.tracing import ToolCall


class StubAssistant:
    """Records calls; raises when asked to, so the error mapping can be tested."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def ask(self, message: str, session_id: str = "default") -> ChatResult:
        self.calls.append((message, session_id))
        if self.error:
            raise self.error
        call = ToolCall("utility", "name_to_smiles", {"name": "x"}, "{}", 1.0, None)
        return ChatResult(
            session_id=session_id,
            answer="ok",
            molecules=[MoleculeRef(id="M1", smiles="CCO")],
            tool_calls=[call],
            warnings=[],
            llm_calls=2,
            input_tokens=10,
            output_tokens=5,
            latency_s=0.1,
        )


@pytest.fixture
def stub_client(monkeypatch):
    def make(error: Exception | None = None):
        stub = StubAssistant(error)

        @asynccontextmanager
        async def fake_open():
            yield stub

        monkeypatch.setattr(api, "open_assistant", fake_open)
        return TestClient(api.app), stub

    return make


def test_health(stub_client):
    client, _ = stub_client()
    with client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"]


def test_chat_returns_answer_and_tool_calls(stub_client):
    client, stub = stub_client()
    with client:
        response = client.post("/chat", json={"message": "hi", "session_id": "abc"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "ok"
    assert body["tool_calls"][0] == {
        "agent": "utility",
        "tool": "name_to_smiles",
        "arguments": {"name": "x"},
        "result_summary": "{}",
        "duration_ms": 1.0,
        "error": None,
    }
    assert stub.calls == [("hi", "abc")]


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "x" * 2001},
        {"message": "hi", "session_id": "has space"},
        {"session_id": "abc"},
    ],
)
def test_chat_validates_input(stub_client, payload):
    client, stub = stub_client()
    with client:
        assert client.post("/chat", json=payload).status_code == 422
    assert stub.calls == []


def test_provider_failure_becomes_503_not_a_stack_trace(stub_client):
    client, _ = stub_client(APIError(429, {"error": {"message": "quota exceeded"}}))
    with client:
        response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 503
    assert "quota exceeded" in response.json()["detail"]
