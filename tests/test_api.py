import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai.errors import APIError

from app import api
from minichem.assistant import ChatResult, MoleculeRef
from minichem.config import Settings
from minichem.sessions import SessionFullError
from minichem.tracing import ToolCall

KEY = "test-key-0123456789"
OTHER_KEY = "other-key-0123456789"


class StubAssistant:
    """Records calls; raises when asked to, so the error mapping can be tested."""

    def __init__(self, error: Exception | None = None, delay_s: float = 0.0) -> None:
        self.error = error
        self.delay_s = delay_s
        self.calls: list[tuple[str, str]] = []
        self.request_ids: list[str] = []

    async def ask(
        self, message: str, session_id: str = "default", request_id: str = ""
    ) -> ChatResult:
        self.calls.append((message, session_id))
        self.request_ids.append(request_id)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error:
            raise self.error
        call = ToolCall("utility", "name_to_smiles", {"name": "x"}, "{}", 1.0, None)
        failed = ToolCall("design", "validate_smiles", {"smiles": "?"}, "{}", 2.0, "bad SMILES")
        return ChatResult(
            session_id=session_id,
            request_id=request_id,
            answer="ok",
            molecules=[MoleculeRef(id="M1", smiles="CCO")],
            tool_calls=[call, failed],
            warnings=[],
            llm_calls=2,
            input_tokens=10,
            output_tokens=5,
            latency_s=0.1,
        )


def make_settings(tmp_path, **overrides) -> Settings:
    """Settings that need no key and write logs under tmp_path."""
    values = {
        "require_llm_key": False,
        "trace_path": tmp_path / "trace.jsonl",
        "api_log_path": tmp_path / "api.jsonl",
        "session_db_path": tmp_path / "sessions.sqlite",
        **overrides,
    }
    return Settings(_env_file=None, **values)


@pytest.fixture
def install_stub(monkeypatch, tmp_path):
    """Patch settings and the assistant; returns (stub, settings)."""

    def install(error: Exception | None = None, delay_s: float = 0.0, **overrides):
        settings = make_settings(tmp_path, **overrides)
        stub = StubAssistant(error, delay_s)

        @asynccontextmanager
        async def fake_open(*args, **kwargs):
            yield stub

        monkeypatch.setattr(api, "get_settings", lambda: settings)
        monkeypatch.setattr(api, "open_assistant", fake_open)
        return stub, settings

    return install


@pytest.fixture
def stub_client(install_stub):
    def make(error: Exception | None = None, **overrides):
        stub, _ = install_stub(error, **overrides)
        return TestClient(api.app), stub

    return make


@asynccontextmanager
async def async_client():
    """httpx client against the app with its lifespan running (for concurrent requests)."""
    app = api.app
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
    ):
        yield client


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


def test_chat_without_keys_configured_accepts_anyone(stub_client):
    client, stub = stub_client()
    with client:
        assert client.post("/chat", json={"message": "hi"}).status_code == 200
    assert len(stub.calls) == 1


def test_chat_with_keys_rejects_missing_and_wrong_key(stub_client):
    client, stub = stub_client(api_keys=f"{KEY},{OTHER_KEY}")
    with client:
        no_key = client.post("/chat", json={"message": "hi"})
        wrong = client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "wrong"})
        ok = client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": OTHER_KEY})
        health = client.get("/health")
    assert no_key.status_code == 401
    assert no_key.headers["www-authenticate"] == "ApiKey"
    assert wrong.status_code == 401
    assert KEY not in wrong.text
    assert ok.status_code == 200
    assert health.status_code == 200  # health stays open for the container healthcheck
    assert len(stub.calls) == 1


def test_rate_limit_is_per_key(stub_client):
    client, _ = stub_client(api_keys=f"{KEY},{OTHER_KEY}", rate_limit_per_minute=2)
    with client:
        statuses = [
            client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": KEY}).status_code
            for _ in range(3)
        ]
        limited = client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": KEY})
        other = client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": OTHER_KEY})
    assert statuses == [200, 200, 429]
    assert 1 <= int(limited.headers["retry-after"]) <= 60
    assert "rate limit" in limited.json()["detail"]
    assert other.status_code == 200


def test_session_full_becomes_409(stub_client):
    client, _ = stub_client(SessionFullError("session 'a' has 200 turns"))
    with client:
        response = client.post("/chat", json={"message": "hi", "session_id": "a"})
    assert response.status_code == 409
    assert "turns" in response.json()["detail"]


def test_slow_turn_becomes_504_and_frees_the_session(install_stub):
    stub, _ = install_stub(delay_s=5, request_timeout_s=0.1)
    with TestClient(api.app) as client:
        slow = client.post("/chat", json={"message": "hi", "session_id": "s"})
        stub.delay_s = 0
        again = client.post("/chat", json={"message": "hi", "session_id": "s"})
    assert slow.status_code == 504
    assert "longer than" in slow.json()["detail"]
    assert again.status_code == 200


def test_request_id_is_returned_logged_and_passed_to_the_assistant(install_stub):
    stub, settings = install_stub()
    with TestClient(api.app) as client:
        supplied = client.post(
            "/chat", json={"message": "hi"}, headers={"X-Request-ID": "req-12345678"}
        )
        generated = client.post("/chat", json={"message": "hi"}, headers={"X-Request-ID": "no"})
    assert supplied.headers["x-request-id"] == "req-12345678"
    assert supplied.json()["request_id"] == "req-12345678"
    new_id = generated.headers["x-request-id"]
    assert new_id != "no"
    assert len(new_id) == 32
    assert stub.request_ids == ["req-12345678", new_id]
    lines = [json.loads(x) for x in settings.api_log_path.read_text().splitlines()]
    assert [x["request_id"] for x in lines] == ["req-12345678", new_id]
    assert lines[0]["status"] == 200
    assert lines[0]["path"] == "/chat"
    assert lines[0]["session_id"] == "default"
    assert lines[0]["key_id"].startswith("anonymous")


def test_request_log_never_contains_the_key(install_stub):
    _, settings = install_stub(api_keys=KEY)
    with TestClient(api.app) as client:
        client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": KEY})
        client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "wrong-key-0123456789"})
    text = settings.api_log_path.read_text()
    assert KEY not in text
    assert "wrong-key" not in text
    assert [json.loads(x)["status"] for x in text.splitlines()] == [200, 401]


def test_metrics_count_requests_tools_errors_and_need_the_key(stub_client):
    client, _ = stub_client(api_keys=KEY)
    headers = {"X-API-Key": KEY}
    with client:
        client.post("/chat", json={"message": "hi"}, headers=headers)
        client.post("/chat", json={"message": "hi"})
        client.get("/nowhere")
        denied = client.get("/metrics")
        text = client.get("/metrics", headers=headers).text
    assert denied.status_code == 401
    assert 'minichem_requests_total{path="/chat",status="200"} 1.0' in text
    assert 'minichem_requests_total{path="/chat",status="401"} 1.0' in text
    assert 'minichem_requests_total{path="unmatched",status="404"} 1.0' in text
    assert 'minichem_tool_calls_total{agent="utility",tool="name_to_smiles"} 1.0' in text
    assert 'minichem_tool_errors_total{tool="validate_smiles"} 1.0' in text
    assert 'minichem_llm_tokens_total{direction="input"} 10.0' in text
    assert "minichem_request_seconds_bucket" in text


async def test_second_message_of_a_busy_session_gets_409(install_stub):
    install_stub(delay_s=0.3)
    async with async_client() as client:
        first = asyncio.create_task(client.post("/chat", json={"message": "a", "session_id": "s"}))
        await asyncio.sleep(0.1)
        second = await client.post("/chat", json={"message": "b", "session_id": "s"})
        other_session = await client.post("/chat", json={"message": "c", "session_id": "t"})
        assert (await first).status_code == 200
    assert second.status_code == 409
    assert other_session.status_code == 200


async def test_too_many_turns_at_once_get_429_with_retry_after(install_stub):
    install_stub(delay_s=0.3, max_concurrent_chats=1)
    async with async_client() as client:
        first = asyncio.create_task(client.post("/chat", json={"message": "a", "session_id": "s"}))
        await asyncio.sleep(0.1)
        busy = await client.post("/chat", json={"message": "b", "session_id": "t"})
        assert (await first).status_code == 200
        after = await client.post("/chat", json={"message": "b", "session_id": "t"})
    assert busy.status_code == 429
    assert busy.headers["retry-after"] == "5"
    assert "busy" in busy.json()["detail"]
    assert after.status_code == 200


def test_startup_fails_without_llm_key_when_required(install_stub):
    _, settings = install_stub(require_llm_key=True)
    assert settings.google_api_key is None
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY is not set"), TestClient(api.app):
        pass
