"""FastAPI backend: POST /chat, GET /health, GET /metrics.

Protection of /chat, in the order a request meets it: API key (401), rate limit per key (429),
one turn at a time per session (409), cap on turns running at once (429), time limit (504).
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from google.genai.errors import APIError
from pydantic import BaseModel, Field

from app.metrics import Metrics
from app.security import ANONYMOUS, API_KEY_HEADER, RateLimiter, key_id, match_key
from minichem.assistant import ChatResult, ChemAssistant, open_assistant
from minichem.config import Settings, get_settings
from minichem.jsonl import JsonlLog
from minichem.sessions import SessionFullError

logger = logging.getLogger("minichem.api")

REQUEST_ID_HEADER = "X-Request-ID"
VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.-]{8,64}$")
BUSY_RETRY_AFTER_S = 5
UNMATCHED_PATH = "unmatched"  # metrics label for 404s, so scanners cannot create endless labels


class ChatRequest(BaseModel):
    """Body of POST /chat."""

    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]{1,64}$")


class Health(BaseModel):
    status: str
    model: str
    pubchem_mode: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Check the configuration, start the MCP server and build the graph for the process life."""
    settings = get_settings()
    settings.check_startup()
    if not settings.api_key_list():
        logger.warning("API_KEYS is empty: /chat and /metrics accept any caller (local use only)")
    app.state.settings = settings
    app.state.metrics = Metrics()
    app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
    app.state.chat_slots = asyncio.Semaphore(settings.max_concurrent_chats)
    app.state.active_sessions = set()
    app.state.request_log = JsonlLog(
        settings.api_log_path, settings.log_max_bytes, settings.log_backups
    )
    async with open_assistant(settings, persistent_sessions=True) as assistant:
        app.state.assistant = assistant
        yield


app = FastAPI(title="MiniChemAgent", lifespan=lifespan)


def _request_id(request: Request) -> str:
    supplied = request.headers.get(REQUEST_ID_HEADER, "")
    return supplied if VALID_REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex


@app.middleware("http")
async def observe(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Give the request an id, then log it as one JSON line and count it in the metrics."""
    request.state.request_id = _request_id(request)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response
    finally:
        seconds = time.perf_counter() - started
        route = request.scope.get("route")
        request.app.state.metrics.observe_request(
            route.path if route else UNMATCHED_PATH, status, seconds
        )
        request.app.state.request_log.write(
            {
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "kind": "request",
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": round(seconds * 1000, 1),
                "key_id": getattr(request.state, "key_id", None),
                "session_id": getattr(request.state, "session_id", None),
            }
        )


def _assistant(request: Request) -> ChemAssistant:
    return request.app.state.assistant


def authorize(request: Request) -> str:
    """Check the API key (when keys are configured) and return the caller's identity.

    The identity is a short hash of the key, or the client address when authentication is off.
    """
    settings: Settings = request.app.state.settings
    keys = settings.api_key_list()
    if not keys:
        client = request.client.host if request.client else "unknown"
        request.state.key_id = f"{ANONYMOUS}:{client}"
    else:
        matched = match_key(request.headers.get(API_KEY_HEADER), keys)
        if matched is None:
            raise HTTPException(
                status_code=401,
                detail=f"missing or invalid API key (header {API_KEY_HEADER})",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        request.state.key_id = key_id(matched)
    return request.state.key_id


@app.get("/health")
async def health(request: Request) -> Health:
    settings: Settings = request.app.state.settings
    return Health(status="ok", model=settings.gemini_model, pubchem_mode=settings.pubchem_mode)


@app.get("/metrics")
async def metrics(request: Request, _: str = Depends(authorize)) -> Response:
    """Request counts and latency, tool calls and errors, tokens (Prometheus text format)."""
    return Response(request.app.state.metrics.render(), media_type="text/plain; version=0.0.4")


def _reject(status: int, detail: str, retry_after_s: float | None = None) -> HTTPException:
    headers = {"Retry-After": str(max(1, math.ceil(retry_after_s)))} if retry_after_s else None
    return HTTPException(status_code=status, detail=detail, headers=headers)


@app.post("/chat")
async def chat(
    body: ChatRequest, request: Request, identity: str = Depends(authorize)
) -> ChatResult:
    """Run one turn. Conversation memory is keyed by `session_id`."""
    state = request.app.state
    settings: Settings = state.settings
    request.state.session_id = body.session_id
    if (wait := state.limiter.check(identity)) is not None:
        raise _reject(
            429, f"rate limit: {settings.rate_limit_per_minute} requests per minute", wait
        )
    if body.session_id in state.active_sessions:
        raise _reject(409, f"session '{body.session_id}' is still answering a previous message")
    if state.chat_slots.locked():
        raise _reject(
            429, "server busy: too many questions running, retry shortly", BUSY_RETRY_AFTER_S
        )
    state.active_sessions.add(body.session_id)
    try:
        async with state.chat_slots, asyncio.timeout(settings.request_timeout_s) as limit:
            result = await _assistant(request).ask(
                body.message, body.session_id, request.state.request_id
            )
    except TimeoutError as exc:
        if not limit.expired():
            raise
        logger.warning("turn exceeded %s s", settings.request_timeout_s)
        raise _reject(504, f"the answer took longer than {settings.request_timeout_s:g} s") from exc
    except SessionFullError as exc:
        raise _reject(409, str(exc)) from exc
    except APIError as exc:
        # Raised after the SDK has exhausted its own retries (rate limit, outage, bad key).
        logger.warning("Gemini API error: %s", exc)
        raise HTTPException(status_code=503, detail=f"LLM provider error: {exc.message}") from exc
    finally:
        state.active_sessions.discard(body.session_id)
    state.metrics.observe_turn(result.tool_calls, result.input_tokens, result.output_tokens)
    return result
