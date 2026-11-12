"""FastAPI backend: POST /chat and GET /health."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from google.genai.errors import APIError
from pydantic import BaseModel, Field

from minichem.assistant import ChatResult, ChemAssistant, open_assistant
from minichem.config import get_settings

logger = logging.getLogger("minichem.api")


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
    """Start the MCP server and build the graph once, for the life of the process."""
    async with open_assistant() as assistant:
        app.state.assistant = assistant
        yield


app = FastAPI(title="MiniChemAgent", lifespan=lifespan)


def _assistant(request: Request) -> ChemAssistant:
    return request.app.state.assistant


@app.get("/health")
async def health() -> Health:
    settings = get_settings()
    return Health(status="ok", model=settings.gemini_model, pubchem_mode=settings.pubchem_mode)


@app.post("/chat")
async def chat(body: ChatRequest, request: Request) -> ChatResult:
    """Run one turn. Conversation memory is keyed by `session_id`."""
    try:
        return await _assistant(request).ask(body.message, body.session_id)
    except APIError as exc:
        # Raised after the SDK has exhausted its own retries (rate limit, outage, bad key).
        logger.warning("Gemini API error: %s", exc)
        raise HTTPException(status_code=503, detail=f"LLM provider error: {exc.message}") from exc
