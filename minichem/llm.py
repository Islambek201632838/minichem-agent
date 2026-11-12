"""LLM construction. Retries with exponential backoff are handled by the Gemini SDK."""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from minichem.config import Settings


def build_chat_model(settings: Settings) -> ChatGoogleGenerativeAI:
    """Gemini chat model.

    `max_retries` sets google-genai's retry policy: exponential backoff with jitter
    (1s doubling up to 60s) on HTTP 408, 429 and 5xx, which covers free-tier rate limits.
    Temperature is left at the model default: Google advises against lowering it for Gemini 3.
    """
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.require_api_key(),
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_s,
    )
