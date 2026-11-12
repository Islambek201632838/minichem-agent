"""Single source of configuration. Everything tunable lives here."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Populate os.environ too, so child processes and libraries see the same values.
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Runtime settings, read from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    google_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY")
    )
    gemini_model: str = "gemini-3.5-flash-lite"
    llm_max_retries: int = Field(default=5, ge=1, description="Attempts incl. the first call")
    llm_timeout_s: float = 60.0

    max_supervisor_steps: int = Field(default=6, ge=1)
    max_agent_iterations: int = Field(default=12, ge=2, description="LangGraph recursion limit")
    history_messages: int = Field(default=6, ge=0, description="Past messages shown to supervisor")

    pubchem_mode: Literal["live", "offline"] = "live"
    pubchem_timeout_s: float = 10.0
    pubchem_cache_path: Path = PROJECT_ROOT / ".cache" / "pubchem.json"

    trace_path: Path = PROJECT_ROOT / "logs" / "trace.jsonl"
    feedback_path: Path = PROJECT_ROOT / "logs" / "feedback.jsonl"
    api_url: str = "http://127.0.0.1:8000"

    enable_retrosynthesis: bool = False
    aizynth_config: Path | None = None

    def require_api_key(self) -> str:
        """Return the Gemini key or fail with an actionable message."""
        if self.google_api_key is None:
            raise RuntimeError(
                "No Gemini API key found. Set GOOGLE_API_KEY in .env (see .env.example)."
            )
        return self.google_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Cached process-wide settings."""
    return Settings()
