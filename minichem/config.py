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


MIN_API_KEY_CHARS = 16
API_KEY_HEADER = "X-API-Key"  # shared by the API and the UI; the UI cannot import the `app` package


class ConfigError(RuntimeError):
    """The configuration cannot run the service. The message never contains secret values."""


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
    api_log_path: Path = PROJECT_ROOT / "logs" / "api.jsonl"
    feedback_path: Path = PROJECT_ROOT / "logs" / "feedback.jsonl"
    log_max_bytes: int = Field(default=5_000_000, ge=10_000, description="Rotate a JSONL log at")
    log_backups: int = Field(default=3, ge=0, description="Rotated files kept per log")
    api_url: str = "http://127.0.0.1:8000"

    # API access. Comma-separated keys; empty means no authentication (local use only).
    api_keys: SecretStr = SecretStr("")
    auth_required: bool = Field(default=False, description="Refuse to start without API_KEYS")
    require_llm_key: bool = Field(
        default=True, description="Refuse to start without a Gemini key; false only for tests"
    )
    rate_limit_per_minute: int = Field(default=20, ge=1, description="/chat calls per key")
    max_concurrent_chats: int = Field(default=4, ge=1, description="Turns running at once")
    request_timeout_s: float = Field(default=300.0, gt=0, description="Hard limit of one turn")
    retro_max_concurrent: int = Field(default=1, ge=1, description="Searches running at once")
    retro_queue_wait_s: float = Field(
        default=150.0, ge=0, description="Wait for a free search slot"
    )

    session_db_path: Path = PROJECT_ROOT / ".cache" / "sessions.sqlite"
    session_ttl_days: float = Field(default=7.0, gt=0)
    session_max_sessions: int = Field(default=200, ge=1)
    session_max_turns: int = Field(default=200, ge=1)

    enable_retrosynthesis: bool = False
    aizynth_config: Path | None = None
    aizynth_python: Path | None = None
    enable_reinvent: bool = False
    reinvent_python: Path | None = None
    reinvent_prior_dir: Path | None = None
    reinvent_device: Literal["cpu", "mps"] = "mps"

    def has_llm_key(self) -> bool:
        """True if a non-empty Gemini key is configured (an empty variable counts as missing)."""
        return bool(self.google_api_key and self.google_api_key.get_secret_value().strip())

    enable_precedents: bool = False
    precedents_python: Path | None = None
    precedents_data_dir: Path | None = None

    def require_api_key(self) -> str:
        """Return the Gemini key or fail with an actionable message."""
        key = self.google_api_key.get_secret_value().strip() if self.google_api_key else ""
        if not key:
            raise RuntimeError(
                "No Gemini API key found. Set GOOGLE_API_KEY in .env (see .env.example)."
            )
        return key

    def api_key_list(self) -> list[str]:
        """The accepted API keys (empty list = authentication off)."""
        return [k.strip() for k in self.api_keys.get_secret_value().split(",") if k.strip()]

    def startup_problems(self) -> list[str]:
        """Everything that would make the service fail later, as messages without any values."""
        problems = []
        if self.require_llm_key and not self.has_llm_key():
            problems.append("GOOGLE_API_KEY is not set (see .env.example)")
        keys = self.api_key_list()
        if self.auth_required and not keys:
            problems.append("AUTH_REQUIRED is true but API_KEYS is empty")
        if any(len(k) < MIN_API_KEY_CHARS for k in keys):
            problems.append(f"every API key must have at least {MIN_API_KEY_CHARS} characters")
        if self.enable_retrosynthesis:
            for name, path in (
                ("AIZYNTH_CONFIG", self.aizynth_config),
                ("AIZYNTH_PYTHON", self.aizynth_python),
            ):
                if path is None or not path.exists():
                    problems.append(
                        f"ENABLE_RETROSYNTHESIS needs {name} to point to an existing file"
                    )
        return problems

    def check_startup(self) -> None:
        """Raise ConfigError listing all problems; used by the API before it builds anything."""
        problems = self.startup_problems()
        if problems:
            raise ConfigError("invalid configuration: " + "; ".join(problems))


@lru_cache
def get_settings() -> Settings:
    """Cached process-wide settings."""
    return Settings()
