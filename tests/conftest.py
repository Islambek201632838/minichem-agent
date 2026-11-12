"""Shared constants and fixtures. No test touches the network or a real LLM."""

from __future__ import annotations

import pytest

ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


# config.py loads .env into os.environ on import; a developer's real keys and paths must not
# leak into tests that build Settings(_env_file=None).
CONFIG_ENV_VARS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "API_KEYS",
    "AUTH_REQUIRED",
    "REQUIRE_LLM_KEY",
    "ENABLE_RETROSYNTHESIS",
    "AIZYNTH_CONFIG",
    "AIZYNTH_PYTHON",
)


@pytest.fixture(autouse=True)
def clean_config_env(monkeypatch):
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "pubchem.json"
