"""Local-production pieces: keys, rate limit, log rotation, tool slots, startup checks."""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.tools import StructuredTool

from app.security import RateLimiter, key_id, match_key
from minichem.concurrency import limit_concurrency
from minichem.config import ConfigError, Settings
from minichem.jsonl import JsonlLog

KEY = "a-long-enough-key-1"


def test_match_key_accepts_only_exact_keys():
    keys = [KEY, "another-long-key-22"]
    assert match_key(KEY, keys) == KEY
    assert match_key("another-long-key-22", keys) == "another-long-key-22"
    assert match_key(KEY + "x", keys) is None
    assert match_key("", keys) is None
    assert match_key(None, keys) is None
    assert match_key("anything", []) is None


def test_key_id_is_short_stable_and_not_the_key():
    assert key_id(KEY) == key_id(KEY)
    assert len(key_id(KEY)) == 8
    assert KEY not in key_id(KEY)


def test_rate_limiter_counts_per_identity_and_forgets_after_the_window():
    now = [100.0]
    limiter = RateLimiter(limit=2, window_s=60, clock=lambda: now[0])
    assert limiter.check("a") is None
    now[0] += 10
    assert limiter.check("a") is None
    now[0] += 10
    wait = limiter.check("a")
    assert wait == pytest.approx(40.0)
    assert limiter.check("b") is None
    now[0] += 41
    assert limiter.check("a") is None


def test_jsonl_log_rotates_and_keeps_a_limited_number_of_files(tmp_path):
    path = tmp_path / "log.jsonl"
    log = JsonlLog(path, max_bytes=300, backups=2)
    for i in range(40):
        log.write({"i": i, "pad": "x" * 40})
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["log.jsonl", "log.jsonl.1", "log.jsonl.2"]
    assert max(p.stat().st_size for p in tmp_path.iterdir()) <= 400
    newest = [json.loads(x) for x in path.read_text().splitlines()]
    assert newest[-1]["i"] == 39


def make_tool(delay_s: float, log: list[str]) -> StructuredTool:
    async def run(smiles: str) -> tuple[str, None]:
        log.append(f"start {smiles}")
        await asyncio.sleep(delay_s)
        log.append(f"end {smiles}")
        return json.dumps({"ok": smiles}), None

    return StructuredTool(
        name="retrosynthesis_routes",
        description="search",
        args_schema={"type": "object", "properties": {"smiles": {"type": "string"}}},
        coroutine=run,
        response_format="content_and_artifact",
    )


async def test_limited_tool_runs_searches_one_after_another():
    log: list[str] = []
    tool = limit_concurrency(make_tool(0.05, log), asyncio.Semaphore(1), wait_s=5)
    await asyncio.gather(tool.ainvoke({"smiles": "a"}), tool.ainvoke({"smiles": "b"}))
    assert log == ["start a", "end a", "start b", "end b"]


async def test_limited_tool_reports_busy_as_a_value_when_no_slot_frees_up():
    log: list[str] = []
    slots = asyncio.Semaphore(1)
    tool = limit_concurrency(make_tool(0.3, log), slots, wait_s=0.05)
    first = asyncio.create_task(tool.ainvoke({"smiles": "a"}))
    await asyncio.sleep(0.02)
    busy = json.loads(await tool.ainvoke({"smiles": "b"}))
    await first
    assert "busy" in busy["error"]
    assert "retry" in busy["hint"]
    assert log == ["start a", "end a"]
    assert not slots.locked()


async def test_slot_is_released_when_the_search_fails():
    async def boom(smiles: str) -> tuple[str, None]:
        raise RuntimeError("search crashed")

    tool = StructuredTool(
        name="t",
        description="d",
        args_schema={"type": "object", "properties": {"smiles": {"type": "string"}}},
        coroutine=boom,
        response_format="content_and_artifact",
    )
    slots = asyncio.Semaphore(1)
    limited = limit_concurrency(tool, slots, wait_s=1)
    with pytest.raises(RuntimeError, match="search crashed"):
        await limited.ainvoke({"smiles": "a"})
    assert not slots.locked()


def test_defaults_need_no_auth_but_need_the_llm_key():
    settings = Settings(_env_file=None)
    assert settings.api_key_list() == []
    assert settings.startup_problems() == ["GOOGLE_API_KEY is not set (see .env.example)"]
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        settings.check_startup()


def test_key_check_can_be_switched_off_for_tests():
    assert Settings(_env_file=None, require_llm_key=False).startup_problems() == []


def test_auth_required_without_keys_and_short_keys_are_startup_problems():
    base = {"_env_file": None, "require_llm_key": False}
    assert Settings(**base, auth_required=True).startup_problems() == [
        "AUTH_REQUIRED is true but API_KEYS is empty"
    ]
    short = Settings(**base, api_keys="short").startup_problems()
    assert short == ["every API key must have at least 16 characters"]
    assert Settings(**base, api_keys=f"{KEY}, ,{KEY}2").api_key_list() == [KEY, KEY + "2"]


def test_retrosynthesis_needs_existing_paths(tmp_path):
    settings = Settings(_env_file=None, require_llm_key=False, enable_retrosynthesis=True)
    assert len(settings.startup_problems()) == 2
    present = tmp_path / "x"
    present.write_text("")
    ready = Settings(
        _env_file=None,
        require_llm_key=False,
        enable_retrosynthesis=True,
        aizynth_config=present,
        aizynth_python=present,
    )
    assert ready.startup_problems() == []


def test_settings_text_never_shows_secrets():
    settings = Settings(
        _env_file=None, GOOGLE_API_KEY="gemini-secret-value", api_keys="client-secret-key-123"
    )
    text = repr(settings) + str(settings) + str(settings.model_dump())
    assert "gemini-secret-value" not in text
    assert "client-secret-key-123" not in text
    with pytest.raises(ConfigError) as excinfo:
        Settings(_env_file=None, api_keys="secret-but-short").check_startup()
    assert "secret-but-short" not in str(excinfo.value)


def test_an_empty_llm_key_counts_as_missing():
    settings = Settings(_env_file=None, GOOGLE_API_KEY="  ")
    assert settings.startup_problems() == ["GOOGLE_API_KEY is not set (see .env.example)"]
    with pytest.raises(RuntimeError, match="No Gemini API key"):
        settings.require_api_key()
