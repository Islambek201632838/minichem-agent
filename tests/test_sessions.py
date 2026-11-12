"""Persistent memory: survives a restart, and the limits keep the database small."""

from __future__ import annotations

import sqlite3
import time

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from minichem.sessions import SessionFullError, SessionLimits, SessionStore
from tests.fakes import ScriptedSupervisor, finish, route, tool_call
from tests.test_agents import make_assistant

DAY = 86400.0


def limits(**overrides) -> SessionLimits:
    values = {"ttl_s": 7 * DAY, "max_sessions": 100, "max_turns": 100, **overrides}
    return SessionLimits(**values)


def rows(db, sql: str) -> list[tuple]:
    with sqlite3.connect(db) as conn:
        return conn.execute(sql).fetchall()


async def test_memory_and_molecules_survive_reopening_the_database(tmp_path):
    db = tmp_path / "sessions.sqlite"

    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        store = SessionStore(saver, limits())
        await store.setup()
        supervisor = ScriptedSupervisor([route("utility", "Resolve ibuprofen."), finish("M1.")])
        assistant = make_assistant(
            tmp_path,
            supervisor,
            checkpointer=saver,
            sessions=store,
            utility=[tool_call("name_to_smiles", name="ibuprofen"), AIMessage(content="ok")],
        )
        await assistant.ask("SMILES of ibuprofen?", session_id="keep")

    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        store = SessionStore(saver, limits())
        await store.setup()
        supervisor = ScriptedSupervisor([finish("Still M1.")])
        assistant = make_assistant(
            tmp_path, supervisor, checkpointer=saver, sessions=store, utility=[]
        )
        result = await assistant.ask("And what was it?", session_id="keep")

    seen = supervisor.seen[0]
    assert any("SMILES of ibuprofen?" in str(m.content) for m in seen)
    assert "M1: C13H18O2" in str(seen[-1].content)
    assert result.answer.startswith("Still `")


async def test_each_turn_leaves_one_checkpoint_per_session_and_the_session_still_works(tmp_path):
    db = tmp_path / "sessions.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        store = SessionStore(saver, limits())
        await store.setup()
        supervisor = ScriptedSupervisor(
            [route("utility", "Resolve ibuprofen."), finish("M1."), finish("Second."), finish("3.")]
        )
        assistant = make_assistant(
            tmp_path,
            supervisor,
            checkpointer=saver,
            sessions=store,
            utility=[tool_call("name_to_smiles", name="ibuprofen"), AIMessage(content="ok")],
        )
        for _ in range(3):
            await assistant.ask("Question", session_id="s")
    assert rows(db, "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 's'") == [(1,)]
    assert rows(db, "SELECT turns FROM session_activity") == [(3,)]
    # the third turn still saw the first two (history is in the one remaining checkpoint)
    assert sum("Question" in str(m.content) for m in supervisor.seen[-1]) >= 3


async def test_turn_cap_refuses_further_turns(tmp_path):
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "s.sqlite")) as saver:
        store = SessionStore(saver, limits(max_turns=2))
        await store.setup()
        for _ in range(2):
            await store.start_turn("a")
            await store.finish_turn("a")
        with pytest.raises(SessionFullError, match="start a new session_id"):
            await store.start_turn("a")
        await store.start_turn("b")


async def test_expired_sessions_are_deleted_with_their_checkpoints(tmp_path):
    db = tmp_path / "s.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        store = SessionStore(saver, limits(ttl_s=3600))
        await store.setup()
        for name in ("old", "fresh"):
            await store.start_turn(name)
            await store.finish_turn(name)
            await saver.conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES (?, 'x')", (name,)
            )
        await saver.conn.execute(
            "UPDATE session_activity SET last_seen = ? WHERE thread_id = 'old'",
            (time.time() - 7200,),
        )
        await saver.conn.commit()
        assert await store.prune() == 1
    assert rows(db, "SELECT thread_id FROM session_activity") == [("fresh",)]
    assert rows(db, "SELECT DISTINCT thread_id FROM checkpoints") == [("fresh",)]


async def test_session_cap_removes_the_least_recently_used(tmp_path):
    db = tmp_path / "s.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        store = SessionStore(saver, limits(max_sessions=2))
        await store.setup()
        for name in ("a", "b", "c"):
            await store.start_turn(name)
            await store.finish_turn(name)
            time.sleep(0.01)
        assert await store.count() == 2
    assert sorted(rows(db, "SELECT thread_id FROM session_activity")) == [("b",), ("c",)]
