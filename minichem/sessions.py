"""Lifetime and size limits for persistent conversation memory (SQLite checkpointer).

LangGraph stores one full copy of the state per step, so without limits the database only
grows. Four limits keep it bounded:
- compaction: after each turn only the newest checkpoint of the thread is kept (this graph
  never resumes from an older one);
- TTL: a session not used for `ttl_s` seconds is deleted;
- session cap: when more than `max_sessions` exist, the least recently used are deleted;
- turn cap: a session with `max_turns` turns refuses more (the caller starts a new session).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class SessionFullError(RuntimeError):
    """The session reached its turn cap."""


@dataclass(frozen=True)
class SessionLimits:
    ttl_s: float
    max_sessions: int
    max_turns: int


class SessionStore:
    """Activity bookkeeping and pruning next to an `AsyncSqliteSaver` (same connection)."""

    def __init__(self, saver: AsyncSqliteSaver, limits: SessionLimits) -> None:
        self._saver = saver
        self._limits = limits

    async def setup(self) -> None:
        """Create the saver's tables and the activity table, then drop expired sessions."""
        await self._saver.setup()
        await self._saver.conn.execute(
            "CREATE TABLE IF NOT EXISTS session_activity ("
            "thread_id TEXT PRIMARY KEY, last_seen REAL NOT NULL, turns INTEGER NOT NULL)"
        )
        await self._saver.conn.commit()
        await self.prune()

    async def start_turn(self, session_id: str) -> None:
        """Register the session; raise SessionFullError if it has used all its turns."""
        conn = self._saver.conn
        await conn.execute(
            "INSERT OR IGNORE INTO session_activity VALUES (?, ?, 0)", (session_id, time.time())
        )
        async with conn.execute(
            "SELECT turns FROM session_activity WHERE thread_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        await conn.commit()
        if row is not None and row[0] >= self._limits.max_turns:
            raise SessionFullError(
                f"session '{session_id}' has {row[0]} turns (limit {self._limits.max_turns}); "
                "start a new session_id"
            )

    async def finish_turn(self, session_id: str) -> None:
        """Count the turn, compact the thread's checkpoints, prune old sessions."""
        conn = self._saver.conn
        await conn.execute(
            "UPDATE session_activity SET last_seen = ?, turns = turns + 1 WHERE thread_id = ?",
            (time.time(), session_id),
        )
        await self._compact(session_id)
        await conn.commit()
        await self.prune()

    async def _compact(self, session_id: str) -> None:
        conn = self._saver.conn
        # checkpoint ids are time-ordered UUIDs, the saver itself sorts on them
        newest = (
            "SELECT MAX(checkpoint_id) FROM checkpoints WHERE thread_id = :t AND checkpoint_ns = ''"
        )
        for table in ("writes", "checkpoints"):
            await conn.execute(
                f"DELETE FROM {table} WHERE thread_id = :t AND checkpoint_ns = '' "
                f"AND checkpoint_id < ({newest})",
                {"t": session_id},
            )

    async def prune(self) -> int:
        """Delete expired sessions and the least recently used beyond the cap; return the count."""
        conn = self._saver.conn
        cutoff = time.time() - self._limits.ttl_s
        async with conn.execute(
            "SELECT thread_id FROM session_activity WHERE last_seen < ? "
            "UNION SELECT thread_id FROM ("
            "SELECT thread_id FROM session_activity ORDER BY last_seen DESC LIMIT -1 OFFSET ?)",
            (cutoff, self._limits.max_sessions),
        ) as cursor:
            doomed = [row[0] for row in await cursor.fetchall()]
        for thread_id in doomed:
            await self._saver.adelete_thread(thread_id)
            await conn.execute("DELETE FROM session_activity WHERE thread_id = ?", (thread_id,))
        await conn.commit()
        return len(doomed)

    async def count(self) -> int:
        """Number of stored sessions."""
        async with self._saver.conn.execute("SELECT COUNT(*) FROM session_activity") as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0
