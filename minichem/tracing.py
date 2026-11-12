"""Structured tracing: one JSON line per agent step, tool call and LLM call.

A `Tracer` is created per request and passed to LangChain as a callback handler, so it sees
local tools and MCP tools alike. The agent name travels in run metadata (`agent`).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

SUMMARY_CHARS = 300


@dataclass
class ToolCall:
    """One tool invocation, as returned by the API and written to the trace."""

    agent: str
    tool: str
    arguments: dict[str, Any]
    result_summary: str
    duration_ms: float
    error: str | None


def _decode(output: Any) -> Any:
    content = getattr(output, "content", output)
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content


def error_of(payload: Any) -> str | None:
    """The error message in a tool result, if it reports one (whole-call or per-item)."""
    if not isinstance(payload, dict):
        return None
    if payload.get("error"):
        return str(payload["error"])
    items = payload.get("errors")
    if items:
        return "; ".join(str(i.get("error", i)) for i in items)
    return None


class Tracer(BaseCallbackHandler):
    """Collects tool calls and token usage in memory and appends trace lines to a JSONL file."""

    run_inline = True

    def __init__(self, path: Path | None, session_id: str = "") -> None:
        self._path = path
        self._session_id = session_id
        self._open: dict[UUID, dict[str, Any]] = {}
        self.tool_calls: list[ToolCall] = []
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _write(self, kind: str, agent: str, **fields: Any) -> None:
        if self._path is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "session_id": self._session_id,
            "kind": kind,
            "agent": agent,
            **fields,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def step(self, agent: str, event: str, **detail: Any) -> None:
        """Record a non-tool step, e.g. a supervisor routing decision."""
        self._write("step", agent, event=event, **detail)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._open[run_id] = {
            "agent": (metadata or {}).get("agent", "unknown"),
            "tool": kwargs.get("name") or serialized.get("name", "unknown"),
            "arguments": inputs if inputs is not None else {"input": input_str},
            "started": time.perf_counter(),
        }

    def _finish_tool(self, run_id: UUID, summary: str, error: str | None) -> None:
        info = self._open.pop(run_id, None)
        if info is None:
            return
        call = ToolCall(
            agent=info["agent"],
            tool=info["tool"],
            arguments=info["arguments"],
            result_summary=summary[:SUMMARY_CHARS],
            duration_ms=round((time.perf_counter() - info["started"]) * 1000, 1),
            error=error,
        )
        self.tool_calls.append(call)
        self._write("tool", call.agent, **{k: v for k, v in asdict(call).items() if k != "agent"})

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        payload = _decode(output)
        error = error_of(payload)
        if error is None and getattr(output, "status", None) == "error":
            error = str(payload)  # e.g. MCP argument-validation failures arrive as plain text
        self._finish_tool(run_id, json.dumps(payload, default=str), error)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish_tool(run_id, "", f"{type(error).__name__}: {error}")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._open[run_id] = {
            "agent": (metadata or {}).get("agent", "unknown"),
            "started": time.perf_counter(),
        }

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        info = self._open.pop(run_id, None)
        in_tok = out_tok = 0
        for generation in response.generations[0] if response.generations else []:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
            in_tok += usage.get("input_tokens", 0)
            out_tok += usage.get("output_tokens", 0)
        self.llm_calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        if info is not None:
            self._write(
                "llm",
                info["agent"],
                duration_ms=round((time.perf_counter() - info["started"]) * 1000, 1),
                input_tokens=in_tok,
                output_tokens=out_tok,
            )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        info = self._open.pop(run_id, None)
        if info is not None:
            self._write("llm", info["agent"], error=f"{type(error).__name__}: {error}")
