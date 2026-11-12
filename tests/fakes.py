"""Deterministic stand-ins for the LLMs, so agent tests are fast and need no API key."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from minichem.agents.supervisor import Decision

Reply = AIMessage | Callable[[list[BaseMessage]], AIMessage]


def tool_call(tool_name: str, /, **args: Any) -> AIMessage:
    """An AI message that requests one tool call."""
    call = {"name": tool_name, "args": args, "id": f"call_{uuid4().hex[:8]}"}
    return AIMessage(content="", tool_calls=[call])


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of replies; a reply may be a function of the incoming messages."""

    replies: list[Any]
    seen: list[Any] = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any):
        if self.cursor >= len(self.replies):
            raise AssertionError("ScriptedChatModel ran out of replies")
        reply = self.replies[self.cursor]
        self.cursor += 1
        self.seen.append(messages)
        message = reply(messages) if callable(reply) else reply
        return ChatResult(generations=[ChatGeneration(message=message)])


class ScriptedSupervisor:
    """Replays Decisions and records every message list it was shown."""

    def __init__(self, decisions: list[Decision | None]) -> None:
        self._decisions = list(decisions)
        self.seen: list[list[BaseMessage]] = []
        self.runnable = RunnableLambda(self._next)

    def _next(self, messages: list[BaseMessage]) -> Decision | None:
        self.seen.append(messages)
        if not self._decisions:
            raise AssertionError("ScriptedSupervisor ran out of decisions")
        return self._decisions.pop(0)

    @property
    def shown_text(self) -> str:
        return "\n".join(str(m.content) for call in self.seen for m in call)


def route(agent: str, task: str) -> Decision:
    return Decision(next=agent, task=task)


def finish(answer: str) -> Decision:
    return Decision(next="FINISH", final_answer=answer)
