"""Shared sub-agent machinery: build a tool-calling agent, run it, and report back."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from minichem import molecules as mol
from minichem.handles import SMILES_ARGS, HandleMiddleware, RunHandles, is_handle
from minichem.prompts import system_prompt
from minichem.state import AgentName, AgentReport

MAX_RESULT_CHARS = 1500

TASK_TEMPLATE = (
    "Task: {task}\n\n"
    "Molecules are given as handles (M1, M2, ...). Put the handle itself wherever a tool asks "
    "for a SMILES; the system substitutes the exact structure. Tool results show handles too. "
    "Never write a SMILES yourself."
)


@dataclass(frozen=True)
class SubAgentRun:
    """Outcome of one delegated task."""

    report: AgentReport
    registry: list[str]
    produced: list[str]  # molecules the tools returned, in order
    warnings: list[str]  # tool arguments where the model wrote a SMILES, not a handle


@dataclass(frozen=True)
class SubAgent:
    """A named tool-calling agent."""

    name: AgentName
    runnable: Runnable


def build_subagent(
    name: AgentName,
    llm: BaseChatModel,
    tools: list[BaseTool],
    prompts: dict[str, str],
) -> SubAgent:
    """Native tool-calling loop (model <-> tools) with the agent's prompt from `prompts/*.md`.

    `checkpointer=False` keeps the inner loop stateless; session memory lives in the parent graph.
    """
    graph = create_agent(
        llm,
        tools,
        system_prompt=system_prompt(prompts, name),
        middleware=[HandleMiddleware()],
        context_schema=RunHandles,
        checkpointer=False,
        name=name,
    )
    return SubAgent(name=name, runnable=graph.with_config(metadata={"agent": name}))


def _compact(payload: dict) -> str:
    text = json.dumps(payload, separators=(",", ":"))
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "...(truncated)"


def _final_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls and message.text:
            return str(message.text)
    return ""


def _raw_arguments(messages: list[Any], question: str) -> list[str]:
    """Tool arguments where the model wrote a SMILES instead of a handle.

    Handles are resolved by code (`HandleMiddleware`), so a SMILES typed by the model is a
    protocol violation: it can be a retyped, wrong structure. The only accepted raw value is
    one the user typed in this question, copied from the task when it had no handle yet.
    """
    flagged: list[str] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            for key in SMILES_ARGS & call["args"].keys():
                value = call["args"][key]
                for item in [value] if isinstance(value, str) else value:
                    if not is_handle(item) and item not in question and item not in flagged:
                        flagged.append(item)
    return flagged


async def run_subagent(
    agent: SubAgent,
    task: str,
    registry: list[str],
    question: str,
    config: RunnableConfig,
    max_iterations: int,
) -> SubAgentRun:
    """Run one delegated task. Molecules in the report are handles, never SMILES text;
    new SMILES enter the registry only from tool results."""
    prompt = TASK_TEMPLATE.format(task=task)
    handles = RunHandles(list(registry))
    notes = ""
    messages: list[Any] = []
    try:
        result = await agent.runnable.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={**config, "recursion_limit": max_iterations},
            context=handles,
        )
        messages = result["messages"]
        notes = _final_text(messages)
    except GraphRecursionError:
        notes = f"The {agent.name} agent stopped: it hit its iteration limit without finishing."

    warnings = [
        f"{agent.name} agent passed a SMILES instead of a handle: {s}"
        for s in _raw_arguments(messages, question)
    ]
    payloads = [
        (m.name or "tool", p)
        for m in messages
        if isinstance(m, ToolMessage) and (p := mol.payload_of(m)) is not None
    ]
    report: AgentReport = {
        "agent": agent.name,
        "task": task,
        "notes": mol.mask_smiles(notes, handles.registry),
        "tool_results": [{"tool": name, "result": _compact(payload)} for name, payload in payloads],
    }
    return SubAgentRun(report, handles.registry, handles.produced, warnings)
