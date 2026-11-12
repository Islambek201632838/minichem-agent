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
from minichem.prompts import system_prompt
from minichem.state import AgentName, AgentReport

MAX_RESULT_CHARS = 1500

TASK_TEMPLATE = (
    "Task: {task}\n\n"
    "SMILES in the task above are exact; copy them character for character into tool calls."
)


SMILES_ARGS = frozenset({"smiles", "smiles_list", "query", "candidates"})


@dataclass(frozen=True)
class SubAgentRun:
    """Outcome of one delegated task."""

    report: AgentReport
    registry: list[str]
    produced: list[str]  # SMILES the tools returned, in order
    warnings: list[str]  # tool arguments whose SMILES were not given verbatim


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
        checkpointer=False,
        name=name,
    )
    return SubAgent(name=name, runnable=graph.with_config(metadata={"agent": name}))


def _compact(payload: dict, registry: list[str]) -> str:
    text = json.dumps(mol.with_handles(payload, registry), separators=(",", ":"))
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "...(truncated)"


def _final_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls and message.text:
            return str(message.text)
    return ""


def _retyped_arguments(messages: list[Any], registry: list[str], question: str) -> list[str]:
    """SMILES passed to tools that were never given to the agent.

    A SMILES is legitimate if it was in the registry before the run, in the user's question,
    or returned by an earlier tool call in this run. Anything else was typed by the model.
    The model is told to copy SMILES, but nothing in the protocol can force it to; this
    detects the cases where it did not.
    """
    known = list(registry)
    flagged: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage) and (payload := mol.payload_of(message)):
            known = mol.register(known, mol.collect_smiles(payload))
        elif isinstance(message, AIMessage):
            for call in message.tool_calls:
                for key in SMILES_ARGS & call["args"].keys():
                    value = call["args"][key]
                    for smiles in [value] if isinstance(value, str) else value:
                        given = mol.find(known, smiles) is not None or smiles in question
                        if not given and smiles not in flagged:
                            flagged.append(smiles)
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
    prompt = TASK_TEMPLATE.format(task=mol.expand_handles(task, registry))
    notes = ""
    messages: list[Any] = []
    try:
        result = await agent.runnable.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={**config, "recursion_limit": max_iterations},
        )
        messages = result["messages"]
        notes = _final_text(messages)
    except GraphRecursionError:
        notes = f"The {agent.name} agent stopped: it hit its iteration limit without finishing."

    warnings = [
        f"{agent.name} agent passed a SMILES it was not given: {s}"
        for s in _retyped_arguments(messages, registry, question)
    ]
    payloads = [
        (m.name or "tool", p)
        for m in messages
        if isinstance(m, ToolMessage) and (p := mol.payload_of(m)) is not None
    ]
    produced: list[str] = []
    for _, payload in payloads:
        produced.extend(s for s in mol.collect_smiles(payload) if s not in produced)
    registry = mol.register(registry, produced)

    report: AgentReport = {
        "agent": agent.name,
        "task": task,
        "notes": mol.mask_smiles(notes, registry),
        "tool_results": [
            {"tool": name, "result": _compact(payload, registry)} for name, payload in payloads
        ],
    }
    return SubAgentRun(report, registry, produced, warnings)
