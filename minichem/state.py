"""Graph state shared by the supervisor and the sub-agents."""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import MessagesState

AgentName = Literal["utility", "design", "synthesis"]
AGENT_NAMES: tuple[AgentName, ...] = ("utility", "design", "synthesis")


class AgentReport(TypedDict):
    """What a sub-agent did, as shown to the supervisor (molecules appear as handles)."""

    agent: str
    task: str
    notes: str
    tool_results: list[dict]


class ChemState(MessagesState):
    """`messages` and `molecules` persist per session; the rest is reset every turn."""

    molecules: list[str]  # session registry; handle Mk is molecules[k-1]
    turn_molecules: list[str]  # SMILES produced by tools this turn, in order
    reports: list[AgentReport]
    warnings: list[str]
    next_agent: str
    task: str
    steps: int
    answer: str
