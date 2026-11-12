"""Supervisor: routes, decomposes into scoped tasks, and decides when to finish."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from minichem import molecules as mol
from minichem.state import AgentReport, ChemState

MAX_KNOWN_SHOWN = 30


class Decision(BaseModel):
    """The supervisor's structured output for one step."""

    next: Literal["utility", "design", "synthesis", "FINISH"] = Field(
        description="Which agent should act next, or FINISH to answer the user."
    )
    task: str = Field(
        default="", description="Scoped instruction for the agent. Use molecule handles (M1)."
    )
    final_answer: str = Field(
        default="", description="Complete answer for the user. Only when next is FINISH."
    )


def render_context(state: ChemState, steps_left: int) -> str:
    """The per-step context block: known molecules, agent reports, remaining budget."""
    registry = state.get("molecules", [])
    shown = range(max(0, len(registry) - MAX_KNOWN_SHOWN), len(registry))
    known = "\n".join(f"M{i + 1}: {mol.describe(registry[i])}" for i in shown) or "(none yet)"
    reports = state.get("reports", [])
    lines = [f"Known molecules:\n{known}", "", "Agent reports this turn:"]
    if not reports:
        lines.append("(none yet)")
    for n, report in enumerate(reports, 1):
        lines.append(f"{n}. [{report['agent']}] task: {report['task']}")
        if report["notes"]:
            lines.append(f"   notes: {report['notes']}")
        lines.extend(f"   {r['tool']} -> {r['result']}" for r in report["tool_results"])
    lines.append("")
    lines.append(f"Steps left: {steps_left}.")
    if steps_left <= 0:
        lines.append("You MUST choose FINISH now and answer from the reports.")
    return "\n".join(lines)


def build_messages(
    state: ChemState, system: str, steps_left: int, history: int
) -> list[BaseMessage]:
    """System prompt, recent conversation, then the context block.

    The conversation is shown with registered SMILES masked as handles, so the supervisor has
    no SMILES to retype.
    """
    registry = state.get("molecules", [])
    recent = list(state["messages"])[-history:] if history else list(state["messages"])[-1:]
    masked = [type(m)(content=mol.mask_smiles(str(m.content), registry)) for m in recent]
    context = HumanMessage(content=render_context(state, steps_left))
    return [SystemMessage(content=system), *masked, context]


def fallback_answer(reports: list[AgentReport]) -> str:
    """Used when the model returns an unusable decision: show raw tool results instead."""
    if not reports:
        return "I could not work out how to handle that request. Please rephrase it."
    lines = ["I could not compose a summary, so here are the raw tool results:"]
    for report in reports:
        lines.extend(f"- {r['tool']}: {r['result']}" for r in report["tool_results"])
    return "\n".join(lines)


async def decide(
    supervisor_llm: Runnable, state: ChemState, system: str, steps_left: int, history: int
) -> Decision | None:
    """Ask the model for the next decision. Returns None if it produced nothing usable."""
    result = await supervisor_llm.ainvoke(build_messages(state, system, steps_left, history))
    return result if isinstance(result, Decision) else None
