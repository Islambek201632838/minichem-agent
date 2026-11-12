"""LangGraph wiring: intake -> supervisor <-> {utility, design, synthesis} -> finalize."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from minichem import molecules as mol
from minichem.agents.base import SubAgent, run_subagent
from minichem.agents.supervisor import decide, fallback_answer
from minichem.config import Settings
from minichem.state import AGENT_NAMES, ChemState
from minichem.tracing import Tracer


@dataclass
class RunContext:
    """Per-request objects that must not live in the checkpointed state."""

    tracer: Tracer


def build_graph(
    *,
    supervisor_llm: Runnable,
    agents: dict[str, SubAgent],
    prompts: dict[str, str],
    settings: Settings,
    checkpointer: BaseCheckpointSaver | None,
):
    """Compile the multi-agent graph. LLMs and agents are injected so tests can fake them."""
    missing = set(AGENT_NAMES) - set(agents)
    if missing:
        raise ValueError(f"Missing sub-agents: {sorted(missing)}")

    async def intake(state: ChemState) -> dict[str, Any]:
        """Register SMILES typed by the user and reset per-turn state."""
        question = str(state["messages"][-1].content)
        registry = mol.register(state.get("molecules", []), mol.smiles_in_text(question))
        return {
            "molecules": registry,
            "turn_molecules": [],
            "reports": [],
            "warnings": [],
            "steps": 0,
            "answer": "",
            "next_agent": "",
            "task": "",
        }

    async def supervisor(state: ChemState, runtime: Runtime[RunContext]) -> Command:
        steps = state["steps"]
        steps_left = settings.max_supervisor_steps - steps
        decision = await decide(
            supervisor_llm, state, prompts["supervisor"], steps_left, settings.history_messages
        )
        reports = state["reports"]
        usable = decision is not None and (
            decision.final_answer.strip()
            if decision.next == "FINISH"
            else decision.task.strip() and steps_left > 0
        )
        if not usable:
            answer = decision.final_answer if decision and decision.final_answer else ""
            answer = answer or fallback_answer(reports)
            runtime.context.tracer.step("supervisor", "finish_fallback")
            return Command(goto="finalize", update={"answer": answer})
        if decision.next == "FINISH":
            runtime.context.tracer.step("supervisor", "finish")
            return Command(goto="finalize", update={"answer": decision.final_answer})
        runtime.context.tracer.step("supervisor", "route", next=decision.next, task=decision.task)
        return Command(
            goto=decision.next,
            update={"next_agent": decision.next, "task": decision.task, "steps": steps + 1},
        )

    def agent_node(agent: SubAgent):
        async def node(
            state: ChemState, config: RunnableConfig, runtime: Runtime[RunContext]
        ) -> dict[str, Any]:
            runtime.context.tracer.step(agent.name, "start", task=state["task"])
            run = await run_subagent(
                agent,
                state["task"],
                state["molecules"],
                str(state["messages"][-1].content),
                config,
                settings.max_agent_iterations,
            )
            for warning in run.warnings:
                runtime.context.tracer.step(
                    agent.name, "unverified_smiles_argument", detail=warning
                )
            turn = list(state["turn_molecules"])
            turn.extend(s for s in run.produced if s not in turn)
            return {
                "molecules": run.registry,
                "turn_molecules": turn,
                "reports": [*state["reports"], run.report],
                "warnings": [*state["warnings"], *run.warnings],
            }

        return node

    async def finalize(state: ChemState) -> dict[str, Any]:
        """Swap handles for exact SMILES and flag any SMILES that no tool produced."""
        registry = state["molecules"]
        answer = mol.expand_handles(state["answer"], registry, quote=True)
        suspicious = mol.unverified_smiles(answer, registry)
        if suspicious:
            answer += (
                "\n\nWarning: these SMILES in the text did not come from a tool and may be "
                "wrong: " + ", ".join(f"`{s}`" for s in suspicious)
            )
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    graph = StateGraph(ChemState, context_schema=RunContext)
    graph.add_node("intake", intake)
    graph.add_node("supervisor", supervisor, destinations=(*AGENT_NAMES, "finalize"))
    for name in AGENT_NAMES:
        graph.add_node(name, agent_node(agents[name]))
        graph.add_edge(name, "supervisor")
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "supervisor")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
