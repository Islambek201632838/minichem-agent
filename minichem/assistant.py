"""Public entry point: build the production graph and answer one message at a time."""

from __future__ import annotations

import sys
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from minichem import molecules as mol
from minichem.agents.base import SubAgent
from minichem.agents.design import build_design_agent
from minichem.agents.supervisor import Decision
from minichem.agents.synthesis import build_synthesis_agent
from minichem.agents.utility import build_utility_agent
from minichem.config import PROJECT_ROOT, Settings, get_settings
from minichem.graph import RunContext, build_graph
from minichem.llm import build_chat_model
from minichem.prompts import load_prompts
from minichem.tools.pubchem import ChemLookup, OfflinePubChem, PubChemClient
from minichem.tools.retrosynthesis import RoutesResponse
from minichem.tracing import ToolCall, Tracer

MCP_SERVER_NAME = "rdkit"
RETRO_SERVER_NAME = "retrosynthesis"
ROUTES_TOOL = "retrosynthesis_routes"
RECURSION_LIMIT = 60  # outer graph: generous, the real budget is max_supervisor_steps


class MoleculeRef(BaseModel):
    id: str
    smiles: str


class ChatResult(BaseModel):
    """One answered turn."""

    session_id: str
    answer: str
    molecules: list[MoleculeRef]
    tool_calls: list[ToolCall]
    routes: list[RoutesResponse] = []  # full retrosynthesis routes, for drawing
    warnings: list[str]
    llm_calls: int
    input_tokens: int
    output_tokens: int
    latency_s: float


def _routes_of(tracer: Tracer) -> list[RoutesResponse]:
    """Routes returned by the retrosynthesis tool this turn (error results are skipped)."""
    return [
        RoutesResponse.model_validate(payload)
        for tool, payload in tracer.results
        if tool == ROUTES_TOOL and isinstance(payload, dict) and "routes" in payload
    ]


class ChemAssistant:
    """Wraps the compiled graph. Conversation memory is keyed by `session_id`."""

    def __init__(self, graph, settings: Settings) -> None:
        self._graph = graph
        self._settings = settings

    async def ask(self, message: str, session_id: str = "default") -> ChatResult:
        """Run one user turn through the supervisor graph."""
        tracer = Tracer(self._settings.trace_path, session_id)
        started = time.perf_counter()
        final = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config={
                "configurable": {"thread_id": session_id},
                "callbacks": [tracer],
                "recursion_limit": RECURSION_LIMIT,
            },
            context=RunContext(tracer=tracer),
        )
        registry = final["molecules"]
        refs = [
            MoleculeRef(id=mol.handle(i), smiles=registry[i])
            for s in final["turn_molecules"]
            if (i := mol.find(registry, s)) is not None
        ]
        return ChatResult(
            session_id=session_id,
            answer=final["answer"],
            molecules=refs,
            tool_calls=tracer.tool_calls,
            routes=_routes_of(tracer),
            warnings=final["warnings"],
            llm_calls=tracer.llm_calls,
            input_tokens=tracer.input_tokens,
            output_tokens=tracer.output_tokens,
            latency_s=round(time.perf_counter() - started, 2),
        )


def _build_lookup(settings: Settings) -> ChemLookup:
    if settings.pubchem_mode == "offline":
        return OfflinePubChem()
    return PubChemClient(settings.pubchem_cache_path, settings.pubchem_timeout_s)


def _retro_server_config(settings: Settings) -> dict:
    """Stdio config for the retrosynthesis server, which runs in AiZynthFinder's own venv."""
    if settings.aizynth_config is None or settings.aizynth_python is None:
        raise RuntimeError(
            "ENABLE_RETROSYNTHESIS needs AIZYNTH_CONFIG and AIZYNTH_PYTHON (see README)."
        )
    return {
        "transport": "stdio",
        "command": str(settings.aizynth_python),
        "args": ["-m", "mcp_server.retro_server"],
        "cwd": str(PROJECT_ROOT),
        "env": {"AIZYNTH_CONFIG": str(settings.aizynth_config), "PYTHONPATH": str(PROJECT_ROOT)},
    }


@asynccontextmanager
async def open_assistant(settings: Settings | None = None) -> AsyncIterator[ChemAssistant]:
    """Start the MCP server (stdio), build the agents and graph, and yield the assistant.

    One persistent MCP session per server is reused for every tool call (no process spawn per
    call). The retrosynthesis server is started only when ENABLE_RETROSYNTHESIS is set.
    """
    settings = settings or get_settings()
    prompts = load_prompts()
    llm = build_chat_model(settings)
    servers = {
        MCP_SERVER_NAME: {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "mcp_server.server"],
            "cwd": str(PROJECT_ROOT),
        }
    }
    if settings.enable_retrosynthesis:
        servers[RETRO_SERVER_NAME] = _retro_server_config(settings)
    client = MultiServerMCPClient(servers)
    async with AsyncExitStack() as stack:
        mcp_tools = await load_mcp_tools(
            await stack.enter_async_context(client.session(MCP_SERVER_NAME))
        )
        retro_tools: list[BaseTool] = []
        if settings.enable_retrosynthesis:
            retro_session = await stack.enter_async_context(client.session(RETRO_SERVER_NAME))
            retro_tools = await load_mcp_tools(retro_session)
        agents: dict[str, SubAgent] = {
            "utility": build_utility_agent(llm, prompts, _build_lookup(settings)),
            "design": build_design_agent(llm, prompts, mcp_tools),
            "synthesis": build_synthesis_agent(llm, prompts, retro_tools),
        }
        supervisor_llm = llm.with_structured_output(Decision).with_config(
            metadata={"agent": "supervisor"}
        )
        graph = build_graph(
            supervisor_llm=supervisor_llm,
            agents=agents,
            prompts=prompts,
            settings=settings,
            checkpointer=MemorySaver(),
        )
        yield ChemAssistant(graph, settings)
