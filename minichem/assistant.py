"""Public entry point: build the production graph and answer one message at a time."""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from minichem import molecules as mol
from minichem.agents.base import SubAgent
from minichem.agents.design import build_design_agent
from minichem.agents.supervisor import Decision
from minichem.agents.synthesis import build_synthesis_agent
from minichem.agents.utility import build_utility_agent
from minichem.concurrency import limit_concurrency
from minichem.config import PROJECT_ROOT, Settings, get_settings
from minichem.graph import RunContext, build_graph
from minichem.jsonl import JsonlLog
from minichem.llm import build_chat_model
from minichem.prompts import load_prompts
from minichem.sessions import SessionLimits, SessionStore
from minichem.tools.pubchem import ChemLookup, OfflinePubChem, PubChemClient
from minichem.tools.retrosynthesis import RoutesResponse
from minichem.tracing import ToolCall, Tracer

MCP_SERVER_NAME = "rdkit"
RETRO_SERVER_NAME = "retrosynthesis"
REINVENT_SERVER_NAME = "reinvent"
PRECEDENT_SERVER_NAME = "precedents"
ROUTES_TOOL = "retrosynthesis_routes"
RECURSION_LIMIT = 60  # outer graph: generous, the real budget is max_supervisor_steps


class MoleculeRef(BaseModel):
    id: str
    smiles: str


class ChatResult(BaseModel):
    """One answered turn."""

    session_id: str
    request_id: str = ""
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

    def __init__(self, graph, settings: Settings, sessions: SessionStore | None = None) -> None:
        self._graph = graph
        self._settings = settings
        self._sessions = sessions
        self._trace_log = JsonlLog(
            settings.trace_path, settings.log_max_bytes, settings.log_backups
        )

    async def ask(
        self, message: str, session_id: str = "default", request_id: str = ""
    ) -> ChatResult:
        """Run one user turn through the supervisor graph.

        `request_id` is written on every trace line of this turn. With persistent sessions,
        raises SessionFullError when the session used up its turns.
        """
        if self._sessions is not None:
            await self._sessions.start_turn(session_id)
        tracer = Tracer(self._trace_log, session_id, request_id)
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
        if self._sessions is not None:
            await self._sessions.finish_turn(session_id)
        registry = final["molecules"]
        refs = [
            MoleculeRef(id=mol.handle(i), smiles=registry[i])
            for s in final["turn_molecules"]
            if (i := mol.find(registry, s)) is not None
        ]
        return ChatResult(
            session_id=session_id,
            request_id=request_id,
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


def _reinvent_server_config(settings: Settings) -> dict:
    """Stdio config for the REINVENT server, which runs in REINVENT 4's own venv."""
    if settings.reinvent_python is None or settings.reinvent_prior_dir is None:
        raise RuntimeError(
            "ENABLE_REINVENT needs REINVENT_PYTHON and REINVENT_PRIOR_DIR (see README)."
        )
    return {
        "transport": "stdio",
        "command": str(settings.reinvent_python),
        "args": ["-m", "mcp_server.reinvent_server"],
        "cwd": str(PROJECT_ROOT),
        "env": {
            "REINVENT_PRIOR_DIR": str(settings.reinvent_prior_dir),
            "REINVENT_DEVICE": settings.reinvent_device,
            "PYTHONPATH": str(PROJECT_ROOT),
        },
    }


def _precedent_server_config(settings: Settings) -> dict:
    """Stdio config for the precedent server.

    The ORD backend runs in the project's Python and needs no data; the local backend runs in
    its own venv (rxnmapper, torch).
    """
    if settings.precedents_backend == "ord":
        return {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "mcp_server.precedent_server"],
            "cwd": str(PROJECT_ROOT),
            "env": {
                "PRECEDENTS_BACKEND": "ord",
                "ORD_API_URL": settings.ord_api_url,
                "ORD_SEARCH_TIMEOUT_S": str(settings.ord_search_timeout_s),
                "PYTHONPATH": str(PROJECT_ROOT),
            },
        }
    if settings.precedents_python is None or settings.precedents_data_dir is None:
        raise RuntimeError(
            "ENABLE_PRECEDENTS needs PRECEDENTS_PYTHON and PRECEDENTS_DATA_DIR (see README)."
        )
    return {
        "transport": "stdio",
        "command": str(settings.precedents_python),
        "args": ["-m", "mcp_server.precedent_server"],
        "cwd": str(PROJECT_ROOT),
        "env": {
            "PRECEDENTS_DATA_DIR": str(settings.precedents_data_dir),
            "PYTHONPATH": str(PROJECT_ROOT),
        },
    }


@asynccontextmanager
async def open_assistant(
    settings: Settings | None = None, *, persistent_sessions: bool = False
) -> AsyncIterator[ChemAssistant]:
    """Start the MCP server (stdio), build the agents and graph, and yield the assistant.

    One persistent MCP session per server is reused for every tool call (no process spawn per
    call). The retrosynthesis server is started only when ENABLE_RETROSYNTHESIS is set, and its
    tools share `retro_max_concurrent` slots. The REINVENT server is started only when
    ENABLE_REINVENT is set, the precedent server only when ENABLE_PRECEDENTS is set. Memory is
    in-process unless `persistent_sessions` is set (the API does): then it is a SQLite file with
    the limits of `minichem.sessions`.
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
    if settings.enable_reinvent:
        servers[REINVENT_SERVER_NAME] = _reinvent_server_config(settings)
    if settings.enable_precedents:
        servers[PRECEDENT_SERVER_NAME] = _precedent_server_config(settings)
    client = MultiServerMCPClient(servers)
    async with AsyncExitStack() as stack:
        mcp_tools = await load_mcp_tools(
            await stack.enter_async_context(client.session(MCP_SERVER_NAME))
        )
        retro_tools: list[BaseTool] = []
        if settings.enable_retrosynthesis:
            retro_session = await stack.enter_async_context(client.session(RETRO_SERVER_NAME))
            slots = asyncio.Semaphore(settings.retro_max_concurrent)
            retro_tools = [
                limit_concurrency(tool, slots, settings.retro_queue_wait_s)
                for tool in await load_mcp_tools(retro_session)
            ]
        reinvent_tools: list[BaseTool] = []
        if settings.enable_reinvent:
            reinvent_session = await stack.enter_async_context(client.session(REINVENT_SERVER_NAME))
            reinvent_tools = await load_mcp_tools(reinvent_session)
        precedent_tools: list[BaseTool] = []
        if settings.enable_precedents:
            precedent_session = await stack.enter_async_context(
                client.session(PRECEDENT_SERVER_NAME)
            )
            precedent_tools = await load_mcp_tools(precedent_session)
        checkpointer: MemorySaver | AsyncSqliteSaver = MemorySaver()
        sessions = None
        if persistent_sessions:
            settings.session_db_path.parent.mkdir(parents=True, exist_ok=True)
            checkpointer = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(settings.session_db_path))
            )
            sessions = SessionStore(
                checkpointer,
                SessionLimits(
                    ttl_s=settings.session_ttl_days * 86400,
                    max_sessions=settings.session_max_sessions,
                    max_turns=settings.session_max_turns,
                ),
            )
            await sessions.setup()
        agents: dict[str, SubAgent] = {
            "utility": build_utility_agent(llm, prompts, _build_lookup(settings)),
            "design": build_design_agent(llm, prompts, mcp_tools, reinvent_tools),
            "synthesis": build_synthesis_agent(llm, prompts, retro_tools + precedent_tools),
        }
        supervisor_llm = llm.with_structured_output(Decision).with_config(
            metadata={"agent": "supervisor"}
        )
        graph = build_graph(
            supervisor_llm=supervisor_llm,
            agents=agents,
            prompts=prompts,
            settings=settings,
            checkpointer=checkpointer,
        )
        yield ChemAssistant(graph, settings, sessions)
