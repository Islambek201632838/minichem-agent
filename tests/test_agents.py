"""Graph-level tests with scripted LLMs: routing, verbatim SMILES, budgets, memory, errors."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from minichem.agents.base import build_subagent
from minichem.assistant import ChemAssistant
from minichem.config import Settings
from minichem.graph import build_graph
from minichem.prompts import load_prompts
from minichem.tools import chem
from minichem.tools.pubchem import OfflinePubChem
from minichem.tools.registry import design_local_tools, synthesis_tools, utility_tools
from tests.conftest import IBUPROFEN
from tests.fakes import ScriptedChatModel, ScriptedSupervisor, finish, route, tool_call

LONG_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)NC1CCN(CC1)C(=O)c1ccccc1"  # not in canonical form
CANONICAL_LONG = chem.canonical(LONG_SMILES)


def mcp_stand_in() -> list[StructuredTool]:
    """Local twin of the MCP property tool, so these tests do not spawn a subprocess."""
    return [
        StructuredTool.from_function(
            func=lambda smiles_list: chem.calculate_properties(smiles_list).model_dump(),
            name="calculate_properties",
            description="Compute properties.",
        )
    ]


def make_assistant(tmp_path, supervisor: ScriptedSupervisor, **replies: list) -> ChemAssistant:
    """`replies` maps utility/design/synthesis to scripted model replies (none = must not run)."""
    checkpointer = replies.pop("checkpointer", None) or MemorySaver()
    sessions = replies.pop("sessions", None)
    prompts = load_prompts()
    settings = Settings(
        _env_file=None,
        trace_path=tmp_path / "trace.jsonl",
        max_supervisor_steps=replies.pop("max_steps", 6),
        max_agent_iterations=replies.pop("max_iterations", 12),
    )
    models = {
        n: ScriptedChatModel(replies=replies.get(n, [])) for n in ("utility", "design", "synthesis")
    }
    tools = {
        "utility": utility_tools(OfflinePubChem()),
        "design": mcp_stand_in() + design_local_tools(),
        "synthesis": synthesis_tools(),
    }
    agents = {n: build_subagent(n, models[n], tools[n], prompts) for n in models}
    graph = build_graph(
        supervisor_llm=supervisor.runnable,
        agents=agents,
        prompts=prompts,
        settings=settings,
        checkpointer=checkpointer,
    )
    assistant = ChemAssistant(graph, settings, sessions)
    assistant.models = models  # type: ignore[attr-defined]
    return assistant


def human_text(messages) -> str:
    return next(str(m.content) for m in messages if isinstance(m, HumanMessage))


async def test_name_lookup_routes_to_utility_only_and_answer_has_exact_smiles(tmp_path):
    supervisor = ScriptedSupervisor(
        [route("utility", "Resolve the name ibuprofen."), finish("Ibuprofen is M1.")]
    )
    assistant = make_assistant(
        tmp_path,
        supervisor,
        utility=[tool_call("name_to_smiles", name="ibuprofen"), AIMessage(content="Resolved.")],
    )
    result = await assistant.ask("What is the SMILES of ibuprofen?")

    assert result.answer == f"Ibuprofen is `{chem.canonical(IBUPROFEN)}`."
    assert [(c.agent, c.tool) for c in result.tool_calls] == [("utility", "name_to_smiles")]
    assert [m.id for m in result.molecules] == ["M1"]
    assert assistant.models["design"].cursor == 0
    assert assistant.models["synthesis"].cursor == 0


async def test_agent_works_with_handles_and_the_tool_receives_the_exact_smiles(tmp_path):
    def design_call(messages):
        assert "M1" in human_text(messages)
        assert CANONICAL_LONG not in human_text(messages)  # the agent is never given the SMILES
        return tool_call("calculate_properties", smiles_list=["M1"])

    supervisor = ScriptedSupervisor(
        [route("design", "Calculate the properties of M1."), finish("MW of M1 is in the report.")]
    )
    assistant = make_assistant(
        tmp_path, supervisor, design=[design_call, AIMessage(content="Done.")]
    )
    result = await assistant.ask(f"Give me properties of {LONG_SMILES}")

    # The tracer sits inside the tool call, so the trace and the API keep the real structure.
    assert result.tool_calls[0].arguments == {"smiles_list": [CANONICAL_LONG]}
    assert result.warnings == []
    context_block = str(supervisor.seen[0][-1].content)
    assert "M1: C" in context_block
    # Not even the user's own message: neither their spelling nor the canonical form.
    assert LONG_SMILES not in supervisor.shown_text
    assert CANONICAL_LONG not in supervisor.shown_text
    followup = str(supervisor.seen[1][-1].content)
    assert '"id":"M1"' in followup
    assert CANONICAL_LONG not in followup


async def test_tool_results_reach_the_model_as_handles_and_new_molecules_chain_in_one_run(
    tmp_path,
):
    expected = chem.canonical(IBUPROFEN)
    shown: list[str] = []

    def second_call(messages):
        shown.append(str(messages[-1].content))  # the name_to_smiles result, as the model sees it
        return tool_call("smiles_to_name", smiles="M1")

    supervisor = ScriptedSupervisor([route("utility", "Name ibuprofen."), finish("Done.")])
    assistant = make_assistant(
        tmp_path,
        supervisor,
        utility=[
            tool_call("name_to_smiles", name="ibuprofen"),
            second_call,
            lambda messages: shown.append(str(messages[-1].content)) or AIMessage(content="ok"),
        ],
    )
    result = await assistant.ask("What is ibuprofen?")

    assert '"id":"M1"' in shown[0]
    assert all(expected not in text for text in shown)
    assert result.warnings == []
    assert [c.tool for c in result.tool_calls] == ["name_to_smiles", "smiles_to_name"]
    assert result.tool_calls[1].arguments == {"smiles": expected}  # resolved before the call


async def test_unknown_handle_gives_a_tool_error_the_agent_can_read(tmp_path):
    supervisor = ScriptedSupervisor([route("design", "Properties of M9."), finish("Failed.")])
    assistant = make_assistant(
        tmp_path,
        supervisor,
        design=[
            tool_call("calculate_properties", smiles_list=["M9"]),
            AIMessage(content="M9 is invalid"),
        ],
    )
    result = await assistant.ask("Properties of M9")
    assert result.tool_calls[0].error is not None
    assert result.warnings == []  # a handle is never a retyped SMILES


async def test_multi_step_workflow_passes_tool_produced_smiles_between_agents(tmp_path):
    expected = chem.canonical(IBUPROFEN)

    def synthesis_call(messages):
        assert "M1" in human_text(messages)
        return tool_call("sa_score", smiles_list=["M1"])

    supervisor = ScriptedSupervisor(
        [
            route("utility", "Resolve ibuprofen."),
            route("synthesis", "Compute the SA score of M1."),
            finish("| Molecule | SA |\n|---|---|\n| M1 | see report |"),
        ]
    )
    assistant = make_assistant(
        tmp_path,
        supervisor,
        utility=[tool_call("name_to_smiles", name="ibuprofen"), AIMessage(content="ok")],
        synthesis=[synthesis_call, AIMessage(content="ok")],
    )
    result = await assistant.ask("How hard is ibuprofen to make?")

    assert [c.agent for c in result.tool_calls] == ["utility", "synthesis"]
    assert result.tool_calls[1].arguments == {"smiles_list": [expected]}
    assert result.warnings == []
    assert f"`{expected}`" in result.answer


async def test_retyped_smiles_argument_is_detected(tmp_path):
    altered = LONG_SMILES.replace("CCN", "CCC", 1)
    supervisor = ScriptedSupervisor([route("design", "Properties of M1."), finish("Done.")])
    assistant = make_assistant(
        tmp_path,
        supervisor,
        design=[tool_call("calculate_properties", smiles_list=[altered]), AIMessage(content="ok")],
    )
    result = await assistant.ask(f"Properties of {LONG_SMILES}")
    assert altered != LONG_SMILES
    assert any(altered in w for w in result.warnings)


async def test_correct_but_raw_smiles_argument_is_still_flagged_unless_the_user_typed_it(
    tmp_path,
):
    supervisor = ScriptedSupervisor(
        [route("design", "Properties of M1."), route("design", "Again."), finish("Done.")]
    )
    typed_by_user = "c1ccccc1O"
    assistant = make_assistant(
        tmp_path,
        supervisor,
        design=[
            tool_call("calculate_properties", smiles_list=[CANONICAL_LONG]),
            AIMessage(content="ok"),
            tool_call("calculate_properties", smiles_list=[typed_by_user]),
            AIMessage(content="ok"),
        ],
    )
    result = await assistant.ask(f"Properties of {LONG_SMILES} and {typed_by_user}")
    assert [w for w in result.warnings if CANONICAL_LONG in w]
    assert not [w for w in result.warnings if typed_by_user in w]


async def test_final_answer_with_an_invented_smiles_is_flagged(tmp_path):
    supervisor = ScriptedSupervisor([finish("It is CCCCCCN1CCCC1=O, trust me.")])
    result = await make_assistant(tmp_path, supervisor).ask("Hello")
    assert "Warning" in result.answer
    assert "CCCCCCN1CCCC1=O" in result.answer


async def test_greeting_finishes_without_calling_any_agent(tmp_path):
    supervisor = ScriptedSupervisor([finish("Hi! Ask me about molecules.")])
    assistant = make_assistant(tmp_path, supervisor)
    result = await assistant.ask("Hello")
    assert result.tool_calls == []
    assert all(m.cursor == 0 for m in assistant.models.values())


async def test_step_budget_forces_a_finish(tmp_path):
    def one_run():
        return [tool_call("name_to_smiles", name="aspirin"), AIMessage(content="ok")]

    supervisor = ScriptedSupervisor([route("utility", "again") for _ in range(3)])
    assistant = make_assistant(tmp_path, supervisor, utility=one_run() + one_run(), max_steps=2)
    result = await assistant.ask("Loop forever")

    assert len(result.tool_calls) == 2
    assert "raw tool results" in result.answer
    assert "MUST choose FINISH" in str(supervisor.seen[-1][-1].content)


async def test_unusable_supervisor_output_gives_a_clear_fallback(tmp_path):
    result = await make_assistant(tmp_path, ScriptedSupervisor([None])).ask("???")
    assert "could not work out" in result.answer


async def test_delegating_with_an_empty_task_is_treated_as_unusable(tmp_path):
    supervisor = ScriptedSupervisor([route("design", "  ")])
    result = await make_assistant(tmp_path, supervisor).ask("Do something")
    assert "could not work out" in result.answer


async def test_session_memory_keeps_molecules_and_history_per_session(tmp_path):
    def synthesis_call(messages):
        assert "M1" in human_text(messages)
        return tool_call("sa_score", smiles_list=["M1"])

    supervisor = ScriptedSupervisor(
        [
            route("utility", "Resolve ibuprofen."),
            finish("Done: M1."),
            route("synthesis", "SA score of M1."),
            finish("Its SA score is in the report."),
            finish("I do not know what 'it' is."),
        ]
    )
    assistant = make_assistant(
        tmp_path,
        supervisor,
        utility=[tool_call("name_to_smiles", name="ibuprofen"), AIMessage(content="ok")],
        synthesis=[synthesis_call, AIMessage(content="ok")],
    )
    await assistant.ask("SMILES of ibuprofen?", session_id="a")
    await assistant.ask("What is its SA score?", session_id="a")
    second_turn = supervisor.seen[2]
    assert any("SMILES of ibuprofen?" in str(m.content) for m in second_turn)
    assert "M1: C13H18O2" in str(second_turn[-1].content)

    await assistant.ask("What is its SA score?", session_id="b")
    assert "(none yet)" in str(supervisor.seen[4][-1].content)


async def test_tool_error_is_visible_to_the_agent_which_can_correct_itself(tmp_path):
    supervisor = ScriptedSupervisor([route("utility", "Resolve aspirin."), finish("Done.")])
    assistant = make_assistant(
        tmp_path,
        supervisor,
        utility=[
            tool_call("name_to_smiles", name="asprin"),
            tool_call("name_to_smiles", name="aspirin"),
            AIMessage(content="ok"),
        ],
    )
    result = await assistant.ask("Resolve aspirin")
    assert [bool(c.error) for c in result.tool_calls] == [True, False]
    assert result.molecules[0].smiles == chem.canonical("CC(=O)Oc1ccccc1C(=O)O")


async def test_agent_that_never_stops_is_cut_off_and_reported(tmp_path):
    spin = [tool_call("name_to_smiles", name="aspirin") for _ in range(10)]
    supervisor = ScriptedSupervisor([route("utility", "Resolve aspirin."), finish("Gave up.")])
    assistant = make_assistant(tmp_path, supervisor, utility=spin, max_iterations=4)
    await assistant.ask("Resolve aspirin")
    assert "iteration limit" in str(supervisor.seen[1][-1].content)


async def test_every_step_and_tool_call_is_written_to_the_trace(tmp_path):
    supervisor = ScriptedSupervisor([route("utility", "Resolve aspirin."), finish("Done.")])
    assistant = make_assistant(
        tmp_path,
        supervisor,
        utility=[tool_call("name_to_smiles", name="aspirin"), AIMessage(content="ok")],
    )
    await assistant.ask("Aspirin?", session_id="trace-test", request_id="req-00000001")
    lines = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    kinds = {(line["kind"], line["agent"]) for line in lines}
    assert ("tool", "utility") in kinds
    assert ("step", "supervisor") in kinds
    assert ("llm", "utility") in kinds
    assert all(line["session_id"] == "trace-test" for line in lines)
    assert all(line["request_id"] == "req-00000001" for line in lines)


def test_graph_refuses_to_build_without_all_agents(tmp_path):
    with pytest.raises(ValueError, match="Missing sub-agents"):
        build_graph(
            supervisor_llm=ScriptedSupervisor([]).runnable,
            agents={},
            prompts=load_prompts(),
            settings=Settings(_env_file=None),
            checkpointer=None,
        )
