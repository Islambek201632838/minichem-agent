import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from minichem.prompts import PROMPT_NAMES, load_prompts, system_prompt
from minichem.tracing import Tracer, error_of


def test_all_prompts_load_and_every_agent_gets_the_shared_rules():
    prompts = load_prompts()
    assert set(prompts) == set(PROMPT_NAMES)
    for agent in ("utility", "design", "synthesis"):
        text = system_prompt(prompts, agent)
        assert "MUST go through tools" in text
        assert "synthesis procedures" in text


def test_missing_prompt_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path)


def test_empty_prompt_file_fails_loudly(tmp_path):
    for name in PROMPT_NAMES:
        (tmp_path / f"{name}.md").write_text("x")
    (tmp_path / "design.md").write_text("  \n")
    with pytest.raises(ValueError, match="empty"):
        load_prompts(tmp_path)


def test_error_of_detects_whole_call_and_per_item_errors():
    assert error_of({"error": "boom"}) == "boom"
    assert error_of({"results": [], "errors": [{"input": "x", "error": "bad"}]}) == "bad"
    assert error_of({"results": [1], "errors": []}) is None


def test_tracer_records_tool_calls_and_writes_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path, session_id="s1")
    run_id = uuid4()
    tracer.on_tool_start(
        {"name": "calculate_properties"},
        "",
        run_id=run_id,
        metadata={"agent": "design"},
        inputs={"smiles_list": ["CCO"]},
    )
    tracer.on_tool_end(
        ToolMessage(content='{"errors": [{"error": "bad"}]}', tool_call_id="1"), run_id=run_id
    )
    tracer.step("supervisor", "route", next="design")

    call = tracer.tool_calls[0]
    assert (call.agent, call.tool, call.error) == ("design", "calculate_properties", "bad")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["tool", "step"]
    assert set(lines[0]) >= {"ts", "agent", "tool", "arguments", "duration_ms", "error"}


def test_tracer_marks_mcp_error_messages_as_errors():
    tracer = Tracer(None)
    run_id = uuid4()
    tracer.on_tool_start({"name": "t"}, "", run_id=run_id, metadata={"agent": "design"}, inputs={})
    message = ToolMessage(
        content="Error executing tool t: bad args", tool_call_id="1", status="error"
    )
    tracer.on_tool_end(message, run_id=run_id)
    assert "bad args" in tracer.tool_calls[0].error


def test_tracer_counts_tokens(tmp_path):
    tracer = Tracer(None)
    run_id = uuid4()
    tracer.on_chat_model_start({}, [[]], run_id=run_id, metadata={"agent": "supervisor"})
    message = AIMessage(
        content="x", usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    )
    tracer.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)
    assert (tracer.llm_calls, tracer.input_tokens, tracer.output_tokens) == (1, 10, 4)


def test_tracer_keeps_full_results_in_memory_but_not_in_the_trace_file(tmp_path):
    path = tmp_path / "t.jsonl"
    tracer = Tracer(path)
    run_id = uuid4()
    tracer.on_tool_start({"name": "big"}, "{}", run_id=run_id, metadata={"agent": "a"}, inputs={})
    payload = {"rows": ["x" * 40] * 50}
    tracer.on_tool_end(json.dumps(payload), run_id=run_id)
    assert tracer.results == [("big", payload)]
    assert len(path.read_text()) < len(json.dumps(payload))
