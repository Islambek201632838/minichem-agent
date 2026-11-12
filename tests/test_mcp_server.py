"""Starts the real MCP server over stdio and calls each tool through langchain-mcp-adapters."""

import sys
from contextlib import asynccontextmanager

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from minichem.config import PROJECT_ROOT
from minichem.molecules import payload_of
from tests.conftest import ASPIRIN, ETHANOL


@asynccontextmanager
async def mcp_tools():
    client = MultiServerMCPClient(
        {
            "rdkit": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "cwd": str(PROJECT_ROOT),
            }
        }
    )
    async with client.session("rdkit") as session:
        yield {t.name: t for t in await load_mcp_tools(session)}


async def call(tool, **args):
    message = await tool.ainvoke({"type": "tool_call", "id": "1", "name": tool.name, "args": args})
    return payload_of(message)


async def test_server_exposes_the_three_rdkit_tools():
    async with mcp_tools() as tools:
        assert set(tools) == {"calculate_properties", "validate_smiles", "tanimoto_similarity"}
        assert all(t.description for t in tools.values())


async def test_tools_over_mcp_match_direct_calls():
    async with mcp_tools() as tools:
        props = await call(tools["calculate_properties"], smiles_list=[ASPIRIN, "bad"])
        assert props["results"][0]["mw"] == pytest.approx(180.16, abs=0.01)
        assert props["errors"][0]["input"] == "bad"

        valid = await call(tools["validate_smiles"], smiles_list=["OCC"])
        assert valid["results"][0]["smiles"] == ETHANOL

        sim = await call(tools["tanimoto_similarity"], query=ETHANOL, candidates=[ETHANOL])
        assert sim["results"][0]["tanimoto"] == 1.0
        bad = await call(tools["tanimoto_similarity"], query="bad", candidates=[ETHANOL])
        assert "not a valid SMILES" in bad["error"]


async def test_schema_violation_comes_back_as_a_readable_error_message():
    async with mcp_tools() as tools:
        message = await tools["calculate_properties"].ainvoke(
            {
                "type": "tool_call",
                "id": "1",
                "name": "calculate_properties",
                "args": {"smiles_list": []},
            }
        )
        assert message.status == "error"
        assert "at least 1 item" in message.content[0]["text"]
