"""Design agent: properties, similarity, analogues, filtering."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from minichem.agents.base import SubAgent, build_subagent
from minichem.tools.registry import design_local_tools


def build_design_agent(
    llm: BaseChatModel, prompts: dict[str, str], mcp_tools: list[BaseTool]
) -> SubAgent:
    """RDKit property/validation/similarity tools arrive through MCP; the rest are local."""
    return build_subagent("design", llm, [*mcp_tools, *design_local_tools()], prompts)
