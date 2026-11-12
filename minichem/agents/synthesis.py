"""Synthesis agent: SA score and structural flags (computed values only)."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from minichem.agents.base import SubAgent, build_subagent
from minichem.tools.registry import synthesis_tools


def build_synthesis_agent(
    llm: BaseChatModel, prompts: dict[str, str], extra_tools: Sequence[BaseTool] = ()
) -> SubAgent:
    """`extra_tools` holds `retrosynthesis_check` when retrosynthesis was explicitly enabled."""
    return build_subagent("synthesis", llm, [*synthesis_tools(), *extra_tools], prompts)
