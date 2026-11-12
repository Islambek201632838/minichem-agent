"""Synthesis agent: SA score and structural flags (computed values only)."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from minichem.agents.base import SubAgent, build_subagent
from minichem.tools.registry import synthesis_tools
from minichem.tools.retrosynthesis import Retrosynthesis


def build_synthesis_agent(
    llm: BaseChatModel, prompts: dict[str, str], retro: Retrosynthesis | None = None
) -> SubAgent:
    """`retro` is None unless retrosynthesis was explicitly enabled."""
    return build_subagent("synthesis", llm, synthesis_tools(retro), prompts)
