"""Utility agent: name <-> SMILES and SMILES validation/canonicalisation."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from minichem.agents.base import SubAgent, build_subagent
from minichem.tools.pubchem import ChemLookup
from minichem.tools.registry import utility_tools


def build_utility_agent(
    llm: BaseChatModel, prompts: dict[str, str], lookup: ChemLookup
) -> SubAgent:
    """PubChem lookups (local tools) plus SMILES canonicalisation."""
    return build_subagent("utility", llm, utility_tools(lookup), prompts)
