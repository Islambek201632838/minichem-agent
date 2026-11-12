"""Handle-based tool arguments: a SMILES never passes through the model, in either direction.

The model writes `M3` where a tool wants a SMILES; `HandleMiddleware` swaps in the exact
registry string before the tool runs, and swaps registered molecules in the tool's result back
to handles before the model reads it. Molecules a tool returns for the first time get the next
free handle immediately, so the agent can chain calls (name -> properties) inside one run.
The tracer sits inside the tool call, so traces and the API keep real SMILES.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from minichem import molecules as mol

# Tool arguments that hold molecules. Everything else (names, counts, constraints) is left alone.
SMILES_ARGS = frozenset({"smiles", "smiles_list", "query", "candidates", "product", "reactants"})
_HANDLE_ONLY = re.compile(r"M(\d+)")
_QUOTES = " `'\""


def is_handle(value: str) -> bool:
    """True for text that is exactly a handle, ignoring quotes and spaces."""
    return _HANDLE_ONLY.fullmatch(value.strip(_QUOTES)) is not None


@dataclass
class RunHandles:
    """Handle table of one sub-agent run. It starts as the session registry and only grows."""

    registry: list[str]
    produced: list[str] = field(default_factory=list)  # molecules in tool results, in order

    def _resolve(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._resolve(item) for item in value]
        if isinstance(value, str) and (match := _HANDLE_ONLY.fullmatch(value.strip(_QUOTES))):
            index = int(match.group(1)) - 1
            if 0 <= index < len(self.registry):
                return self.registry[index]
        return value  # unknown handle or raw text: the tool reports it as an invalid SMILES

    def resolve(self, args: dict[str, Any]) -> dict[str, Any]:
        """Tool arguments with handles replaced by their exact SMILES."""
        return {k: self._resolve(v) if k in SMILES_ARGS else v for k, v in args.items()}

    def present(self, payload: dict[str, Any]) -> str:
        """JSON text for the model: new molecules registered, all registered ones as handles."""
        self.registry = mol.register(self.registry, mol.collect_smiles(payload))
        for smiles in mol.collect_smiles(payload):
            index = mol.find(self.registry, smiles)
            if index is not None and self.registry[index] not in self.produced:
                self.produced.append(self.registry[index])
        return json.dumps(mol.with_handles(payload, self.registry), separators=(",", ":"))


class HandleMiddleware(AgentMiddleware):
    """Wraps every tool call of a sub-agent; the run's `RunHandles` arrive as the context."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        handles: RunHandles = request.runtime.context
        call = {**request.tool_call, "args": handles.resolve(request.tool_call["args"])}
        result = await handler(request.override(tool_call=call))
        if not isinstance(result, ToolMessage) or (payload := mol.payload_of(result)) is None:
            return result
        return ToolMessage(
            content=handles.present(payload),
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            artifact=result.artifact,
        )
