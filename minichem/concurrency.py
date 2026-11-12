"""Cap on simultaneous heavy tool calls (retrosynthesis searches are CPU-bound, up to 120 s)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import StructuredTool


def busy_result(tool_name: str, wait_s: float) -> str:
    """The tool result returned when no slot frees up in time (tools never raise)."""
    return json.dumps(
        {
            "error": f"{tool_name} is busy: other searches are still running",
            "hint": f"No free slot after {wait_s:g} s. Tell the user to retry in a minute.",
        }
    )


def limit_concurrency(
    tool: StructuredTool, slots: asyncio.Semaphore, wait_s: float
) -> StructuredTool:
    """Copy of `tool` that runs only when a slot is free; waits up to `wait_s`, then reports busy.

    Parallel searches measurably slow each other down (about 188 s each for two at once against
    1 to 2 minutes alone), so waiting for a slot is cheaper than running them side by side.
    """
    if tool.coroutine is None:
        raise ValueError(f"tool {tool.name} has no coroutine to limit")
    inner = tool.coroutine

    async def limited(*args: Any, **kwargs: Any) -> Any:
        try:
            async with asyncio.timeout(wait_s):
                await slots.acquire()
        except TimeoutError:
            return busy_result(tool.name, wait_s), None
        try:
            return await inner(*args, **kwargs)
        finally:
            slots.release()

    return tool.model_copy(update={"coroutine": limited})
