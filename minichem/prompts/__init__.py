"""Prompt loading. Prompts are Markdown files next to this module, read once at startup."""

from __future__ import annotations

from pathlib import Path

PROMPT_NAMES = ("common", "supervisor", "utility", "design", "synthesis")
_DIR = Path(__file__).resolve().parent


def load_prompts(directory: Path = _DIR) -> dict[str, str]:
    """Read every required prompt file; fail loudly if one is missing or empty."""
    prompts: dict[str, str] = {}
    for name in PROMPT_NAMES:
        path = directory / f"{name}.md"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prompt file: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt file is empty: {path}")
        prompts[name] = text
    return prompts


def system_prompt(prompts: dict[str, str], agent: str) -> str:
    """Shared rules followed by the agent-specific instructions."""
    return f"{prompts['common']}\n\n{prompts[agent]}"
