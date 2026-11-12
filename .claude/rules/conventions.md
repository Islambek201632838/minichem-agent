# Conventions

- Python 3.11+, type hints everywhere, ruff (line length 100, rules E F I B UP SIM BLE C4 RET PT). Small functions, files under ~200 lines. No dead code.
- Comments only for the non-obvious why. Docstrings on public functions and tools. No emojis in code or README.
- Pin dependencies with `==` in `pyproject.toml`. Dev tools in `[dependency-groups].dev`. Do not add a dependency that is only pulled in transitively (e.g. tenacity was removed).
- Errors: explicit and useful, never silent. The only catch-all `except Exception` places are the tool boundary (`tools/registry._safe`) and the per-run guard in `eval/run_eval.py`, both with `noqa: BLE001` and a reason.
- Settings only via `minichem/config.py`; `load_dotenv(PROJECT_ROOT/".env")` runs on import.

## Adding a tool
1. Pure function + Pydantic result in `tools/chem.py` (or a sibling) with `smiles` as the key for each molecule.
2. Wrap in `tools/registry.py` (args schema, "when to use" description) or expose via `mcp_server/server.py`.
3. Mention it in the agent prompt (`prompts/<agent>.md`), add unit tests, and golden questions if it changes behaviour.

## Adding an agent
New module in `agents/` using `build_subagent`, add the name to `AGENT_NAMES` in `state.py`, a prompt file in `prompts/` and `PROMPT_NAMES`, wire it in `assistant.open_assistant`, update the supervisor prompt, extend `tests/test_agents.py`.

## Do not commit
`.env`, `logs/`, `.cache/` (see `.gitignore`). Remote: private repo Islambek201632838/minichem-agent (branch main). Commits use the GitHub noreply email; never put the token in the remote URL, pass it per command with `git -c http.extraheader`.
