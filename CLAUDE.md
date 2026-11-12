# MiniChemAgent

Small multi-agent chemistry assistant: a LangGraph supervisor routes requests to three agents
(utility, design, synthesis) that call RDKit/PubChem tools via native tool calling. Portfolio project
mirroring AstraZeneca's ChatInvent paper. Owner must be able to explain every decision in an interview,
so **readability and explainability beat features**. Keep it small; do not add auth, databases, Docker, CI.

Detailed docs (loaded automatically): `.claude/rules/`. User-facing docs: `README.md`.

## Commands (always `uv run`; the project venv is Python 3.12 managed by uv)
`/usr/bin/python3` is macOS's 3.9 and will not work. On the owner's Mac `python3` in `PATH` is now uv's 3.12
(`~/.local/bin`, installed with `uv python install 3.12 --default --preview`), but still use `uv run`.
```bash
uv sync                                          # install, versions pinned with == in pyproject.toml
uv run pytest -q                                 # 106 tests, no network, no API key, ~5 s
uv run ruff check . && uv run ruff format .      # must be clean before finishing
uv run python -m minichem.cli -v "question"      # real Gemini + MCP + PubChem, needs .env
uv run uvicorn app.api:app --port 8000           # API
uv run python -m streamlit run ui/app.py        # UI (API must be running; -m puts the project root on sys.path)
uv run python -m eval.run_eval [--runs N] [--ids t01,w03] [--mock-tools]   # real model, ~6 min for all
```

## Map
- `minichem/config.py` all settings (pydantic-settings). `state.py` graph state. `graph.py` LangGraph wiring.
- `minichem/agents/` `supervisor.py` (Decision, context), `base.py` (sub-agent build/run, retyping detector), one file per agent.
- `minichem/tools/` `chem.py` pure RDKit core, `models.py` Pydantic I/O, `analogues.py`, `synthesis.py`,
  `pubchem.py`, `retrosynthesis.py` (optional, off), `registry.py` (LangChain tool wrappers).
- `minichem/molecules.py` SMILES handles, masking, guard. `tracing.py` JSONL trace + token counts.
- `minichem/prompts/*.md` all prompts. `assistant.py` public entry (`open_assistant`, `ChemAssistant.ask`).
- `mcp_server/server.py` FastMCP stdio server (properties, validation, similarity), `mcp_server/retro_server.py` (retrosynthesis check + routes, runs in its own venv, off by default; steps are validated by `minichem/tools/route_check.py`). `app/api.py`, `ui/app.py`, `minichem/cli.py`.
- `eval/` `golden.yaml`, `checks.py`, `run_eval.py`, `results/` (+ `archive/`). `tests/` mirrors the code.

## Hard rules
- **Never print or commit `.env`** (holds the Gemini key; `GEMINI_API_KEY` is accepted as alias of `GOOGLE_API_KEY`). Only `.env.example` is committed.
- **All chemistry goes through tools.** The LLM must never compute or guess chemical facts. This rule lives in `prompts/common.md`.
- **SMILES are data.** The supervisor must never see or write SMILES; it uses handles (`M1`). See `.claude/rules/smiles-handles.md`.
- **Safety:** tools are read-only computation. No synthesis procedures, reagent amounts, or hazardous-substance advice. The synthesis agent reports computed scores/flags and, if retrosynthesis is on, template-derived disconnections (product <- reactants) without conditions; hazardous and controlled substances are declined without tools.
- **No invented APIs.** Check installed signatures before using a library (several things differed from memory, see gotchas).
- **Tools never raise into the agent loop**; they return `{"error","hint"}` or per-item `errors`.
- Prompts stay in `prompts/*.md`, config stays in `config.py`, no magic strings scattered, no bare `except`, no emojis in code/README, no commented-out code.
- Never fabricate eval numbers. README numbers must come from a JSON in `eval/results/`.

## Gotchas learned the hard way
- Gemini `AIMessage.content` is a list of blocks (with signatures). Use `message.text`, not `.content`.
- Use `langchain.agents.create_agent`, not the deprecated `langgraph.prebuilt.create_react_agent`. Pass `checkpointer=False` so the inner loop does not inherit the parent's checkpointer.
- PubChem property names are `SMILES`/`ConnectivitySMILES`; `CanonicalSMILES`/`IsomericSMILES` are silently ignored now. SMILES->name uses POST.
- MCP schema violations come back as a `ToolMessage` with `status="error"` and plain text, not as an exception. The tracer handles this.
- Keep ONE MCP session open (`client.session` + `load_mcp_tools`); `client.get_tools()` spawns a process per call.
- Trace agent names travel in run metadata: `graph.with_config(metadata={"agent": name})`. The supervisor runnable is configured the same way.
- Test fakes: each scripted tool call needs a unique id and a distinct message object. Reusing one object (`[msg]*10`) makes `add_messages` treat it as an update and breaks the agent loop (`KeyError: 'model'`).
- RDKit HBA/HBD definitions differ from PubChem (aspirin HBA = 3 vs 4). Golden values follow RDKit.
- Temperature is deliberately unset (Google advises against lowering it for Gemini 3).

## Definition of done for a change
`uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q` all clean; if agent behaviour or prompts changed, run at least `uv run python -m eval.run_eval --runs 1` and report real numbers; update README/CLAUDE.md if behaviour or numbers changed.
