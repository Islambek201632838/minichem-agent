# Architecture

## Turn flow (minichem/graph.py)
`intake -> supervisor <-> {utility | design | synthesis} -> finalize`.
- `intake`: registers SMILES typed by the user (canonical form), resets per-turn state (`reports`, `warnings`, `steps`, `turn_molecules`, `answer`).
- `supervisor`: LLM with `with_structured_output(Decision)`; `Decision{next, task, final_answer}`. Unusable output (None, FINISH without answer, delegation with empty task or no steps left) goes to `finalize` with a fallback answer built from raw tool results. Budget: `MAX_SUPERVISOR_STEPS` (6).
- sub-agent node: `run_subagent` runs a `create_agent` loop, collects ToolMessages, registers SMILES from tool results, builds an `AgentReport` where molecules appear as ids, and detects SMILES the model passed to tools but was never given.
- `finalize`: expands handles to backticked exact SMILES and appends a warning for SMILES in the text that are not in the registry.
- Persistent state (MemorySaver, keyed by `session_id`): `messages`, `molecules`. Everything else is per turn. Per-request objects (the `Tracer`) travel in `RunContext` via `context=`, not in state.

## Agents and tools
- utility: `name_to_smiles`, `smiles_to_name`, `canonicalize_smiles` (local; PubChem client or `OfflinePubChem`).
- design: MCP tools `calculate_properties`, `validate_smiles`, `tanimoto_similarity` + local `generate_analogues`, `filter_molecules`.
- synthesis: `sa_score`, `structural_flags`, optional `retrosynthesis_check` (only if `ENABLE_RETROSYNTHESIS`).
- `tools/chem.py` is the single home of RDKit logic; the MCP server and LangChain wrappers are thin. Batch args (`smiles_list`, max 50) to save LLM calls.
- Tool outputs: every molecule record uses the key **`smiles`** (canonical). `molecules.collect_smiles` and `with_handles` depend on that key name.

## MCP
`mcp_server/server.py` (FastMCP, stdio) is started by `assistant.open_assistant` as `python -m mcp_server.server` with `cwd=PROJECT_ROOT`. A2A is a documented future direction only.

## Analogue generator (stand-in for REINVENT)
`tools/analogues.py`: 12 SMARTS point mutations (2 rounds) + BRICS recombination with 8 donor molecules (`scrambleReagents=False`, product cap 400), Tanimoto floor (default 0.4), constraints, dedupe, sort by (-similarity, smiles). Deterministic. Do not describe it as generative in docs.

## Retrosynthesis
`Retrosynthesis` protocol returns counts/score only. `MockRetrosynthesis` for tests. `AiZynthRetrosynthesis` uses only documented calls and reads stats with `.get`. Never auto-download models.

AiZynthFinder status (smoke-tested by hand on 2026-11-12, NOT part of the test suite, NOT wired into the agent):
- **It cannot be installed into the project venv.** aizynthfinder 4.4.1 needs `rdkit<2024` and `numpy<2`; the project pins `rdkit==2026.9.2`. Do not add it to `pyproject.toml`.
- It ran in a separate Python 3.12 venv. Extra pin needed there: `pyarrow==17.0.0` (latest pyarrow is built for NumPy 2 and fails to import with `numpy<2`).
- Models and stock come from `download_public_data <dir>` (about 770 MB, writes `config.yml` last, takes a few minutes). The adapter ran against that `config.yml` with the project's `AiZynthRetrosynthesis`, and the stats keys it reads (`is_solved`, `number_of_solved_routes`, `number_of_steps`, `top_score`) exist. Ethanol and aspirin solve in seconds, ibuprofen in about 20 s.
- Real integration needs a separate process with its own venv (for example a second MCP server), because of the rdkit conflict. Not built yet.

## LLM
`minichem/llm.py`: `ChatGoogleGenerativeAI(model=GEMINI_MODEL, max_retries=..., timeout=...)`. `max_retries` enables google-genai exponential backoff with jitter on 408/429/5xx. Default model `gemini-3.5-flash-lite` (stable, from aistudio.google.com/docs/models). API maps provider `APIError` to HTTP 503.

## Observability
`Tracer` (LangChain callback, `run_inline=True`) writes `logs/trace.jsonl`: kinds `tool`, `llm`, `step`; fields ts, session_id, agent, tool, arguments, duration_ms, error, tokens. Eval writes `logs/eval_trace.jsonl`. Langfuse intentionally not added.
