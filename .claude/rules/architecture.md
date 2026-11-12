# Architecture

## Turn flow (minichem/graph.py)
`intake -> supervisor <-> {utility | design | synthesis} -> finalize`.
- `intake`: registers SMILES typed by the user (canonical form), resets per-turn state (`reports`, `warnings`, `steps`, `turn_molecules`, `answer`).
- `supervisor`: LLM with `with_structured_output(Decision)`; `Decision{next, task, final_answer}`. Unusable output (None, FINISH without answer, delegation with empty task or no steps left) goes to `finalize` with a fallback answer built from raw tool results. Budget: `MAX_SUPERVISOR_STEPS` (6).
- sub-agent node: `run_subagent` runs a `create_agent` loop with `HandleMiddleware` (`handles.py`): handles in tool arguments are resolved to SMILES before the tool runs and tool results are shown to the model with handles; the node reads the grown handle table (`RunHandles`) back as the new registry, builds an `AgentReport` where molecules appear as ids, and flags SMILES the model wrote into arguments instead of handles.
- `finalize`: expands handles to backticked exact SMILES and appends a warning for SMILES in the text that are not in the registry.
- Persistent state (checkpointer keyed by `session_id`: SQLite via `minichem/sessions.py` in the API, `MemorySaver` in the CLI and eval): `messages`, `molecules`. Everything else is per turn. Per-request objects (the `Tracer`) travel in `RunContext` via `context=`, not in state.

## Agents and tools
- utility: `name_to_smiles`, `smiles_to_name`, `canonicalize_smiles` (local; PubChem client or `OfflinePubChem`).
- design: MCP tools `calculate_properties`, `validate_smiles`, `tanimoto_similarity` + local `generate_analogues`, `filter_molecules`.
- synthesis: `sa_score`, `structural_flags`, optional `retrosynthesis_check` (only if `ENABLE_RETROSYNTHESIS`).
- `tools/chem.py` is the single home of RDKit logic; the MCP server and LangChain wrappers are thin. Batch args (`smiles_list`, max 50) to save LLM calls.
- Tool outputs: every molecule record uses the key **`smiles`** (canonical). `molecules.collect_smiles` and `with_handles` depend on that key name.

## MCP
`mcp_server/server.py` (FastMCP, stdio) is started by `assistant.open_assistant` as `python -m mcp_server.server` with `cwd=PROJECT_ROOT`. A second server (`retro_server`, see Retrosynthesis) is started only when retrosynthesis is enabled. A third server (`precedent_server`, see `.claude/rules/precedents.md`) is started only when `ENABLE_PRECEDENTS` is set. Its tool `find_reaction_precedents` is given to the synthesis agent as an `extra_tools` entry, like the retrosynthesis tools. A2A is a documented future direction only.

## Analogue generator (stand-in for REINVENT)
`tools/analogues.py`: 12 SMARTS point mutations (2 rounds) + BRICS recombination with 8 donor molecules (`scrambleReagents=False`, product cap 400), Tanimoto floor (default 0.4), constraints, dedupe, sort by (-similarity, smiles). Deterministic. Do not describe it as generative in docs.

## REINVENT 4 Mol2Mol (optional, off by default: `ENABLE_REINVENT`)
`tools/mol2mol.py` (pure selection logic in RDKit, tested; `ReinventMol2Mol` imports torch and REINVENT lazily) and `mcp_server/reinvent_server.py` (tool `mol2mol_analogues`: smiles, n <= 30, min_similarity 0.4, seed 0, constraints). It runs as its own MCP server in `~/tools/reinvent/.venv` (`REINVENT_PYTHON`, `REINVENT_PRIOR_DIR`, `REINVENT_DEVICE` mps|cpu, passed as env), its tool is given to the design agent as `extra_tools`, and `generate_analogues` stays as the fallback (tool errors name it in `hint`; `prompts/design.md` says to fall back and to say so).
- Draws `max(100, 4n)` multinomial samples (temperature 1), keeps valid, distinct, non-query molecules, recomputes Tanimoto with the project fingerprint, applies floor and constraints, ranks by (-tanimoto, smiles). The seed is returned in every result. Calls REINVENT's own `create_adapter`, `setup_sampler`, `set_seed`; no files written.
- **Why custom, not a ready MCP server (rule 2, checked 2026-11-12):** `lcmd-epfl/Solitarius-mcp` (Apache-2.0, 0 stars, created 2026-08-09) takes file paths (the LLM cannot write them, handles do not resolve them), has no seed or similarity floor, spawns `reinvent` per call, returns CSV summaries not `smiles` records, and also exposes plugin-writing and training tools the design agent must not get. The lobehub "REINVENT4 LangGraph MCP Server" points to a repo that returns 404. MolMIM MCP needs an enterprise subscription.
- **Why a separate venv:** torch 2.13.0, pandas 3, polars, mmpdb; the project rdkit and pydantic pins match REINVENT's, but the extra weight does not belong in the lockfile of an off-by-default feature.
- Install: `uv venv --python 3.12 --seed`, then `python install.py mac -d none` from the clone (the default `all` would pull OpenEye, closed source), then `mcp==1.30.0` and `scipy` by hand. Priors are downloaded by hand from Zenodo 15641297 (md5 checked), never automatically.
- Measured values (time, memory, reproducibility) are in README. Not done: staged learning (RL), beam search (not tried). Rejected: a per-call `reinvent` subprocess (reloads the model, file I/O).

## Retrosynthesis
Off by default (`ENABLE_RETROSYNTHESIS`). `Retrosynthesis` protocol has `analyse` (counts/score); `MockRetrosynthesis` for tests; `AiZynthRetrosynthesis` adds `routes` (steps from `ReactionTree.to_dict()`), uses documented calls and reads stats with `.get` (search: 120 s, 300 iterations, 12 transforms). `routes` returns solved routes first, then partial ones (`all_in_stock` false, `unresolved` = non-purchasable leaves, `n_steps`); routes with no steps are dropped. `Route` puts `n_steps` and `unresolved` BEFORE `steps` because the supervisor sees tool results cut to 1500 chars. The full routes reach the API as `ChatResult.routes` (from `Tracer.results`, in memory only) and `ui/route_view.py` draws them. `tools/reaction_types.py` names a step (16 textbook reactions as forward SMARTS plus a class-level reagent hint) only if the template reproduces the step's product; unmatched steps stay unnamed. Names are labels from a table, never LLM-written.
- **It cannot be installed into the project venv.** aizynthfinder 4.4.1 needs `rdkit<2024` and `numpy<2`; the project pins `rdkit==2026.9.2`. Do not add it to `pyproject.toml`.
- It runs as a SECOND MCP server, `mcp_server/retro_server.py` (tools `retrosynthesis_check`, max 5 SMILES, and `retrosynthesis_routes`, one molecule, up to 3 routes; steps are `product <- reactants` with in_stock, policy_probability, template_support and forward_check from `tools/route_check.py`; solved routes first, else partial routes with `unresolved` molecules; no conditions exist in the data), started by `assistant.open_assistant` with `AIZYNTH_PYTHON` (the Python of the separate venv, `cwd=PROJECT_ROOT`, `PYTHONPATH=PROJECT_ROOT`, `AIZYNTH_CONFIG` passed as env). Its tools are given to the synthesis agent as `extra_tools`. The server loads the models lazily on the first call.
- Separate venv (3.12) pins: `aizynthfinder==4.4.1`, `pyarrow==17.0.0` (latest pyarrow is built for NumPy 2), `mcp==1.30.0` (same as the project; an unpinned install gets mcp 2.x, which has no `mcp.server.fastmcp`). Models and stock: `download_public_data <dir>` (about 770 MB, writes `config.yml` last). Never auto-download.
- The downloader does not verify sizes: a run stopped silently at 39% of `uspto_model.onnx` (symptom: `INVALID_PROTOBUF` at the first search). Compare sizes with the Zenodo records or resume with `curl -L -C -`.
- Verified by hand on 2026-11-12 (real Gemini + real models), NOT in the test suite and not in the golden set: aspirin and ethanol solved in a few seconds, ibuprofen solved (1 route, 4 steps, top score 0.975) in about 30 s including the LLM. The supervisor prompt must mention routes, otherwise it never delegates them to synthesis.

## LLM
`minichem/llm.py`: `ChatGoogleGenerativeAI(model=GEMINI_MODEL, max_retries=..., timeout=...)`. `max_retries` enables google-genai exponential backoff with jitter on 408/429/5xx. Default model `gemini-3.5-flash-lite` (stable, from aistudio.google.com/docs/models). API maps provider `APIError` to HTTP 503.

## API layer (app/)
`api.py`: middleware gives every request an id (`X-Request-ID`), writes one line to `logs/api.jsonl` and feeds `metrics.py`; `/chat` passes through `security.py` (key, rate limit), then the session lock, the cap on running turns and `asyncio.timeout`. `open_assistant(persistent_sessions=True)` opens the SQLite checkpointer and a `SessionStore` (activity, TTL, session and turn caps, compaction); retrosynthesis tools are wrapped by `concurrency.limit_concurrency`. Startup validation is `Settings.check_startup()`. Details and rejected alternatives: `production-roadmap.md`.

## Observability
`Tracer` (LangChain callback, `run_inline=True`) writes `logs/trace.jsonl` through `JsonlLog` (rotating): kinds `tool`, `llm`, `step`; fields ts, request_id, session_id, agent, tool, arguments, duration_ms, error, tokens. Eval writes `logs/eval_trace.jsonl`. Langfuse intentionally not added.
