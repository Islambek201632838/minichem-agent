# Testing and evaluation

## Unit/agent tests (`tests/`, no network, no key)
- `test_chem_tools`, `test_analogues`, `test_synthesis`: known molecules (ethanol, aspirin, ibuprofen), bad SMILES, similarity bounds, determinism.
- `test_pubchem`: fake `requests` session (success, 404, timeout, 503, malformed, cache hit across instances, corrupt cache, POST for SMILES).
- `test_mcp_server`: starts the REAL stdio server; use an `asynccontextmanager` helper inside each test (async generator fixtures break anyio cancel scopes).
- `test_agents` + `fakes.py`: `ScriptedChatModel` (bind_tools returns self; replies may be callables of the messages) and `ScriptedSupervisor`. Cover routing, verbatim SMILES, supervisor never sees SMILES, retyped-argument detection, step budget, fallback, memory per session, tool-error recovery, iteration limit, trace file.
- `test_api`: stubs `open_assistant`; checks validation (422) and 503 mapping. `test_eval_checks`: matchers, checks, golden file shape.

## Golden suite (`eval/golden.yaml`, 24 questions: 13 tool_specific `t*`, 11 workflow `w*`)
- `expected_tools.complete/partial`: accepted tool-name sequences, matched as ORDERED SUBSEQUENCES after collapsing immediate repeats. `[]` means no tools. Extra calls are tolerated.
- `checks` (LLM-free, `eval/checks.py`): `smiles_present` (canonical compare), `numbers` (tol), `regex`, `regex_any`, `regex_none`, `no_smiles`, `smiles_property` (recomputes properties with RDKit; counts table rows if the answer has a table, else all SMILES).
- Outcome: complete / partial / incorrect. A failed answer check is `incorrect` even if tools were right.
- Expected values must be computed independently with RDKit, never copied from model output. Names used must exist in `OFFLINE_COMPOUNDS` so `--mock-tools` works.
- `--mock-tools` switches PubChem to `OfflinePubChem`; MCP is still real. Latency was ~equal to live: the LLM dominates.

## Rules for changing the eval
- A full run is ~6 min and ~420k tokens on the free tier. Prefer `--ids`/`--runs 1` while iterating.
- Changing a check or `min_count` after seeing results weakens the evidence. If you must, say so in the README and keep earlier result files in `eval/results/archive/` (the current README describes three runs and two defects fixed in checks).
- README tables must be copied from a JSON in `eval/results/`; update them whenever code, prompts, or the golden set change behaviour.
- Result files: `YYYYMMDDTHHMMSSZ[-mock-tools].json` with summary, by_category, per_question, per-run records (tools, answer, warnings, tokens).
