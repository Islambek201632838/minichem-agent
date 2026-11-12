# Comparison with AstraZeneca's open code (langdmta-lab)

Source: https://github.com/MolecularAI/langdmta-lab (Apache-2.0), commit `3b7742b` of 2026-10-06,
cloned shallow into a scratch folder outside this project. I READ the code; nothing from it was run
or copied. Paper facts come from the handoff notes, not re-checked here.

## What it is

Supervisor + four agents (Design, Synthesis, Analyzer, Utility) built from YAML: `graph.yaml` lists
nodes and edges, each agent has a YAML with its prompt and a `prompt_to_supervisor` text that is
injected into the supervisor prompt automatically from the graph edges. Every agent talks to its own
MCP server over streamable HTTP (one process per server and port). GPT-4o through Azure or OpenAI.
Linux and conda only. Design wraps REINVENT 4, Synthesis wraps AiZynthFinder and PrecedentFinder.

## Where theirs is better

| Area | Theirs | Ours |
|---|---|---|
| Design tools | Real REINVENT 4: `mol2mol_medium_similarity` beam-search sampling, REINVENT scoring components, staged reinforcement learning with the property window turned into a double-sigmoid transform (`rl_helper.py`) | Deterministic SMARTS and BRICS stand-in |
| Reaction precedents | `precedent_finder` tool (atom-maps the reaction with rxnmapper, then searches USPTO) | None |
| Evaluation | Question generator (templates, 10 phrasing levels, entities sampled from CSV tables), Langfuse traces, DSPy LLM judge with 6 metrics, human annotation web app, Cohen and Fleiss kappa, judge optimisation (MIPROv2), annotated train/validation split | 24 hand-written questions with exact checks, no judge |
| Config-driven graph | Adding an agent is a YAML plus a small class | Python wiring in `assistant.py` and `state.py` |
| Retries | Per-node `RetryPolicy` that separates retryable from non-retryable errors; backoff on rate limits | Client-level retries in the Gemini client |
| Tool-call testing | `mock_if_available` decorator with recorded inputs and outputs per tool | Fakes and offline fixtures (different, not worse) |
| Scoring semantics | Prompt text that explains REINVENT quirks (for example `MatchingSubstructure` returns 1.0 or 0.5, never 0) | Not needed until REINVENT is added |

## Where ours is better

| Area | Why |
|---|---|
| SMILES handling | Theirs only tells the supervisor in a prompt to "use the WHOLE, EXACT SMILES string". Ours has the handle registry, masking and a guard that flags retyped SMILES. This is the paper's own rule, enforced in code |
| Data between agents | Theirs passes CSV file paths that the supervisor must copy character by character. Ours passes typed results and handles |
| Tool process model | Theirs starts `conda run reinvent ...` or `aizynthcli ...` as a new process for every call and reloads models each time. Ours keeps one MCP session and loads models once on first use |
| Retrosynthesis depth | Theirs takes only the first tree (`trees[0]`) and the in-stock list. Ours returns several routes, a forward check of every step, partial routes with unresolved molecules, textbook reaction names from a fixed table, and drawings |
| Safety in the agent | A search of their `langdmta_lab` code and prompts finds no refusal rule for hazardous or controlled substances. Ours has a prompt rule and a golden question `t13` (still not a hard filter). They DO test it: `tests/test_questions_edge_cases.toml` has an "ethical boundary" set (VX nerve agent, methamphetamine, MDMA analogues) and the judge has an "ethical awareness" metric. Worth copying as a test idea |
| Analyzer | Theirs lets an LLM write and run pandas code. Ours never executes model-written code |
| Reproducibility | Ours pins every dependency with `==`, runs on macOS, and has 126 offline tests. Theirs declares `>=` ranges (a `uv.lock` exists), supports Linux only, and its question runner starts the MCP servers and needs Langfuse and OpenAI keys |
| Errors | Ours returns `{"error","hint"}` values from tools; theirs raises, uses broad `except Exception`, and an extra LLM call strips file paths from answers (`_strip_file_path`) |

## What is worth taking, and how

Nothing has been copied. The list below is what I propose; each item needs the owner's yes first.
If anything is copied, the file keeps its Apache-2.0 header and the repo gets a `NOTICE` file naming
the source and commit.

| Candidate | Action | Reason |
|---|---|---|
| REINVENT TOML templates and the transform constants (`rl_helper.py`, `scoring_*_template.toml`, `utils/transform.py`) | Adapt (rewrite around our models), credit in `NOTICE` | Saves guessing REINVENT's config format; the sigmoid parameters are theirs and tested on their data |
| PrecedentFinder idea | Re-implement the call pattern, not the code | Their tool calls it as a subprocess per reaction; we need a persistent server |
| Ethical-boundary question set (edge cases) | Take the idea: write our own questions, expect a refusal without tools | We have one such question (`t13`); they have a whole category |
| Question generator and phrasing levels | Take the idea, write our own on our golden set | Our suite is small and easy; phrasing variety is the cheapest way to make it harder |
| LLM judge with human alignment | Do later, only after a larger question set exists | Needs human annotation to be trustworthy; without it a judge is noise |
| YAML-defined graph | Do not take | It would be a rewrite with no behaviour gain at three agents |
| Memory retriever (LLM summarises the history every turn) | Do not take as is | Costs one extra LLM call per turn; summarise only when the history is long |
| Analyzer agent (LLM-written pandas) | Do not take | Executes model-written code; a read-only, typed table tool is safer |

## What was not verified

Their pipeline was not run (Linux only, needs Azure or OpenAI keys, Langfuse and REINVENT).
Claims about how it behaves come from reading the code. The unpublished proprietary scoring and
internal databases are, per the handoff, not in the open version; I did not look for them.
