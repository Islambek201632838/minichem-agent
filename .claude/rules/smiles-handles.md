# Verbatim SMILES: the handle protocol (minichem/molecules.py)

Goal (paper rule): the LLM never retypes a SMILES. Implementation:
1. **Registry** `state["molecules"]`: canonical SMILES, session-persistent. Handle `Mk` = `registry[k-1]`. `register()` dedupes by canonical form and stores canonical text.
2. **Supervisor sees no SMILES.** `render_context` shows `Mk: <formula>`; `mask_smiles` hides registered molecules (any spelling of the same molecule) in the conversation history and in sub-agent notes. The supervisor writes `Mk` in tasks and answers.
3. **Code expands handles:** `expand_handles` before a sub-agent sees its task, and with backticks in the final answer.
4. **New SMILES enter the registry only from tool results** (`collect_smiles` over tool payloads).
5. **Guard:** `unverified_smiles` flags structural-looking valid SMILES in the final answer that are not in the registry (appended as a Warning). `_retyped_arguments` (agents/base.py) flags SMILES passed to tools that were not in the registry, the user question, or an earlier tool result of the same run; surfaced as `ChatResult.warnings` and a trace step.

Known limits (documented in README): inside a sub-agent the model still copies SMILES into tool args (detected, not prevented; eval saw 2 of 69 runs); the supervisor can still emit a correct SMILES from its own knowledge if it matches the registry. Proposed fix: handle-based tool arguments resolved in code.

Tokenizer notes: `_clean` strips prose punctuation and wrapping parentheses in a loop (bug history: a SMILES written as ``(`CC(C)...`)`` used to be missed by both detector and guard). User-SMILES detection accepts tokens with structural characters or all-caps atom strings (`CCO`); the guard requires structural characters to avoid flagging acronyms.
