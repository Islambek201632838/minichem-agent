# Reaction precedents (optional, off by default)

Owner's decision of 2026-11-12 (variant B): show reagents and solvents from a patent record, the
yield, patent number and year, labelled "from US patents, text-mined, not verified". Never
LLM-written conditions, amounts, temperatures or times (the data has none). This is the ONLY
place where the synthesis agent names solvents or reagents; `prompts/common.md` rule 4,
`prompts/synthesis.md`, `prompts/supervisor.md` and CLAUDE.md (Safety) state the exception.

## Choice (rule 2: check for a ready open solution first), checked 2026-11-12

| Candidate | Finding | Verdict |
|---|---|---|
| MolecularAI/precedent_finder (code) + Zenodo 15798247 (data) | Apache-2.0 code; data CC BY 4.0, 1 803 445 reactions, 661 MB; authors: example code, not maintained. Method = reaction-type filter, fingerprint similarity, Pareto over similarity, yield, year | USED as the method and the data. Not as a subprocess: its CLI loads the data on every call and its similarity loops over rows in Python; `precedent_rank.py` is a sparse-matrix rewrite with the same formulas, checked against the original (same similarities, same top results up to ties) |
| An open MCP server for reaction precedents | A web search for MCP servers found only USPTO patent-document servers (full text, metadata; some need an API key). None searches reactions | Nothing to reuse |
| Open Reaction Database (ord-data) | Code Apache-2.0, data CC BY-SA 4.0 (share-alike), Zenodo snapshot 1.1 GB, a few million reactions; has no ready similarity search, so we would build the index, fingerprints and ranking ourselves. Its USPTO part is the same Lowe extraction | Rejected as first source (more work, share-alike licence, no gain); possible second source later |
| Lowe's USPTO dataset directly | Same origin as the PrecedentFinder data, but unmapped and without the cleaning, fingerprints and reaction-type tags | Rejected: PrecedentFinder's data is that dataset after the work we would repeat |
| rxnmapper (atom mapping of the query step) | MIT, bundled model (3 MB), pip install; needs `setuptools<81` (it imports `pkg_resources`, removed in newer setuptools); same mapper that built the database, so the query is mapped the same way | USED. Rejected alternative: use AiZynthFinder's own atom map (not available inside the precedent server, and the agent cannot pass long mapped SMILES around) |
| `reaction_utils` (rxnutils) 1.9.4 | Apache-2.0, works on Python 3.12; gives `ChemicalReaction`, fingerprints, the SMARTS library for reaction types | USED |
| `paretoset` 1.2.5 | MIT, pip | USED (reuse before writing) |

`group_smarts.json` (the SMARTS library of reaction types) is not in the Zenodo record; it is
`data/group_smarts.json` of the precedent_finder repository (commit 025b91d, Apache-2.0) and is
copied into the data directory by hand.

## Layout

- `mcp_server/precedent_server.py`: third MCP server, one tool `find_reaction_precedents(product,
  reactants, max_precedents)`. Runs in its own venv (Python 3.12, `~/tools/precedents/.venv`)
  because torch (rxnmapper), pandas, scipy and numba would enter the project's lockfile for a
  feature that is off by default. Started by `assistant.open_assistant` like the retro server.
- `minichem/tools/precedents.py`: Pydantic models, `PROVENANCE` text, `MockPrecedents`, SMILES helpers
  (project venv). `precedent_search.py`: engine (precedent venv only). `precedent_rank.py`: ranking.
- Settings: `ENABLE_PRECEDENTS`, `PRECEDENTS_PYTHON`, `PRECEDENTS_DATA_DIR` (config.py).
- `handles.SMILES_ARGS` gained `product` and `reactants`, so the model passes handles. Patent
  molecules (reactants, reagents, product of each precedent) get handles when first seen, so the
  session registry grows by a few molecules per call; the final answer expands them to SMILES.
- Tool results never raise: `{"error","hint"}` for invalid SMILES, unreliable atom mapping (confidence <= 0.1, the threshold used to build the
  data), more than 5 reactants, no reaction of this type in the database, missing data.

## Facts about the data (read from the files, 2026-11-12)

Columns: ID (`US03931151;;27`: patent;;record), Year (1976-2016), ReactionSmiles (mapped, as
extracted), TextMinedYield, CalculatedYield, CuratedYield (0-100, always present), ReactionSmilesClean
(re-mapped with rxnmapper; reactants that give no atom to the product are moved to the middle
part), ReactiveFunction (reaction-type tag, 21 655 distinct values). The middle part holds the
reagents AND solvents together, plus work-up chemicals the extraction picked up (for example
ethyl acetate and sodium hydroxide in a lithium aluminium hydride reduction). It is not
possible to tell reagent from solvent from the data, so the tool does not try.
There is no temperature, time or amount anywhere in the data.

## Honest limits

- Text mining: yields and even reaction parts can be wrong. The similarity values are fingerprint
  heuristics, not probabilities. A "similar" precedent can be an analogue (a bromo-salicylic acid for
  aspirin, similarity 0.746); the precedent's own reactants and product are shown so this is visible.
- Coverage: US patents 1976-2016 only. Chemistry from papers and later patents is absent.
- No check that the step is chemically sound. An element check (product elements must occur in the
  reactants) was tried and removed: real steps list only the main reactant and leave water, acid
  or base implicit (hydrolysis of a nitrile to ibuprofen has no oxygen in its reactants). A
  nonsense step (benzene -> ethanol) is still mapped, with mapping confidence 0.481, and returns
  2796 candidates of the reaction type; the caller must pass real route steps, and the
  `mapping_confidence` field is returned so low values are visible.
- The query step comes from a retrosynthesis model; if the step is chemically odd (for ibuprofen the
  model proposed acetic acid plus a nitrile giving the acid), the precedents are for what rxnmapper
  makes of it.
- Memory: about 3.5 GB resident when loaded (measured), 8-10 s to load, searches 0.01-1.4 s.
