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
| Open Reaction Database (ord-data) | Code Apache-2.0, data CC BY-SA 4.0 (share-alike), Zenodo snapshot 1.1 GB, a few million reactions; has no ready similarity search, so we would build the index, fingerprints and ranking ourselves. Its USPTO part is the same Lowe extraction | Rejected as first source (more work, share-alike licence, no gain); possible second source later. Evaluated on 2026-11-12, see the last section: not recommended for now |
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

## Evaluation of the Open Reaction Database as a second source (2026-11-12, owner: check, do not connect)

Nothing was added to the project. Method: GitHub and Hugging Face APIs, parquet footers read with HTTP
range requests (names and row counts of all 94 files, nothing else downloaded for that), plus two
small samples downloaded into a scratchpad (`uspto-grants-1976`, `uspto-grants-1990`, 23.8 MB, and the
750-reaction AstraZeneca ELN set, 0.3 MB), parsed with `ord-schema` and compared with the local
PrecedentFinder table. The 1976 and 1990 numbers are measured; everything about other years is not.

**What `ord-data` is now.** Git LFS repository (the LFS objects are served from the Hugging Face mirror
`open-reaction-database/ord-data`, which is why the README no longer mentions the Zenodo snapshot of
1.1 GB written above): 94 parquet files, 1.29 GB, 2 429 275 reactions, one protobuf `Reaction` per row,
read with `ord-schema` 0.9.6 (Apache-2.0; needs numpy, pandas, pyarrow, rdkit, protobuf<6). Parts:
41 files `uspto-grants-YYYY` (1976-2016): 1 771 032 reactions, 1.14 GB; 3 files (409 035 + 40 000 +
30 000 reactions, 125 MB) of reaction SMILES "downloaded from rexgen_direct" (Coley's USPTO training
set, so also patent-derived, no conditions); 50 other files, about 179 000 reactions, about 22 MB, mostly
high-throughput experiments (HTE).

**(a) Licence, from the official files.** Data under `data/`: CC BY-SA 4.0 (`LICENSE`, `CITATION.cff`,
GitHub API `cc-by-sa-4.0`); scripts: Apache-2.0. Attribution (3a) means: cite ORD (Kearnes et al., JACS
2021, 143, 18820, doi 10.1021/jacs.1c09820) and the source of a dataset. ShareAlike (3b) applies only
"if You Share Adapted Material", and "Share" means to provide material to the public. So a private
local index (fingerprints, mapped copies) built from it carries no obligation as long as it is not
published: it must stay out of git, the Docker image and the README. If it were ever shared, the shared
part would have to be CC BY-SA 4.0 too. This is a reading of the licence text, not legal advice. For
comparison PrecedentFinder's data is CC BY 4.0 (no share-alike) and Lowe's original extraction is CC0
(NextMove and the figshare record; the licence field itself was not read).

**(c) Overlap with the patent table we already use: ORD contains it, we are not missing a source.**
Measured: 1976, PrecedentFinder 2 914 reactions against ORD 17 855; 983 of PrecedentFinder's 987
patents are in ORD; of 3 000 random ORD reactions (reactants > product, atom maps removed, canonical)
495 (16.5 %) are in PrecedentFinder, which equals the 16.3 % ratio of the two tables. 1990: 7 986
against 18 641, 1 573 of 1 578 patents, 1 283 of 3 000 sampled (42.8 %, ratio 42.8 %). So
PrecedentFinder's grant rows are a filtered subset of ORD's grant rows (ORD keeps 2 to 6 times more
rows in those years; whether the extra rows are usable was not checked). The opposite direction: 964 034
of PrecedentFinder's 1 803 445 rows have application numbers (`US2005...A1`); ORD has no applications
("uspto-grants" only). Per year the tables cross over (2005: PrecedentFinder 63 759, ORD 36 478). The
agent C remark "the patent part of ORD is the same as ours" is therefore right for grants and wrong
about the whole table. Not verified: years other than 1976 and 1990.

**(b) What ORD would add.** Only the non-patent part, about 179 000 reactions, and it is narrow: the
largest sets are 50 688 C-N couplings (2 amines x 4 aryl halides, per its description), 47 015 amide
couplings (632 product pairs under 95 conditions), 39 347 Pfizer HTE, 9 632 Chan-Lam, 5 760 and 984
Suzuki/Buchwald screens, 750 AstraZeneca ELN Buchwald reactions. Many conditions for few substrates:
useful to say which catalyst, ligand, base and solvent gave which yield for a similar coupling, weak for
a step with unusual substrates (expected from the descriptions, NOT measured on our route steps). These
files have no atom maps and no reaction-type tags, so our method (rxnmapper, then the SMARTS library)
would have to be run on them. CORRECTION (2026-11-13, an earlier version of this paragraph said there
was no ready search): the files have no search, and the `ord-interface` repository (Apache-2.0; Postgres
with the RDKit cartridge + Valkey) is archived (GitHub API `archived: true`, last push 2026-10-06), but
`https://open-reaction-database.org/api` is a live free API (OpenAPI at `/api/openapi.json`, no key was
needed, title "Open Reaction Database viewer", 93 datasets, 2 428 291 reactions: the same corpus). It has
`POST /api/search` with substructure, similarity (Tanimoto on Morgan fingerprints) and reaction SMARTS
(reactant, product and agent templates matched by reaction role), comparisons on any field, aggregates,
`GET /api/reactions` (up to 100 records by ID) and `POST /api/download`. Measured on 2026-11-13: `/api/health`
answers; a reaction-SMARTS amide query (`C(=O)O.N>>C(=O)N`) returned HTTP 504 from the load balancer, and a
similarity search for ibuprofen at threshold 0.7 returned HTTP 504 with the message "the search exceeded
30.0 seconds while matching similarity ... it ran to completion because that phase cannot be interrupted".
So the hosted search has a 30 s cap and the two broad queries tried did not fit in it; narrower queries
(a dataset filter, a rare substructure) were not tried. The local index answers in 0.01-1.4 s. Terms of
use and rate limits of the hosted service were not found, and every query sends the molecules to a third
party. Without the hosted search we would write a loader that turns ORD rows into the columns
`precedent_search.py` reads. Cost
(estimates, not measured): download of the HTE files alone about 22 MB, all of ORD about 1.3 GB (our
Hugging Face download of 12.7 MB took about two minutes), about +10 % rows in the 3.5 GB resident
index, rxnmapper over 179 000 reactions is the unknown part (hours at most, not tested).

**Quality is not better than ours.** The USPTO rows are the same text mining (`is_mined`), with the same
errors. In the 1976 sample the first reaction lists triethylamine as `SOLVENT` and gives amounts as
`moles 0.0` placeholders. A yield (`percentage`) exists for 950 of 5 000 sampled 1976 reactions (19 %)
and for 2 475 of 5 000 in 1990 (50 %), while PrecedentFinder has a yield for every row. What ORD has
that PrecedentFinder lacks: a `reaction_role` per component (REACTANT, REAGENT, SOLVENT, CATALYST,
WORKUP), so reagent and solvent could be told apart (see Honest limits), but the role is whatever the
extraction wrote and was wrong in the one reaction read in detail.

**(d) Conflict with the safety rule, measured on the samples.** The patent rows carry what the project
forbids to show: a verbatim `notes.procedure_details` paragraph in 5 000 of 5 000 sampled reactions in
both years (amounts, temperatures, times, work-up), `conditions.temperature.setpoint` in 471 of 5 000
(1976) and 787 of 5 000 (1990), `outcomes.reaction_time` in 1 538 and 2 247, component amounts (mass,
moles, volume) in most, and structured `workups` (4 046 and 4 570 of 5 000). The AstraZeneca ELN rows (the only HTE set
read) carry a temperature in 750 of 750 and moles in every component; the other HTE sets were not read. Nothing of this exists in the
PrecedentFinder table, which is why variant B was possible without touching the rule.

Options, none decided:
1. **Do not connect ORD (recommended).** PrecedentFinder already holds the useful patent rows, with the
   share-alike licence and the loader cost avoided; ORD's extra grant rows have the same text-mining
   quality and its roles are unreliable; the parts that are truly new (HTE) are narrow. Revisit if the
   golden or manual questions show couplings (Suzuki, Buchwald, amide) with no precedent found.
2. **Add only the HTE sets, reduced at load time** to product, reactants, catalyst/ligand/base/solvent
   names, yield and the dataset citation; temperature, time and amounts dropped when the index is built,
   never stored or shown. This keeps the letter of the "no amounts, temperatures, times" part but breaks
   the sentence "solvents and reagents only from a US patent record", so it needs the owner to widen the
   rule in CLAUDE.md, `prompts/common.md`, `synthesis.md`, `supervisor.md` and here (new label, for
   example "from a published HTE dataset, not verified"), plus the work above and a CC BY-SA review of
   the stored index. Only choose it if option 1 shows a real gap.
3. **Use ORD with conditions** (temperature, time, amounts, procedure text). Not recommended: it is the
   thing the safety rule exists to prevent, and a verbatim patent procedure would be a synthesis
   procedure whatever the label says.

4. **Query the hosted ORD API for the HTE/non-patent part only** (allowed since 2026-11-13: free, no
   download). Needs the same rule widening as option 2, a filter on dataset names, and dropping the
   temperature, time, amount and procedure fields from every record on our side before the model or the
   user sees it (the API returns them). Open points: the 30 s cap and 504s measured above, no stated
   terms or rate limits, the molecule of every query goes to a third party, and no atom-map or
   reaction-type filter, so the "same reaction type" logic of the local method is not available. Try it
   only if option 1 shows a gap, and measure narrower queries first.

Sample files are in the session scratchpad, not in the project; no dependency, setting or prompt was
changed.


## Hosted ORD backend (added 2026-11-13, owner: "replace the local one with the ORD API")

Implemented as a second backend (`PRECEDENTS_BACKEND=ord`, default `local`), not as a replacement, because the
hosted service failed the reliability test (owner's own condition: delete the local files only if it works well).
- Code: `minichem/tools/ord_client.py` (HTTP, 503 retried twice, 504 not retried), `ord_precedents.py` (search,
  parse, rank), `minichem/ord_proto/reaction_pb2.py` (copied unchanged from ord-schema 0.9.6, Apache-2.0; ord-schema
  itself is not installed: it pins protobuf<6 and pulls pandas and pyarrow; `protobuf==7.36.2` is pinned in the project).
- Search: patent reactions only (`provenance.patent` not null), so the rule "reagents and solvents only from a US
  patent record" is unchanged. Step 1 exact product (`same_compound`), step 2 similar products (0.7) if fewer than
  10 hits. Rejected: the API's `reaction_smarts` (504 in earlier tests) and `/ask` (an LLM on their side).
- Reagents and solvents are the middle part of the record's `REACTION_CXSMILES` (as in the local data); reactants
  without an atom map shared with the product move there. Component roles are NOT used (they were wrong in the
  record read: ether as reactant, triethylamine as solvent).
- Safety: only reaction SMILES, patent, year and yield are read; temperature, time, amounts, workups, procedure text
  are never read (test `test_parsed_record_holds_no_conditions`). Licence: ORD data CC BY-SA 4.0; nothing is stored
  or shared, results are shown in answers with the ORD citation in `ORD_PROVENANCE`.
- Privacy: the product SMILES leaves the machine. Down: tool returns `{"error","hint"}`.
- Measured 2026-11-13: 3.6 s exact and 9 s similar search when free; afterwards 503/504/timeouts for the rest of the
  session (also on `dataset_id` queries, 40 s). Possibly partly caused by our own earlier heavy queries (the server
  finishes a search after the 504); not established. NOT verified: ranking quality on live results, real-model run.
- Next: re-measure at another time (3 to 5 spaced calls); if it answers reliably under about 15 s, run the precedent
  questions with the real model and only then consider removing `~/tools/precedents` (2.0 GB).
