# Research: REINVENT 4 and reaction precedents (no code written)

Date: 2026-11-12. Sources: the REINVENT4 and precedent_finder GitHub pages, their Zenodo records,
PyPI, and the AstraZeneca `langdmta-lab` code. Web pages were summarised by a fetch tool, so every
version pin below must be re-read from the installed package before it is relied on.

## This Mac (measured today)

Apple M4 Pro, 12 cores (8 performance, 4 efficiency), 24 GB unified memory, arm64, 344 GB free of
460 GB. Download speed measured with a 50 MB test file: 53.6 MB/s (about 430 Mbit/s), one connection.

## Part 1: REINVENT 4 as the analogue generator

| Question | Finding |
|---|---|
| Licence | Apache-2.0 (OpenEye ROCS component needs its own licence; not used) |
| Python | `>=3.11` |
| Pins on the main branch | `torch==2.13.0`, `rdkit>=2025.09.1`, `numpy>=2,<3`, `pandas>=3,<4`, `pydantic>=2,<3`, `polars<2`, `mmpdb>=2.1,<3` and others |
| Conflict with this project | Much smaller than AiZynthFinder's: our `rdkit==2026.9.2` satisfies `>=2025.09.1`. It still goes into its own venv: torch and `mmpdb`/`polars`/`pandas>=3` would enter the project's tests and lockfile for a feature that is off by default |
| macOS | Installer: `python install.py mac` (other targets `cu126`, `rocm6.4`, `xpu`, `cpu`). Linux is the fully validated platform; macOS runs through PyTorch's MPS backend or CPU. Whether every REINVENT operation works on MPS is NOT verified |
| torch wheel | `torch-2.13.0` for macOS arm64, Python 3.12: 111.2 MB (PyPI) |
| Priors (Zenodo 15641297, Apache-2.0) | `mol2mol_medium_similarity.prior` 80.2 MB, `reinvent.prior` 23.2 MB, `mol2mol_scaffold_generic.prior` 80.2 MB. The whole record is 1.0 GB and is not needed |
| Modes | `sampling` (used by AstraZeneca with `mol2mol_medium_similarity`, beam search, CPU), `scoring`, `staged_learning` (reinforcement learning), plus transfer learning; configs are TOML or JSON |
| How AstraZeneca calls it | A new subprocess `reinvent <config>` per request, reading and writing CSV files. We would keep one MCP server and read the CSV in code |

**Download and disk for the minimum setup** (sampling with mol2mol): torch 111 MB + other wheels
(not measured, a few hundred MB) + priors 103 MB. At 53 MB/s the download is about 10 to 20 s of
transfer plus install time; plan for a few minutes in total. Disk: a few GB at most (estimate, not
measured). Adding `mol2mol_scaffold_generic` for reinforcement learning adds 80 MB.

**Can the Mac run it?** Sampling 50 to 100 molecules from an 80 MB transformer: yes, with margin.
8 GiB of memory is what the REINVENT authors call enough for typical design tasks and this Mac has
24 GB. Reinforcement learning (batch 64, 25 to 100 steps, in AstraZeneca's template) is the heavy
part: it is slower than sampling by orders of magnitude and a GPU is not available here. I expect
minutes to tens of minutes on this CPU. That is an estimate; it will be measured on the real prior
before any claim goes into the README. Model training is out of scope and would need a GPU.

**Plan if approved** (same pattern as `retro_server.py`):
1. Separate venv with `uv`, versions re-read from the installed package, priors downloaded by hand.
2. `mcp_server/reinvent_server.py`: one tool for mol2mol sampling (seed argument, number of
   molecules, similarity floor) returning molecule records with the key `smiles`, loaded lazily.
3. Settings `ENABLE_REINVENT` (default false), `REINVENT_PYTHON`, `REINVENT_PRIOR_DIR` in `config.py`.
4. The deterministic generator stays as the fallback when REINVENT is off or fails.
5. Tests with a mock; a real run with real priors reporting time, memory, and a reproducibility
   check (same seed twice, and different seeds: overlap of results). REINVENT is stochastic: the
   README must say so and the seed must be reported with every result.
6. Reinforcement learning is a separate, later step: it is slow and needs a scoring design.

## Part 2: reaction precedents (the PrecedentFinder question)

**An open implementation exists.** `github.com/MolecularAI/precedent_finder` (Apache-2.0). The
authors call it example code, "never intended to be production ready" and not maintained.

| Question | Finding |
|---|---|
| Install | conda environment `pfenv`, Python 3.8 to 3.11 (not 3.12); needs `rxnutils`, `paretoset`, pandas, SciPy. AstraZeneca additionally runs rxnmapper in a Python 3.6 environment |
| Data (Zenodo 15798247, CC BY 4.0) | `metadata_cleaned.csv.gz` 275.1 MB, `rfp_precomp.npz` 326.2 MB, `rcfp_precomp.npz` 60.0 MB: 661.3 MB in total, about 15 s of transfer at 53 MB/s |
| Input | An atom-mapped reaction SMILES. Our AiZynthFinder steps are not mapped, so rxnmapper (a transformer, extra model download, not measured) is needed first |
| Method | Reaction fingerprint similarity plus reaction-centre similarity, then a Pareto selection over similarity, yield and year; `goodness` is a weighted distance to the ideal point. It is a ranking heuristic, not a probability |
| Memory on this Mac | Not measured. The fingerprints are sparse matrices over a USPTO-size set; 24 GB should be enough, to be checked |

**What the data really contains** (I downloaded the first 3 MB of `metadata_cleaned.csv.gz` with an
HTTP range request and read 3 rows): columns `ID`, `Year`, `ReactionSmiles`, `TextMinedYield`,
`CalculatedYield`, `CuratedYield`, `ReactionSmilesClean`, `ReactiveFunction`. The reaction SMILES
has the form `reactants>reagents and solvents>products` (for example `...>C(O)C>...` for ethanol),
atom-mapped, 1976 patent rows among them. There is NO temperature, time or amount in the columns
seen, and the yields are text-mined, so they can be wrong. Rows with empty `CuratedYield` exist
(the three seen had text-mined yields of 78 to 82%).

**Conflict with our safety rule.** CLAUDE.md says the synthesis agent never gives solvents, and
the middle part of every precedent IS the reagent and solvent list. Showing a precedent therefore
changes the rule. Options for the owner (decision needed before any code):

- A. Show only the precedent reaction's reactants and products, its yield and its patent id, and hide the
  reagent and solvent field (consistent with the current rule; much less useful).
- B. Also show the reagents and solvents exactly as in the cited patent record, labelled as
  "from patent US..., text-mined, not validated" (useful; relaxes the rule for database-sourced
  data only, never LLM-written, still no amounts or temperatures because the data has none).
- C. Skip precedents.

Controlled and hazardous substances stay declined before any tool runs (existing rule, and the
roadmap item 8 would make it a code-level filter).

**Recommended design if B or A:** a third MCP server `precedent_server.py` in its own venv
(Python 3.11, `ENABLE_PRECEDENTS`, off by default), loading the 661 MB once. Input: a step from a
`retrosynthesis_routes` result; the server maps it with rxnmapper and returns up to N precedents
with similarity scores, patent id and year. Tested with a mock; real run on aspirin and ibuprofen
steps with real numbers. Rejected alternative: ORD (Open Reaction Database) as the first source:
it is smaller and has richer conditions for some entries, but I did not evaluate its size,
coverage or licence; it can be a second source later.

## What was not verified

Torch and REINVENT behaviour on this Mac (MPS vs CPU), real timings, memory, the install itself,
rxnmapper's download size and Python requirements, the full column list of the precedent data beyond
three rows, and the data's coverage of our molecules. All of that is checked when an install is
approved.
