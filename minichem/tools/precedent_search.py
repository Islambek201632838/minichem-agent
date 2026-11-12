"""Precedent search over the PrecedentFinder dataset (USPTO 1976-2016, text-mined).

Needs the packages and data of the separate precedent environment (rxnmapper, reaction_utils,
pandas, scipy, paretoset; see README). Nothing is downloaded here. It is used only by
`mcp_server/precedent_server.py`. Smoke-tested by hand and by the engine tests, which run in
that environment (they are skipped in the project environment).

Steps: atom-map the query step with rxnmapper (the database is mapped the same way, so
reactants that give no atom to the product become reagents), find the database reactions of
the same reaction type (functional groups that change), rank them (`precedent_rank.py`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem

from minichem.tools.models import ToolError
from minichem.tools.precedents import (
    MAX_PRECEDENTS,
    Precedent,
    PrecedentMolecule,
    PrecedentSetupError,
    PrecedentsResponse,
    clean_mapped,
    parse_record,
)

METADATA_FILE = "metadata_cleaned.csv.gz"
GLOBAL_FP_FILE = "rfp_precomp.npz"
CENTRE_FP_FILE = "rcfp_precomp.npz"
SMARTS_FILE = "group_smarts.json"  # from the precedent_finder repository, data/
REQUIRED_FILES = (METADATA_FILE, GLOBAL_FP_FILE, CENTRE_FP_FILE, SMARTS_FILE)
NUMBITS = 1024
MAX_CENTRES = 6
MAX_REACTANTS = 5
MIN_MAPPING_CONFIDENCE = 0.1  # the threshold used when the database was built
COLUMNS = ["ID", "Year", "CuratedYield", "ReactionSmilesClean", "ReactiveFunction"]


def missing_files(data_dir: Path) -> list[str]:
    """Names of required data files that are not in `data_dir`."""
    return [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]


class PatentPrecedents:
    """Loads the dataset once (about 1.8 million reactions) and answers searches."""

    def __init__(self, data_dir: Path) -> None:
        if missing := missing_files(data_dir):
            raise PrecedentSetupError(
                f"Precedent data incomplete in {data_dir}: missing {missing}."
            )
        import pandas as pd
        from rxnmapper import RXNMapper
        from rxnutils.chem.smartslib import SmartsLibrary
        from scipy import sparse

        self._meta = pd.read_csv(
            data_dir / METADATA_FILE,
            sep="\t",
            usecols=COLUMNS,
            dtype={"ReactiveFunction": "category", "ID": "string"},
        )
        self._groups = self._meta.groupby("ReactiveFunction", observed=True).indices
        self._global = sparse.load_npz(data_dir / GLOBAL_FP_FILE)
        self._centre = sparse.load_npz(data_dir / CENTRE_FP_FILE)
        self._smarts = SmartsLibrary.load(str(data_dir / SMARTS_FILE))
        self._mapper = RXNMapper()

    def _map(self, product: str, reactants: list[str]) -> tuple[str, float] | ToolError:
        query = f"{'.'.join(reactants)}>>{product}"
        result = self._mapper.get_attention_guided_atom_maps([query])[0]
        mapped = result.get("mapped_rxn", "")
        confidence = float(result.get("confidence", 0.0))
        if ">>" not in mapped or confidence <= MIN_MAPPING_CONFIDENCE:
            return ToolError(
                error="The step could not be atom-mapped reliably.",
                hint="Precedents cannot be searched for this step.",
            )
        return clean_mapped(mapped), confidence

    def find(
        self, product: str, reactants: list[str], max_precedents: int
    ) -> PrecedentsResponse | ToolError:
        """Precedents for `reactants -> product`, best first (at most `max_precedents`)."""
        from rxnutils.chem.features.reaction_centre_fp import ReactionCentreFingerprint
        from rxnutils.chem.features.simple_rdkit import SimpleRdkitFingerprint
        from rxnutils.chem.reaction import ChemicalReaction

        from minichem.tools.precedent_rank import centre_similarity, global_similarity, select_best

        if any(Chem.MolFromSmiles(s) is None for s in [product, *reactants]):
            return ToolError(error="Invalid SMILES.", hint="Pass molecules from tool results.")
        if len(reactants) > MAX_REACTANTS:
            return ToolError(error=f"At most {MAX_REACTANTS} reactants are supported.")
        mapped = self._map(product, reactants)
        if isinstance(mapped, ToolError):
            return mapped
        clean, confidence = mapped
        reaction = ChemicalReaction(clean, clean_smiles=False)
        try:
            functions, _ = self._smarts.detect_reactive_functions(
                reaction, sort=True, add_none=True, target_size=None, max_reactants=MAX_REACTANTS
            )
        except ValueError as error:  # no reactants or product left after cleaning
            return ToolError(error=f"The step cannot be compared: {error}")
        rows = self._groups.get("|".join(functions))
        if rows is None or len(rows) == 0:
            return ToolError(
                error="No patent reaction of this reaction type in the database.",
                hint="Report that no precedent was found; do not guess conditions.",
            )
        query_global = np.asarray(
            SimpleRdkitFingerprint("fingerprint_mixed", numbits=NUMBITS)(reaction)
        )
        centres = np.asarray(ReactionCentreFingerprint(numbits=NUMBITS)(reaction)[:MAX_CENTRES])
        sim = global_similarity(self._global[rows], query_global)
        centre = centre_similarity(self._centre[rows], centres, NUMBITS)
        meta = self._meta.iloc[rows]
        best, _ = select_best(
            sim,
            centre,
            meta["CuratedYield"].to_numpy(dtype=float),
            meta["Year"].to_numpy(dtype=float),
            min(max_precedents, MAX_PRECEDENTS),
        )
        precedents = [self._precedent(meta.iloc[i], sim[i], centre[i]) for i in best]
        return PrecedentsResponse(
            mapping_confidence=round(confidence, 3),
            n_candidates=len(rows),
            precedents=precedents,
        )

    @staticmethod
    def _precedent(record, sim: float, centre: float) -> Precedent:
        reactants, reagents, product = parse_record(record["ReactionSmilesClean"])
        yield_percent = record["CuratedYield"]
        return Precedent(
            patent=str(record["ID"]).split(";")[0],
            year=int(record["Year"]),
            yield_percent=None if yield_percent != yield_percent else float(yield_percent),
            similarity=round(float(sim), 3),
            centre_similarity=round(float(centre), 3),
            reactants=[PrecedentMolecule(smiles=s) for s in reactants],
            reagents_and_solvents=[PrecedentMolecule(smiles=s) for s in reagents],
            product=PrecedentMolecule(smiles=product),
        )
