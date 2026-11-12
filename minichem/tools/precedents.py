"""Reaction precedents from US patents (text-mined), OFF by default.

Given one retrosynthesis step (product and reactants), the search returns similar reactions
that were published in US patents 1976-2016: the patent number, the year, the text-mined
yield and the reagents and solvents exactly as the patent record lists them. The records come
from the PrecedentFinder dataset (USPTO reactions extracted by text mining, CC BY 4.0), so they
are NOT checked: the extraction can be wrong. The data holds no temperatures, times or amounts,
and none is ever added here.

`MockPrecedents` is used in tests. `PatentPrecedents` (in `precedent_search.py`) needs the
dataset and the packages of the separate environment; it is used only by
`mcp_server/precedent_server.py`.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, Field
from rdkit import Chem

from minichem.tools.models import ToolError

MAX_PRECEDENTS = 5
_MAP_NUMBER = re.compile(r":\d+(?=\])")
PROVENANCE = (
    "From US patents 1976-2016, text-mined, not verified. Reagents and solvents are listed "
    "as the patent record has them (not told apart); no temperatures, times or amounts exist "
    "in the data."
)


class PrecedentSetupError(RuntimeError):
    """The precedent database or its settings are missing; the server reports it as a tool error."""


class PrecedentMolecule(BaseModel):
    """A molecule of a precedent reaction. Atom maps are removed."""

    smiles: str


class Precedent(BaseModel):
    """One reaction from a patent record, ranked by similarity to the query step.

    `similarity` compares the whole reaction and `centre_similarity` the atoms that change,
    both 0-1 from fingerprints (a ranking heuristic, not a probability). `yield_percent` is the
    text-mined yield of the record. Summary fields come first because the supervisor sees tool
    results cut to a fixed length.
    """

    patent: str = Field(description="US patent number as in the record, e.g. US03931151")
    year: int
    yield_percent: float | None = None
    similarity: float
    centre_similarity: float
    reactants: list[PrecedentMolecule]
    reagents_and_solvents: list[PrecedentMolecule]
    product: PrecedentMolecule


class PrecedentsResponse(BaseModel):
    """Precedents for one step. `provenance` must be passed on with every precedent shown."""

    provenance: str = PROVENANCE
    mapping_confidence: float = Field(
        description="Confidence (0-1) of the atom mapping of the query step; low means the "
        "step was probably mapped wrongly and the precedents are less relevant."
    )
    n_candidates: int = Field(description="Reactions of the same reaction type in the database.")
    precedents: list[Precedent]


class PrecedentSearch(Protocol):
    """Anything that can find precedents for a step."""

    def find(
        self, product: str, reactants: list[str], max_precedents: int
    ) -> PrecedentsResponse | ToolError: ...


class MockPrecedents:
    """Deterministic stand-in for tests: always returns one fixed precedent."""

    def find(
        self, product: str, reactants: list[str], max_precedents: int
    ) -> PrecedentsResponse | ToolError:
        precedent = Precedent(
            patent="US00000000",
            year=1999,
            yield_percent=80.0,
            similarity=0.9,
            centre_similarity=1.0,
            reactants=[PrecedentMolecule(smiles=s) for s in reactants],
            reagents_and_solvents=[PrecedentMolecule(smiles="CCO")],
            product=PrecedentMolecule(smiles=product),
        )
        return PrecedentsResponse(
            mapping_confidence=0.9, n_candidates=1, precedents=[precedent][:max_precedents]
        )


def strip_maps(smiles: str) -> str:
    """SMILES without atom-map numbers (canonical when RDKit can read it)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _MAP_NUMBER.sub("", smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _map_numbers(smiles: str) -> set[int]:
    mol = Chem.MolFromSmiles(smiles)
    return set() if mol is None else {a.GetAtomMapNum() for a in mol.GetAtoms()} - {0}


def clean_mapped(mapped: str) -> str:
    """Reaction SMILES as stored in the database: `reactants>reagents>product`.

    Reactants that give no mapped atom to the product move to the reagents and atom maps
    that do not appear in the product are removed, as when the database was built.
    """
    left, product = mapped.split(">>")
    product_maps = _map_numbers(product)
    reactants: list[str] = []
    reagents: list[str] = []
    for smiles in left.split("."):
        (reactants if _map_numbers(smiles) & product_maps else reagents).append(smiles)
    return f"{'.'.join(reactants)}>{'.'.join(reagents)}>{product}"


def parse_record(clean: str) -> tuple[list[str], list[str], str]:
    """Split a database `ReactionSmilesClean` into reactants, reagents and product SMILES."""
    reactants, reagents, product = clean.split()[0].split(">")
    return _strip_all(reactants), list(dict.fromkeys(_strip_all(reagents))), strip_maps(product)


def _strip_all(part: str) -> list[str]:
    return [strip_maps(s) for s in part.split(".") if s]
