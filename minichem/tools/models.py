"""Typed inputs and outputs shared by the local tools and the MCP server."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

MAX_BATCH = 50
SmilesList = Annotated[
    list[str],
    Field(
        min_length=1,
        max_length=MAX_BATCH,
        description="Handles of the molecules, e.g. ['M1', 'M2'].",
    ),
]


class ItemError(BaseModel):
    """A per-item failure the agent can read and correct."""

    input: str
    error: str


class ToolError(BaseModel):
    """A whole-call failure. Returned instead of raising."""

    error: str
    hint: str = ""


class Constraints(BaseModel):
    """Optional property windows. Unset fields are not applied."""

    mw_min: float | None = Field(default=None, description="Minimum molecular weight (g/mol)")
    mw_max: float | None = Field(default=None, description="Maximum molecular weight (g/mol)")
    logp_min: float | None = Field(default=None, description="Minimum Crippen logP")
    logp_max: float | None = Field(default=None, description="Maximum Crippen logP")
    hbd_max: int | None = Field(default=None, ge=0, description="Maximum H-bond donors")
    hba_max: int | None = Field(default=None, ge=0, description="Maximum H-bond acceptors")
    rotb_max: int | None = Field(default=None, ge=0, description="Maximum rotatable bonds")
    tpsa_max: float | None = Field(default=None, ge=0, description="Maximum TPSA (A^2)")
    qed_min: float | None = Field(default=None, ge=0, le=1, description="Minimum QED")


class MoleculeProperties(BaseModel):
    """Computed physicochemical descriptors for one valid molecule."""

    smiles: str = Field(description="Canonical SMILES")
    mw: float
    logp: float
    hbd: int
    hba: int
    rotatable_bonds: int
    tpsa: float
    qed: float


class PropertiesResponse(BaseModel):
    results: list[MoleculeProperties]
    errors: list[ItemError] = []


class ValidationItem(BaseModel):
    input: str
    valid: bool
    smiles: str | None = Field(default=None, description="Canonical SMILES if valid")
    formula: str | None = None
    error: str | None = None


class ValidationResponse(BaseModel):
    results: list[ValidationItem]


class SimilarityHit(BaseModel):
    smiles: str
    tanimoto: float = Field(ge=0, le=1)


class SimilarityResponse(BaseModel):
    query: str
    fingerprint: str = "Morgan radius 2, 2048 bits"
    results: list[SimilarityHit]
    errors: list[ItemError] = []


class FilterResponse(BaseModel):
    passed: list[MoleculeProperties]
    failed: list[MoleculeProperties]
    errors: list[ItemError] = []


class Analogue(BaseModel):
    smiles: str
    tanimoto: float
    mw: float
    logp: float
    hbd: int
    hba: int
    origin: str = Field(description="How it was made: 'mutation' or 'brics'")


class AnalogueResponse(BaseModel):
    query: str
    requested: int
    returned: int
    analogues: list[Analogue]
    note: str = ""


class NameLookup(BaseModel):
    name: str
    smiles: str
    title: str | None = None
    iupac_name: str | None = None
    formula: str | None = None
    source: str = "PubChem"


class SaScore(BaseModel):
    smiles: str
    sa_score: float = Field(description="1 (easy) to 10 (very hard), Ertl & Schuffenhauer")
    label: str


class SaResponse(BaseModel):
    results: list[SaScore]
    errors: list[ItemError] = []


class StructuralFlags(BaseModel):
    smiles: str
    n_rings: int
    largest_ring_size: int
    n_stereocenters: int
    n_unspecified_stereocenters: int
    n_spiro_atoms: int
    n_bridgehead_atoms: int
    alerts: list[str] = Field(description="Brenk structural alert names")
    concerns: list[str]


class FlagsResponse(BaseModel):
    results: list[StructuralFlags]
    errors: list[ItemError] = []
