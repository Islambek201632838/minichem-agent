"""Synthetic-accessibility score and structural flags. Computed values only."""

from __future__ import annotations

import os
import sys

from rdkit import Chem, RDConfig
from rdkit.Chem import FindMolChiralCenters, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from minichem.tools import chem
from minichem.tools.models import (
    FlagsResponse,
    ItemError,
    SaResponse,
    SaScore,
    StructuralFlags,
)

# sascorer ships in RDKit's Contrib directory and is not an importable package.
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

EASY_MAX = 3.0
MODERATE_MAX = 6.0
MACROCYCLE_MIN_RING = 8
MANY_STEREOCENTERS = 3

_params = FilterCatalogParams()
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_BRENK = FilterCatalog(_params)


def _label(score: float) -> str:
    if score <= EASY_MAX:
        return "easy"
    if score <= MODERATE_MAX:
        return "moderate"
    return "hard"


def sa_scores(smiles_list: list[str]) -> SaResponse:
    """SA score (1 easy .. 10 hard) with a coarse label for each SMILES."""
    results: list[SaScore] = []
    errors: list[ItemError] = []
    for smiles in smiles_list:
        mol = chem.parse(smiles)
        if mol is None:
            errors.append(ItemError(input=smiles, error="Not a valid SMILES."))
            continue
        score = round(sascorer.calculateScore(mol), 2)
        results.append(SaScore(smiles=Chem.MolToSmiles(mol), sa_score=score, label=_label(score)))
    return SaResponse(results=results, errors=errors)


def _concerns(
    largest_ring: int, unspecified: int, spiro: int, bridgehead: int, alerts: list[str]
) -> list[str]:
    concerns: list[str] = []
    if largest_ring >= MACROCYCLE_MIN_RING:
        concerns.append(f"macrocycle (ring of {largest_ring} atoms)")
    if spiro:
        concerns.append("spiro centre")
    if bridgehead:
        concerns.append("bridged ring system")
    if unspecified >= MANY_STEREOCENTERS:
        concerns.append(f"{unspecified} unspecified stereocentres")
    if alerts:
        concerns.append("structural alerts: " + ", ".join(alerts))
    return concerns


def structural_flags(smiles_list: list[str]) -> FlagsResponse:
    """Ring, stereo and Brenk-alert flags that hint at hard-to-make or unstable molecules."""
    results: list[StructuralFlags] = []
    errors: list[ItemError] = []
    for smiles in smiles_list:
        mol = chem.parse(smiles)
        if mol is None:
            errors.append(ItemError(input=smiles, error="Not a valid SMILES."))
            continue
        ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
        largest = max(ring_sizes, default=0)
        centers = FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        unspecified = sum(1 for _, tag in centers if tag == "?")
        spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        alerts = sorted({e.GetDescription() for e in _BRENK.GetMatches(mol)})
        results.append(
            StructuralFlags(
                smiles=Chem.MolToSmiles(mol),
                n_rings=len(ring_sizes),
                largest_ring_size=largest,
                n_stereocenters=len(centers),
                n_unspecified_stereocenters=unspecified,
                n_spiro_atoms=spiro,
                n_bridgehead_atoms=bridgehead,
                alerts=alerts,
                concerns=_concerns(largest, unspecified, spiro, bridgehead, alerts),
            )
        )
    return FlagsResponse(results=results, errors=errors)
