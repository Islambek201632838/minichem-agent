"""Pure RDKit functions. No LLM, no network. These back both the MCP server and the agents."""

from __future__ import annotations

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from minichem.tools.models import (
    Constraints,
    FilterResponse,
    ItemError,
    MoleculeProperties,
    PropertiesResponse,
    SimilarityHit,
    SimilarityResponse,
    ToolError,
    ValidationItem,
    ValidationResponse,
)

RDLogger.DisableLog("rdApp.*")  # parse failures are reported as results, not stderr noise

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def parse(smiles: str) -> Chem.Mol | None:
    """Parse and sanitise a SMILES string, or return None."""
    if not smiles or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def canonical(smiles: str) -> str | None:
    """Canonical SMILES, or None if the input is not valid."""
    mol = parse(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _invalid(smiles: str) -> ItemError:
    return ItemError(
        input=smiles,
        error="Not a valid SMILES (RDKit could not parse or sanitise it).",
    )


def properties_of(mol: Chem.Mol) -> MoleculeProperties:
    """Descriptors for an already-parsed molecule."""
    return MoleculeProperties(
        smiles=Chem.MolToSmiles(mol),
        mw=round(Descriptors.MolWt(mol), 2),
        logp=round(Crippen.MolLogP(mol), 2) + 0.0,  # + 0.0 turns -0.0 into 0.0
        hbd=rdMolDescriptors.CalcNumHBD(mol),
        hba=rdMolDescriptors.CalcNumHBA(mol),
        rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        qed=round(QED.qed(mol), 3),
    )


def calculate_properties(smiles_list: list[str]) -> PropertiesResponse:
    """MW, logP, HBD, HBA, rotatable bonds, TPSA and QED for each SMILES."""
    results: list[MoleculeProperties] = []
    errors: list[ItemError] = []
    for smiles in smiles_list:
        mol = parse(smiles)
        if mol is None:
            errors.append(_invalid(smiles))
        else:
            results.append(properties_of(mol))
    return PropertiesResponse(results=results, errors=errors)


def validate_smiles(smiles_list: list[str]) -> ValidationResponse:
    """Check each SMILES and return its canonical form and molecular formula."""
    items: list[ValidationItem] = []
    for smiles in smiles_list:
        mol = parse(smiles)
        if mol is None:
            items.append(ValidationItem(input=smiles, valid=False, error=_invalid(smiles).error))
        else:
            items.append(
                ValidationItem(
                    input=smiles,
                    valid=True,
                    smiles=Chem.MolToSmiles(mol),
                    formula=CalcMolFormula(mol),
                )
            )
    return ValidationResponse(results=items)


def tanimoto_similarity(query: str, candidates: list[str]) -> SimilarityResponse | ToolError:
    """Tanimoto similarity (Morgan r=2, 2048 bits) of each candidate to the query."""
    query_mol = parse(query)
    if query_mol is None:
        return ToolError(
            error=f"Query is not a valid SMILES: {query!r}",
            hint="Fix or re-derive the query SMILES, then call again.",
        )
    query_fp = _MORGAN.GetFingerprint(query_mol)
    hits: list[SimilarityHit] = []
    errors: list[ItemError] = []
    for smiles in candidates:
        mol = parse(smiles)
        if mol is None:
            errors.append(_invalid(smiles))
            continue
        score = DataStructs.TanimotoSimilarity(query_fp, _MORGAN.GetFingerprint(mol))
        hits.append(SimilarityHit(smiles=Chem.MolToSmiles(mol), tanimoto=round(score, 3)))
    return SimilarityResponse(query=Chem.MolToSmiles(query_mol), results=hits, errors=errors)


def similarity_value(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    """Raw Tanimoto between two parsed molecules (used by the analogue generator)."""
    return DataStructs.TanimotoSimilarity(
        _MORGAN.GetFingerprint(mol_a), _MORGAN.GetFingerprint(mol_b)
    )


def satisfies(props: MoleculeProperties, c: Constraints) -> bool:
    """True if the molecule meets every constraint that is set."""
    checks = (
        (c.mw_min is None or props.mw >= c.mw_min),
        (c.mw_max is None or props.mw <= c.mw_max),
        (c.logp_min is None or props.logp >= c.logp_min),
        (c.logp_max is None or props.logp <= c.logp_max),
        (c.hbd_max is None or props.hbd <= c.hbd_max),
        (c.hba_max is None or props.hba <= c.hba_max),
        (c.rotb_max is None or props.rotatable_bonds <= c.rotb_max),
        (c.tpsa_max is None or props.tpsa <= c.tpsa_max),
        (c.qed_min is None or props.qed >= c.qed_min),
    )
    return all(checks)


def filter_molecules(smiles_list: list[str], constraints: Constraints) -> FilterResponse:
    """Split molecules into those that meet the constraints and those that do not."""
    computed = calculate_properties(smiles_list)
    passed = [p for p in computed.results if satisfies(p, constraints)]
    failed = [p for p in computed.results if not satisfies(p, constraints)]
    return FilterResponse(passed=passed, failed=failed, errors=computed.errors)
