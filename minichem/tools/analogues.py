"""Analogue generation.

This is a deliberately simple stand-in for a generative model such as REINVENT:
(1) rule-based point mutations of the query, and (2) BRICS recombination of the
query's fragments with fragments from a few common drug-like molecules. Candidates
are kept only if they are similar to the query (Morgan/Tanimoto) and meet the
user's property constraints. It is deterministic, so results are reproducible.
"""

from __future__ import annotations

from itertools import islice

from rdkit import Chem
from rdkit.Chem import BRICS, AllChem

from minichem.tools import chem
from minichem.tools.models import Analogue, AnalogueResponse, Constraints, ToolError

# name -> reaction SMARTS. Each is applied at every matching site, one change at a time.
MUTATIONS: dict[str, str] = {
    "aryl H -> F": "[cH:1]>>[c:1]F",
    "aryl H -> Cl": "[cH:1]>>[c:1]Cl",
    "aryl H -> CH3": "[cH:1]>>[c:1]C",
    "aryl CH -> N": "[cH:1]>>[n:1]",
    "acid -> primary amide": "[CX3:1](=O)[OX2H]>>[C:1](=O)N",
    "acid -> methyl ester": "[CX3:1](=O)[OX2H]>>[C:1](=O)OC",
    "O -> S": "[OX2:1]>>[SX2:1]",
    "OH -> NH2": "[OX2H:1]>>[NX3H2:1]",
    "CH3 -> CF3": "[CH3:1]>>[C:1](F)(F)F",
    "CH2 -> NH": "[CH2;!R:1]>>[NH:1]",
    "add methyl": "[CH2;!R:1]>>[CH1:1]C",
    "remove methyl": "[CH3:1][*:2]>>[*:2]",
}
_REACTIONS = {name: AllChem.ReactionFromSmarts(smarts) for name, smarts in MUTATIONS.items()}

# Stand-in building blocks (paracetamol, naproxen, salicylic acid, pyridine, indole,
# N-methylpiperazine, fluorobenzene, piperidine): a tiny pool to diversify BRICS products.
DONOR_SMILES = (
    "CC(=O)Nc1ccc(O)cc1",
    "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "OC(=O)c1ccccc1O",
    "c1ccncc1",
    "c1ccc2[nH]ccc2c1",
    "CN1CCN(C)CC1",
    "Fc1ccccc1",
    "C1CCNCC1",
)
_BRICS_PRODUCT_LIMIT = 400
DEFAULT_MIN_SIMILARITY = 0.4


def _mutate(mol: Chem.Mol) -> set[str]:
    """All single-step mutants of `mol` that sanitise to valid molecules."""
    products: set[str] = set()
    for reaction in _REACTIONS.values():
        for outcome in reaction.RunReactants((mol,)):
            product = outcome[0]
            try:
                Chem.SanitizeMol(product)
            except Chem.rdchem.MolSanitizeException:
                continue
            products.add(Chem.MolToSmiles(product))
    return products


def _brics(mol: Chem.Mol) -> set[str]:
    """BRICS recombination of the query's fragments with the donor fragments."""
    fragment_smiles: set[str] = set(BRICS.BRICSDecompose(mol))
    for donor in DONOR_SMILES:
        donor_mol = Chem.MolFromSmiles(donor)
        fragment_smiles |= set(BRICS.BRICSDecompose(donor_mol))
    fragments = [Chem.MolFromSmiles(s) for s in sorted(fragment_smiles)]
    built = BRICS.BRICSBuild(fragments, scrambleReagents=False, maxDepth=2)
    products: set[str] = set()
    for product in islice(built, _BRICS_PRODUCT_LIMIT):
        try:
            Chem.SanitizeMol(product)
        except Chem.rdchem.MolSanitizeException:
            continue
        products.add(Chem.MolToSmiles(product))
    return products


def generate_analogues(
    smiles: str,
    n: int = 10,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    constraints: Constraints | None = None,
) -> AnalogueResponse | ToolError:
    """Propose up to `n` analogues of `smiles`, most similar first."""
    query = chem.parse(smiles)
    if query is None:
        return ToolError(
            error=f"Query is not a valid SMILES: {smiles!r}",
            hint="Validate or canonicalise the SMILES first.",
        )
    query_smiles = Chem.MolToSmiles(query)
    constraints = constraints or Constraints()

    first_round = _mutate(query)
    second_round = {p for s in first_round for p in _mutate(Chem.MolFromSmiles(s))}
    origins = dict.fromkeys(first_round | second_round, "mutation")
    for s in _brics(query):
        origins.setdefault(s, "brics")
    origins.pop(query_smiles, None)

    kept: list[Analogue] = []
    for cand, origin in origins.items():
        mol = Chem.MolFromSmiles(cand)
        if mol is None:
            continue
        similarity = chem.similarity_value(query, mol)
        props = chem.properties_of(mol)
        if similarity < min_similarity or not chem.satisfies(props, constraints):
            continue
        kept.append(
            Analogue(
                smiles=props.smiles,
                tanimoto=round(similarity, 3),
                mw=props.mw,
                logp=props.logp,
                hbd=props.hbd,
                hba=props.hba,
                origin=origin,
            )
        )
    kept.sort(key=lambda a: (-a.tanimoto, a.smiles))
    chosen = kept[:n]
    note = ""
    if len(chosen) < n:
        note = (
            f"Only {len(chosen)} candidates met min_similarity={min_similarity} and the "
            "constraints. Lower min_similarity or relax constraints to get more."
        )
    return AnalogueResponse(
        query=query_smiles,
        requested=n,
        returned=len(chosen),
        analogues=chosen,
        note=note,
    )
