"""Checks on a single retrosynthetic step, using only RDKit (no AiZynthFinder needed).

AiZynthFinder proposes a step by applying a retro template (product pattern >> reactant
patterns). `forward_check` runs that template the other way round on the proposed reactants
and asks whether the product comes back. It is an independent sanity check of the step, not a
proof that the reaction works in a flask.
"""

from __future__ import annotations

from itertools import permutations

from rdkit import Chem
from rdkit.Chem import AllChem

TEMPLATE_SEPARATOR = ">>"


def _plain(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def forward_check(template: str, reactants: list[str], product: str) -> bool:
    """True if the reversed template turns `reactants` into `product` (stereo ignored)."""
    parts = template.split(TEMPLATE_SEPARATOR)
    target = Chem.MolFromSmiles(product)
    mols = [Chem.MolFromSmiles(s) for s in reactants]
    if len(parts) != 2 or target is None or any(m is None for m in mols):
        return False
    reaction = AllChem.ReactionFromSmarts(f"{parts[1]}{TEMPLATE_SEPARATOR}{parts[0]}")
    if reaction.GetNumReactantTemplates() != len(mols):
        return False
    wanted = _plain(target)
    for ordered in permutations(mols):
        for products in reaction.RunReactants(tuple(ordered)):
            for candidate in products:
                try:
                    Chem.SanitizeMol(candidate)
                except Chem.rdchem.MolSanitizeException:
                    continue
                if _plain(candidate) == wanted:
                    return True
    return False
