"""Name a retrosynthetic step by textbook reaction type, using RDKit only.

Each entry is a forward reaction SMARTS (reactants >> product) with a name and a class-level
reagent hint. A step is named only if its template, run on the step's reactants, reproduces the
step's product, the same test as `route_check.forward_check`. Steps that match nothing stay
unnamed: this is a short list of common reactions, not a classifier of all chemistry.

The hints say which KIND of reagent the reaction class needs (a base, a catalyst). They are not
predictions for the molecule at hand and never include amounts, solvents or temperatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class ReactionType:
    """A named reaction class and the kind of reagent it needs."""

    name: str
    reagent_hint: str
    smarts: str


# Order matters: the first template that reproduces the product wins, so specific before general.
REACTION_TYPES: tuple[ReactionType, ...] = (
    ReactionType(
        "Acylation of an alcohol with an anhydride",
        "a base or an acid catalyst",
        "[C:1](=[O:2])O[C]=O.[OH:3][#6:4]>>[C:1](=[O:2])[O:3][#6:4]",
    ),
    ReactionType(
        "Amide formation from an acid chloride",
        "a base to take up HCl",
        "[C:1](=[O:2])[Cl].[N;H1,H2;!$(N-C=O):3]>>[C:1](=[O:2])[N:3]",
    ),
    ReactionType(
        "Amide coupling",
        "a coupling agent and a base",
        "[C:1](=[O:2])[OH].[N;H1,H2;!$(N-C=O):3]>>[C:1](=[O:2])[N:3]",
    ),
    ReactionType(
        "Esterification",
        "an acid catalyst, or a coupling agent",
        "[C:1](=[O:2])[OH].[OH:3][#6:4]>>[C:1](=[O:2])[O:3][#6:4]",
    ),
    ReactionType(
        "Ester hydrolysis",
        "water with an acid or a base",
        "[C:1](=[O:2])[O:5][CX4]>>[C:1](=[O:2])[O:5]",
    ),
    ReactionType(
        "Decarboxylation",
        "usually none (loses CO2)",
        "[C;!$(C=O):1][C](=O)[OH]>>[C:1]",
    ),
    ReactionType(
        "Friedel-Crafts acylation",
        "a Lewis acid",
        "[cH:1].[C:2](=[O:3])[Cl]>>[c:1][C:2]=[O:3]",
    ),
    ReactionType(
        "Suzuki coupling",
        "a palladium catalyst and a base",
        "[c:1][Cl,Br,I].[c:2][B]>>[c:1][c:2]",
    ),
    ReactionType(
        "N-arylation (SNAr or Buchwald-Hartwig)",
        "a base; a palladium catalyst if the aryl halide is not activated",
        "[c:1][F,Cl,Br,I].[N;H1,H2;!$(N-C=O):2]>>[c:1][N:2]",
    ),
    ReactionType(
        "Alpha-arylation of a carbonyl compound",
        "a base; sometimes a palladium catalyst",
        "[c:1][F,Cl,Br,I].[CH1,CH2:2][C:3]=[O:4]>>[c:1][C:2][C:3]=[O:4]",
    ),
    ReactionType(
        "O-alkylation (Williamson ether synthesis)",
        "a base",
        "[CX4:1][Cl,Br,I].[OH:2][#6:3]>>[CX4:1][O:2][#6:3]",
    ),
    ReactionType(
        "N-alkylation",
        "a base",
        "[CX4:1][Cl,Br,I].[N;H1,H2;!$(N-C=O):2]>>[CX4:1][N:2]",
    ),
    ReactionType(
        "Reductive amination",
        "a hydride reducing agent",
        "[CX3;!$(C-[OX2,NX3]):1]=O.[N;H1,H2;!$(N-C=O):3]>>[C:1][N:3]",
    ),
    ReactionType(
        "Reduction of an aryl ketone to a methylene group",
        "a Clemmensen or Wolff-Kishner type reduction",
        "[c:1][C:2](=O)[#6:3]>>[c:1][CH2:2][#6:3]",
    ),
    ReactionType(
        "Reduction of a nitro group to an amine",
        "hydrogen with a metal catalyst, or a metal in acid",
        "[c:1][N+](=O)[O-]>>[c:1][NH2]",
    ),
    ReactionType(
        "Reduction of a carbonyl to an alcohol",
        "a hydride reducing agent",
        "[CX3;!$(C-[OX2,NX3,F,Cl,Br,I]):1]=[O:2]>>[C:1][O:2]",
    ),
)


def _plain(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _reproduces(reaction_type: ReactionType, mols: list[Chem.Mol], wanted: str) -> bool:
    reaction = AllChem.ReactionFromSmarts(reaction_type.smarts)
    if reaction.GetNumReactantTemplates() != len(mols):
        return False
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


def classify_step(reactants: list[str], product: str) -> ReactionType | None:
    """The first named reaction type that turns `reactants` into `product`, else None."""
    target = Chem.MolFromSmiles(product)
    mols = [Chem.MolFromSmiles(s) for s in reactants]
    if target is None or any(m is None for m in mols):
        return None
    wanted = _plain(target)
    return next((t for t in REACTION_TYPES if _reproduces(t, mols, wanted)), None)
