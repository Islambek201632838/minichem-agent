"""Analogue generation with REINVENT 4 Mol2Mol (optional, OFF by default).

Mol2Mol is a transformer trained on pairs of similar molecules from ChEMBL. Given one molecule it
samples new molecules; unlike `analogues.py` (fixed edit rules, deterministic) it is stochastic
and it has no notion of activity: it proposes molecules that look like the input. Sampling is
seeded, so the same seed on the same machine gives the same list, and the seed is part of every
result.

`select_candidates` (filtering, similarity, ranking) is plain RDKit and is tested without REINVENT.
`ReinventMol2Mol` needs REINVENT 4 and a downloaded prior; it is imported only by
`mcp_server/reinvent_server.py`, which runs in REINVENT's own virtualenv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, Protocol

from pydantic import BaseModel, Field
from rdkit import Chem

from minichem.tools import chem
from minichem.tools.models import Constraints, ToolError

PRIOR_FILE = "mol2mol_medium_similarity.prior"
MODEL_NAME = "REINVENT 4 Mol2Mol (mol2mol_medium_similarity)"
DEFAULT_MIN_SIMILARITY = 0.4
DEFAULT_SEED = 0
MAX_MOLECULES = 30
MIN_SAMPLES = 100  # molecules drawn from the model; many are duplicates or fall below the floor
SAMPLES_PER_RESULT = 4


class RawSample(NamedTuple):
    """One sampled molecule: its SMILES and the model's negative log-likelihood for it."""

    smiles: str
    nll: float


class GeneratedMolecule(BaseModel):
    smiles: str = Field(description="Canonical SMILES")
    tanimoto: float = Field(description="Morgan radius 2, 2048 bits, to the query")
    nll: float = Field(
        description="Model negative log-likelihood given the query; lower = likelier"
    )
    mw: float
    logp: float
    hbd: int
    hba: int


class GenerationResponse(BaseModel):
    """Fields that the supervisor must see come first: its view of tool results is truncated."""

    query: str
    seed: int
    requested: int
    returned: int
    molecules: list[GeneratedMolecule]
    model: str = MODEL_NAME
    note: str = ""


class Mol2MolGenerator(Protocol):
    """Anything that can draw raw samples for one molecule."""

    def sample(self, smiles: str, num_samples: int, seed: int) -> list[RawSample]: ...


def samples_needed(n: int) -> int:
    """How many molecules to draw so that `n` survive the similarity floor and constraints."""
    return max(MIN_SAMPLES, SAMPLES_PER_RESULT * n)


def select_candidates(
    query: Chem.Mol,
    raw: list[RawSample],
    n: int,
    min_similarity: float,
    constraints: Constraints,
) -> list[GeneratedMolecule]:
    """Keep valid, new, distinct molecules above the floor that meet the constraints.

    Ranked by similarity to the query (then SMILES) so that the output order does not depend on
    the sampling order. Similarity is recomputed here with the same fingerprint as the project's
    `tanimoto_similarity` tool, not taken from REINVENT.
    """
    query_smiles = Chem.MolToSmiles(query)
    best: dict[str, GeneratedMolecule] = {}
    for sample in raw:
        mol = chem.parse(sample.smiles)
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(mol)
        similarity = chem.similarity_value(query, mol)
        props = chem.properties_of(mol)
        if smiles == query_smiles or similarity < min_similarity:
            continue
        if not chem.satisfies(props, constraints):
            continue
        candidate = GeneratedMolecule(
            smiles=smiles,
            tanimoto=round(similarity, 3),
            nll=round(sample.nll, 2),
            mw=props.mw,
            logp=props.logp,
            hbd=props.hbd,
            hba=props.hba,
        )
        if smiles not in best or candidate.nll < best[smiles].nll:
            best[smiles] = candidate
    ranked = sorted(best.values(), key=lambda m: (-m.tanimoto, m.smiles))
    return ranked[:n]


def generate(
    generator: Mol2MolGenerator,
    smiles: str,
    n: int = 10,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    seed: int = DEFAULT_SEED,
    constraints: Constraints | None = None,
) -> GenerationResponse | ToolError:
    """Sample Mol2Mol analogues of `smiles` and return the best `n` after filtering."""
    query = chem.parse(smiles)
    if query is None:
        return ToolError(
            error=f"Query is not a valid SMILES: {smiles!r}",
            hint="Validate or canonicalise the SMILES first.",
        )
    constraints = constraints or Constraints()
    raw = generator.sample(Chem.MolToSmiles(query), samples_needed(n), seed)
    chosen = select_candidates(query, raw, n, min_similarity, constraints)
    note = ""
    if len(chosen) < n:
        note = (
            f"Only {len(chosen)} of {len(raw)} sampled molecules were new, valid, at least "
            f"{min_similarity} similar and within the constraints. Lower min_similarity, relax "
            "the constraints or use another seed to get more."
        )
    return GenerationResponse(
        query=Chem.MolToSmiles(query),
        seed=seed,
        requested=n,
        returned=len(chosen),
        molecules=chosen,
        note=note,
    )


class MockMol2Mol:
    """Test double: returns a fixed list and records the calls it received."""

    def __init__(self, samples: list[RawSample]) -> None:
        self._samples = samples
        self.calls: list[tuple[str, int, int]] = []

    def sample(self, smiles: str, num_samples: int, seed: int) -> list[RawSample]:
        self.calls.append((smiles, num_samples, seed))
        return list(self._samples)


class ReinventMol2Mol:
    """Mol2Mol sampling with a locally installed REINVENT 4 (multinomial, temperature 1).

    The prior is loaded on the first sample, not at construction, so the MCP handshake stays
    fast. Uses REINVENT's own `create_adapter`, `setup_sampler` and `set_seed`, the calls its
    `sampling` run mode makes; no files are written. The seed fixes torch, numpy and `random`.
    """

    def __init__(self, prior_dir: Path, device: str = "mps") -> None:
        self._prior = prior_dir / PRIOR_FILE
        self._device = device
        self._adapter = None

    def _load_adapter(self):
        import torch
        from reinvent.runmodes import create_adapter

        logging.getLogger("reinvent").setLevel(logging.WARNING)  # it logs the whole network at INFO
        torch.set_default_device(self._device)
        adapter, _, _ = create_adapter(str(self._prior), "inference", torch.device(self._device))
        return adapter

    def sample(self, smiles: str, num_samples: int, seed: int) -> list[RawSample]:
        """Draw `num_samples` molecules for one input; invalid and duplicate ones are dropped."""
        if not self._prior.is_file():
            raise FileNotFoundError(f"Mol2Mol prior not found: {self._prior}")
        import torch
        from reinvent.models.model_factory.sample_batch import SmilesState
        from reinvent.runmodes.setup_sampler import setup_sampler
        from reinvent.utils import set_seed

        if self._adapter is None:
            self._adapter = self._load_adapter()
        config = {
            "batch_size": num_samples,
            "randomize_smiles": False,
            "temperature": 1.0,
            "sample_strategy": "multinomial",
            "isomeric_smiles": True,
        }
        sampler, _ = setup_sampler("Mol2Mol", config, self._adapter)
        set_seed(seed)
        with torch.no_grad():
            batch = sampler.sample([smiles])
        nlls = batch.nlls.cpu().tolist()
        return [
            RawSample(s, nll)
            for s, state, nll in zip(batch.smilies, batch.states, nlls, strict=True)
            if state == SmilesState.VALID
        ]
