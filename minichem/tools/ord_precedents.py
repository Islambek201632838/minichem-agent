"""Reaction precedents from the hosted Open Reaction Database (ORD), US patent records only.

The same job as `precedent_search.py`, without a local database: the product of the step is
searched in the ORD patent reactions (exact compound first, then similar products), the
records are fetched, and the candidates are ranked here with RDKit fingerprints. The records are
the text-mined USPTO extraction (Lowe), so they are NOT verified.

Safety: a record is read through `ParsedRecord` only. It takes the reaction SMILES (reactants,
reagents and solvents, product), the patent number, the dataset year and the yield. ORD records
also hold temperatures, times, amounts and a verbatim procedure text; those fields are never
read, so they cannot reach the model or the user.

What is sent to the hosted service: the product SMILES of the step (no reactants, no question).
The step is not atom-mapped and no reaction type is matched; the ranking compares fingerprints.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

from google.protobuf.message import DecodeError
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from minichem.ord_proto import reaction_pb2
from minichem.tools.models import ToolError
from minichem.tools.ord_client import OrdApiError, OrdClient
from minichem.tools.precedents import (
    MAX_PRECEDENTS,
    ORD_PROVENANCE,
    Precedent,
    PrecedentMolecule,
    PrecedentsResponse,
    move_unmapped_reactants,
    parse_record,
)

MAX_REACTANTS = 5
SEARCH_LIMIT = 40
MIN_EXACT_HITS = 10  # fewer exact-product reactions than this: also search similar products
SIMILAR_THRESHOLD = 0.7
_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True)
class ParsedRecord:
    """The only fields of an ORD reaction that this module reads."""

    reaction_id: str
    patent: str
    year: int
    yield_percent: float | None
    reactants: list[str]
    reagents: list[str]
    product: str


def _product_query(match: dict) -> dict:
    patent_only = {"op": "not_null", "path": "provenance.patent"}
    products = {"op": "exists", "path": "outcomes.products", "where": match}
    return {
        "query": {"where": {"op": "and", "clauses": [patent_only, products]}, "limit": SEARCH_LIMIT}
    }


def parse_ord_record(record: dict, years: dict[str, int]) -> ParsedRecord | None:
    """Read one API record; None if it is not a usable patent reaction."""
    try:
        reaction = reaction_pb2.Reaction.FromString(base64.b64decode(record["binpb"]))
    except (KeyError, binascii.Error, DecodeError):
        return None
    year = years.get(record.get("dataset_id") or "")
    extracted = next(
        (
            i.value
            for i in reaction.identifiers
            if i.type == reaction_pb2.ReactionIdentifier.REACTION_CXSMILES
        ),
        "",
    )
    if not year or not reaction.provenance.patent or extracted.count(">") != 2:
        return None
    try:
        reactants, reagents, product = parse_record(move_unmapped_reactants(extracted))
    except ValueError:
        return None
    if not reactants or not product:
        return None
    return ParsedRecord(
        reaction_id=reaction.reaction_id,
        patent=reaction.provenance.patent,
        year=year,
        yield_percent=_yield_percent(reaction),
        reactants=reactants,
        reagents=reagents,
        product=product,
    )


def _yield_percent(reaction) -> float | None:
    for outcome in reaction.outcomes:
        for product in outcome.products:
            for measurement in product.measurements:
                if measurement.type == reaction_pb2.ProductMeasurement.YIELD and (
                    measurement.HasField("percentage")
                ):
                    return round(measurement.percentage.value, 1)
    return None


def _fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else _FPGEN.GetFingerprint(mol)


def _best_match(query_fp, smiles_list: list[str]) -> float:
    fps = [fp for s in smiles_list if (fp := _fingerprint(s)) is not None]
    return max(DataStructs.BulkTanimotoSimilarity(query_fp, fps), default=0.0) if fps else 0.0


def _score(record: ParsedRecord, product_fp, reactant_fps: list) -> tuple[float, float]:
    """(product similarity, mean best-match similarity of the query reactants)."""
    product_sim = _best_match(product_fp, record.product.split("."))
    reactant_sim = sum(_best_match(fp, record.reactants) for fp in reactant_fps) / len(reactant_fps)
    return product_sim, reactant_sim


class OrdPrecedents:
    """Precedents from the hosted ORD API. Never raises: failures come back as `ToolError`."""

    def __init__(self, client: OrdClient) -> None:
        self._client = client
        self._cache: dict[tuple[str, tuple[str, ...]], PrecedentsResponse] = {}

    def find(
        self, product: str, reactants: list[str], max_precedents: int
    ) -> PrecedentsResponse | ToolError:
        """Precedents for `reactants -> product`, best first (at most `max_precedents`)."""
        product_fp = _fingerprint(product)
        reactant_fps = [_fingerprint(s) for s in reactants]
        if product_fp is None or any(fp is None for fp in reactant_fps):
            return ToolError(error="Invalid SMILES.", hint="Pass molecules from tool results.")
        if len(reactants) > MAX_REACTANTS:
            return ToolError(error=f"At most {MAX_REACTANTS} reactants are supported.")
        key = (product, tuple(reactants))
        if key not in self._cache:
            found = self._search(product, product_fp, reactant_fps)
            if isinstance(found, ToolError):
                return found
            self._cache[key] = found
        response = self._cache[key]
        return response.model_copy(
            update={"precedents": response.precedents[: min(max_precedents, MAX_PRECEDENTS)]}
        )

    def _search(
        self, product: str, product_fp, reactant_fps: list
    ) -> PrecedentsResponse | ToolError:
        try:
            ids = self._client.search_ids(
                _product_query({"op": "same_compound", "path": "smiles", "smiles": product})
            )
            if len(ids) < MIN_EXACT_HITS:
                similar = _product_query(
                    {"op": "similarity", "path": "smiles", "smiles": product,
                     "threshold": SIMILAR_THRESHOLD}
                )  # fmt: skip
                try:
                    ids = list(dict.fromkeys([*ids, *self._client.search_ids(similar)]))
                except OrdApiError:
                    if not ids:
                        raise
            if not ids:
                return ToolError(
                    error="No patent reaction with this product (or a similar one) in ORD.",
                    hint="Report that no precedent was found; do not guess conditions.",
                )
            years = self._client.dataset_years()
            records = [
                parsed
                for raw in self._client.reactions(ids)
                if (parsed := parse_ord_record(raw, years))
            ]
        except OrdApiError as error:
            return ToolError(
                error=str(error),
                hint="Precedents are unavailable now; say so and do not guess conditions.",
            )
        return self._rank(records, product_fp, reactant_fps)

    @staticmethod
    def _rank(
        records: list[ParsedRecord], product_fp, reactant_fps: list
    ) -> PrecedentsResponse | ToolError:
        if not records:
            return ToolError(
                error="ORD returned reactions, but none could be read as a patent step.",
                hint="Report that no precedent was found; do not guess conditions.",
            )
        scored = []
        for record in records:
            product_sim, reactant_sim = _score(record, product_fp, reactant_fps)
            scored.append((round((product_sim + reactant_sim) / 2, 3), record))
        scored.sort(
            key=lambda s: (-s[0], -(s[1].yield_percent or -1.0), -s[1].year, s[1].reaction_id)
        )
        return PrecedentsResponse(
            provenance=ORD_PROVENANCE,
            n_candidates=len(records),
            precedents=[_precedent(record, sim) for sim, record in scored[:MAX_PRECEDENTS]],
        )


def _precedent(record: ParsedRecord, similarity: float) -> Precedent:
    return Precedent(
        patent=record.patent,
        reaction_id=record.reaction_id,
        year=record.year,
        yield_percent=record.yield_percent,
        similarity=similarity,
        reactants=[PrecedentMolecule(smiles=s) for s in record.reactants],
        reagents_and_solvents=[PrecedentMolecule(smiles=s) for s in record.reagents],
        product=PrecedentMolecule(smiles=record.product),
    )
