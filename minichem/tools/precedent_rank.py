"""Ranking of precedent reactions: fingerprint similarity, then a Pareto selection.

The method is the one of AstraZeneca's PrecedentFinder (MolecularAI/precedent_finder,
Apache-2.0): reactions of the same reaction type are compared with the query by a whole-reaction
fingerprint and a reaction-centre fingerprint, the near-best ones are reduced to the Pareto front
over (similarity, centre similarity, yield, recency), and the front is sorted by a weighted
distance to the ideal point. It is a ranking heuristic, not a probability.
This module is a rewrite for sparse matrices (the original densifies and loops over rows in
Python); the formulas are the same. It needs numpy, scipy and paretoset, which exist only in
the precedent environment, so it is imported only by `precedent_search.py`.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
from paretoset import paretoset
from scipy import sparse

RELATIVE_CUTOFF = 0.5  # squared distance to the best similarities that a candidate may have
RANK_WEIGHTS = np.array([1.0, 1.0, 0.8, 0.2])  # similarity, centre similarity, yield, recency


def _jaccard(inter: np.ndarray, count_a: np.ndarray, count_b: np.ndarray | float) -> np.ndarray:
    """Jaccard similarity from intersection and set sizes; two empty sets count as equal."""
    union = count_a + count_b - inter
    return np.divide(inter, union, out=np.ones_like(inter, dtype=float), where=union > 0)


def _binary(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Pattern of non-zero entries as float32 (scipy's 'jaccard' also reads vectors as bool)."""
    matrix = matrix.copy()
    matrix.eliminate_zeros()
    matrix.data = np.ones_like(matrix.data, dtype=np.float32)
    return matrix


def global_similarity(rows: sparse.csr_matrix, query: np.ndarray) -> np.ndarray:
    """Similarity between each stored reaction fingerprint and the query fingerprint.

    The fingerprints hold signed counts (product minus reactants). As in scipy's Jaccard
    distance on numeric vectors, only positions that are non-zero in at least one vector count,
    and a position matches only if both values are equal.
    """
    query = np.asarray(query)
    rows = rows.tocsr()
    rows.eliminate_zeros()
    where = np.flatnonzero(query)
    chosen = rows[:, where].toarray()
    both = (chosen != 0).sum(axis=1)
    equal = (chosen == query[where]).sum(axis=1)
    union = rows.getnnz(axis=1) + len(where) - both
    return np.divide(equal, union, out=np.ones(len(union)), where=union > 0)


def centre_similarity(rows: sparse.csr_matrix, query: np.ndarray, numbits: int) -> np.ndarray:
    """Best pairing of reaction-centre fingerprints (mean Jaccard); 0 if the counts differ.

    A stored row holds the number of centres first, then that many `numbits`-long fingerprints.
    """
    count = len(query)
    result = np.zeros(rows.shape[0])
    if count == 0:
        return result
    counts = np.asarray(rows[:, 0].todense()).ravel()
    same = np.where(counts == count)[0]
    if same.size == 0:
        return result
    chosen = rows[same]
    query_bits = (np.asarray(query) != 0).astype(np.float32)
    pair = np.empty((same.size, count, count))
    for j in range(count):
        block = _binary(chosen[:, 1 + j * numbits : 1 + (j + 1) * numbits])
        inter = np.asarray(block @ query_bits.T)
        pair[:, j, :] = _jaccard(inter, block.getnnz(axis=1)[:, None], query_bits.sum(axis=1))
    order = np.arange(count)
    best = np.max([pair[:, order, list(p)].sum(axis=1) for p in permutations(order)], axis=0)
    result[same] = best / count
    return result


def select_best(
    sim: np.ndarray, centre: np.ndarray, yields: np.ndarray, years: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray]:
    """Indices (into the inputs) of the best reactions and their goodness, best first."""
    if sim.size == 0:
        return np.array([], dtype=int), np.array([])
    pair = np.column_stack([sim, centre])
    distance = ((pair - pair.max(axis=0)) ** 2).sum(axis=1)
    keep = np.where(distance < RELATIVE_CUTOFF)[0]
    if keep.size == 0:
        keep = np.where(distance < min(1.5 * RELATIVE_CUTOFF, 1.0))[0]
    if keep.size == 0:
        return np.array([], dtype=int), np.array([])
    span = years.max() - years.min()
    recency = (years - years.min()) / span if span > 0 else np.ones_like(years, dtype=float)
    data = np.column_stack([sim, centre, yields / 100.0, recency])[keep]
    front = np.where(paretoset(data, sense=["max"] * 4, distinct=False))[0]
    # Same expressions as the original (weights multiply the squared gaps; the scale uses squared
    # weights), so goodness values match PrecedentFinder's and can be slightly below zero.
    gap = np.sqrt((RANK_WEIGHTS * (1.0 - data[front]) ** 2).sum(axis=1))
    scale = np.sqrt((RANK_WEIGHTS**2).sum())
    goodness = (scale - gap) / scale
    top = np.argsort(-goodness, kind="stable")[:limit]
    return keep[front[top]], goodness[top]
