"""Ranking maths against the reference implementations (scipy, rxnutils).

These packages exist only in the precedent environment, so the tests are skipped in the project
environment. Run them there:
`PYTHONPATH=. <precedents-python> -m pytest tests/test_precedent_rank.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("paretoset")
pytest.importorskip("rxnutils")
from scipy import sparse  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

from minichem.tools.precedent_rank import (  # noqa: E402
    centre_similarity,
    global_similarity,
    select_best,
)

rng = np.random.default_rng(0)


def test_global_similarity_equals_scipy_jaccard_on_signed_counts():
    rows = rng.integers(-3, 4, size=(60, 40)) * (rng.random((60, 40)) < 0.2)
    query = rng.integers(-3, 4, size=40) * (rng.random(40) < 0.3)
    expected = 1 - cdist(rows, query.reshape(1, -1), "jaccard").ravel()
    got = global_similarity(sparse.csr_matrix(rows.astype(np.int8)), query)
    np.testing.assert_allclose(got, expected)


def test_centre_similarity_equals_reference_for_equal_and_unequal_counts():
    from rxnutils.chem.features.reaction_centre_fp import reaction_center_similarity

    bits = 16
    query = (rng.random((2, bits)) < 0.4).astype(np.int8)
    stored, expected = [], []
    for count in [2, 2, 1, 3, 2]:
        centres = (rng.random((count, bits)) < 0.4).astype(np.int8)
        stored.append(np.concatenate([[count], centres.ravel(), np.zeros((3 - count) * bits)]))
        expected.append(reaction_center_similarity(query.tolist(), centres.tolist()))
    got = centre_similarity(sparse.csr_matrix(np.array(stored, dtype=np.int8)), query, bits)
    np.testing.assert_allclose(got, expected)
    assert got[2] == got[3] == 0.0


def test_select_best_puts_the_exact_recent_high_yield_match_first():
    sim = np.array([1.0, 0.6, 0.9, 0.2])
    centre = np.array([1.0, 0.5, 1.0, 0.1])
    yields = np.array([90.0, 90.0, 40.0, 90.0])
    years = np.array([2015.0, 2015.0, 1990.0, 2015.0])
    indices, goodness = select_best(sim, centre, yields, years, limit=3)
    assert indices[0] == 0
    assert 3 not in indices
    assert list(goodness) == sorted(goodness, reverse=True)


def test_select_best_handles_one_year_and_empty_input():
    ones = np.ones(2)
    indices, _ = select_best(ones, ones, np.array([50.0, 60.0]), np.array([2000.0, 2000.0]), 2)
    assert len(indices) >= 1
    empty, goodness = select_best(*(np.array([]),) * 4, limit=3)
    assert len(empty) == 0
    assert len(goodness) == 0
