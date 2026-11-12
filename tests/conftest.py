"""Shared constants and fixtures. No test touches the network or a real LLM."""

from __future__ import annotations

import pytest

ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "pubchem.json"
