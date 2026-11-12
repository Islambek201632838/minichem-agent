"""PubChem PUG REST lookups with a small on-disk cache, plus an offline fixture backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import requests

from minichem.tools import chem
from minichem.tools.models import NameLookup, ToolError

PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
NAME_PROPERTIES = "SMILES,Title,IUPACName,MolecularFormula"


class ChemLookup(Protocol):
    """The two lookups the utility agent needs."""

    def name_to_smiles(self, name: str) -> NameLookup | ToolError: ...

    def smiles_to_name(self, smiles: str) -> NameLookup | ToolError: ...


class PubChemClient:
    """Live PubChem client. Never raises: failures come back as `ToolError`."""

    def __init__(
        self,
        cache_path: Path,
        timeout_s: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        self._cache_path = cache_path
        self._timeout = timeout_s
        self._session = session or requests.Session()
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict]:
        if not self._cache_path.exists():
            return {}
        try:
            return json.loads(self._cache_path.read_text())
        except json.JSONDecodeError:
            return {}  # a corrupt cache only costs us a refetch

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, indent=1, sort_keys=True))
        os.replace(tmp, self._cache_path)

    def _request(self, method: str, url: str, **kwargs) -> dict | ToolError:
        try:
            response = self._session.request(method, url, timeout=self._timeout, **kwargs)
        except requests.Timeout:
            return ToolError(
                error=f"PubChem timed out after {self._timeout}s.",
                hint="Retry once; if it fails again, tell the user PubChem is unavailable.",
            )
        except requests.RequestException as exc:
            return ToolError(error=f"PubChem request failed: {exc}", hint="Retry once.")
        if response.status_code == 404:
            return ToolError(
                error="PubChem has no compound matching that input.",
                hint="Try a more common synonym or the generic (non-brand) name.",
            )
        if response.status_code != 200:
            return ToolError(
                error=f"PubChem returned HTTP {response.status_code}.",
                hint="Retry once; PubChem rate-limits bursts.",
            )
        try:
            return response.json()["PropertyTable"]["Properties"][0]
        except (ValueError, KeyError, IndexError):
            return ToolError(error="Unexpected PubChem response format.")

    @staticmethod
    def _to_lookup(query: str, props: dict) -> NameLookup:
        return NameLookup(
            name=query,
            smiles=chem.canonical(props["SMILES"]) or props["SMILES"],
            title=props.get("Title"),
            iupac_name=props.get("IUPACName"),
            formula=props.get("MolecularFormula"),
        )

    def name_to_smiles(self, name: str) -> NameLookup | ToolError:
        """Resolve a compound name or synonym to a canonical SMILES."""
        key = f"name:{name.strip().lower()}"
        if key in self._cache:
            return NameLookup(**self._cache[key])
        url = f"{PUG_BASE}/name/{quote(name.strip(), safe='')}/property/{NAME_PROPERTIES}/JSON"
        props = self._request("GET", url)
        if isinstance(props, ToolError):
            return props
        lookup = self._to_lookup(name, props)
        self._cache[key] = lookup.model_dump()
        self._save_cache()
        return lookup

    def smiles_to_name(self, smiles: str) -> NameLookup | ToolError:
        """Find the PubChem title and IUPAC name for a SMILES."""
        if chem.parse(smiles) is None:
            return ToolError(error=f"Not a valid SMILES: {smiles!r}", hint="Validate it first.")
        key = f"smiles:{chem.canonical(smiles)}"
        if key in self._cache:
            return NameLookup(**self._cache[key])
        # POST so characters such as '#', '/' and '\' need no URL escaping.
        url = f"{PUG_BASE}/smiles/property/{NAME_PROPERTIES}/JSON"
        props = self._request("POST", url, data={"smiles": smiles})
        if isinstance(props, ToolError):
            return props
        lookup = self._to_lookup(smiles, props)
        self._cache[key] = lookup.model_dump()
        self._save_cache()
        return lookup


# Offline fixtures: SMILES are checked against RDKit in tests; names are common synonyms.
OFFLINE_COMPOUNDS: dict[str, tuple[str, str]] = {
    "ethanol": ("CCO", "Ethanol"),
    "aspirin": ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
    "ibuprofen": ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),
    "paracetamol": ("CC(=O)Nc1ccc(O)cc1", "Acetaminophen"),
    "acetaminophen": ("CC(=O)Nc1ccc(O)cc1", "Acetaminophen"),
    "caffeine": ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "Caffeine"),
    "naproxen": ("COc1ccc2cc(C(C)C(=O)O)ccc2c1", "Naproxen"),
    "benzene": ("c1ccccc1", "Benzene"),
    # Larger drugs (eval/complex.yaml). SMILES are PubChem isomeric SMILES fetched 2026-11-12;
    # formula and mass were compared with RDKit (tests/test_pubchem.py keeps the formulas).
    "imatinib": (
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "Imatinib",
    ),
    "nilotinib": (
        "CC1=C(C=C(C=C1)C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N3C=C(N=C3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "Nilotinib",
    ),
    "dasatinib": (
        "CC1=C(C(=CC=C1)Cl)NC(=O)C2=CN=C(S2)NC3=CC(=NC(=N3)C)N4CCN(CC4)CCO",
        "Dasatinib",
    ),
    "osimertinib": (
        "CN1C=C(C2=CC=CC=C21)C3=NC(=NC=C3)NC4=C(C=C(C(=C4)NC(=O)C=C)N(C)CCN(C)C)OC",
        "Osimertinib",
    ),
    "sildenafil": (
        "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
        "Sildenafil",
    ),
    "tofacitinib": ("C[C@@H]1CCN(C[C@@H]1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N", "Tofacitinib"),
    "ruxolitinib": ("C1CCC(C1)[C@@H](CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3", "Ruxolitinib"),
    "baricitinib": ("CCS(=O)(=O)N1CC(C1)(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3", "Baricitinib"),
    "paclitaxel": (
        "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)"
        "(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)"
        "OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
        "Paclitaxel",
    ),
    "strychnine": (
        "C1CN2CC3=CCO[C@H]4CC(=O)N5[C@H]6[C@H]4[C@H]3C[C@H]2[C@@]61C7=CC=CC=C75",
        "Strychnine",
    ),
}


class OfflinePubChem:
    """Deterministic backend for evals and demos without network access."""

    def name_to_smiles(self, name: str) -> NameLookup | ToolError:
        entry = OFFLINE_COMPOUNDS.get(name.strip().lower())
        if entry is None:
            return ToolError(
                error="Compound not in the offline fixture table.",
                hint="Offline mode only knows: " + ", ".join(sorted(OFFLINE_COMPOUNDS)),
            )
        smiles, title = entry
        return NameLookup(
            name=name,
            smiles=chem.canonical(smiles) or smiles,
            title=title,
            source="offline fixture",
        )

    def smiles_to_name(self, smiles: str) -> NameLookup | ToolError:
        target = chem.canonical(smiles)
        for name, (known, title) in OFFLINE_COMPOUNDS.items():
            if chem.canonical(known) == target:
                return NameLookup(
                    name=name, smiles=target or smiles, title=title, source="offline fixture"
                )
        return ToolError(error="Structure not in the offline fixture table.")
