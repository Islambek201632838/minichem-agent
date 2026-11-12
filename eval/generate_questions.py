"""Generate golden-style questions from templates, with several phrasings per template.

uv run python -m eval.generate_questions [--seed 0] [--count 40]    writes eval/generated.yaml
uv run python -m eval.run_eval --file eval/generated.yaml --runs 1

Expected values are computed here with RDKit (never taken from a model), the entities come from
the offline compound table so `--mock-tools` works, and the output depends only on the seed.
Phrasing levels: plain, polite, verbose, jargon, Russian (with "use a decimal point", because the
number check does not read decimal commas), and a lowercase fragment.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import yaml

from minichem.tools import chem, synthesis
from minichem.tools.pubchem import OFFLINE_COMPOUNDS

OUTPUT_PATH = Path(__file__).resolve().parent / "generated.yaml"
# Entities: distinct molecules only (paracetamol and acetaminophen would be one molecule twice).
NAMES = ["ethanol", "aspirin", "ibuprofen", "paracetamol", "caffeine", "naproxen", "benzene"]
SIMILARITY_TOL = 0.005
RU_NOTE = " Числа пиши с точкой."

PROPERTIES: dict[str, dict[str, Any]] = {
    "mw": {"en": "molecular weight", "short": "MW", "ru": "молекулярную массу", "tol": 0.05},
    "logp": {"en": "logP", "short": "logP", "ru": "logP", "tol": 0.05},
    "tpsa": {"en": "TPSA", "short": "TPSA", "ru": "TPSA", "tol": 0.1},
    "qed": {"en": "QED", "short": "QED", "ru": "QED", "tol": 0.005},
}

PROPERTY_PHRASINGS = [
    "What is the {en} of {a}?",
    "Could you please tell me the {en} of {a}?",
    "I am screening small molecules for a project and need one number for {a}: its {en}.",
    "{short} of {a}?",
    "Посчитай {ru} для {a}." + RU_NOTE,
    "{a} {short}",
]
SIMILARITY_PHRASINGS = [
    "How similar are {a} and {b} by Tanimoto similarity?",
    "Please compute the Tanimoto similarity between {a} and {b}.",
    "I want to know whether {a} and {b} look alike structurally; give me the Tanimoto number.",
    "Tanimoto {a} vs {b}",
    "Какое сходство Танимото у {a} и {b}?" + RU_NOTE,
    "similarity of {a} to {b}",
]
SA_PHRASINGS = [
    "Which is harder to synthesise, {a} or {b}? Give the SA scores.",
    "Could you compare the synthetic accessibility of {a} and {b}?",
    "For planning purposes I need to know which of {a} and {b} is more difficult to make, "
    "with the numbers behind it.",
    "SA score {a} and {b}, which is higher",
    "Что сложнее синтезировать: {a} или {b}? Покажи SA score." + RU_NOTE,
    "sa scores for {a} and {b}",
]


def _smiles(name: str) -> str:
    smiles, _ = OFFLINE_COMPOUNDS[name]
    return chem.canonical(smiles) or smiles


def property_entry(rng: random.Random) -> dict[str, Any]:
    key = rng.choice(sorted(PROPERTIES))
    spec = PROPERTIES[key]
    while True:
        name = rng.choice(NAMES)
        value = getattr(chem.calculate_properties([_smiles(name)]).results[0], key)
        if abs(value) > spec["tol"]:  # a zero (benzene TPSA) would match any stray "0"
            break
    phrasing = rng.choice(PROPERTY_PHRASINGS)
    return {
        "question": phrasing.format(a=name, en=spec["en"], short=spec["short"], ru=spec["ru"]),
        "expected_tools": {
            "complete": [["name_to_smiles", "calculate_properties"]],
            "partial": [["name_to_smiles"]],
        },
        "checks": {"numbers": [{"value": round(value, 3), "tol": spec["tol"]}]},
    }


def similarity_entry(rng: random.Random) -> dict[str, Any]:
    while True:
        a, b = rng.sample(NAMES, 2)
        hit = chem.tanimoto_similarity(_smiles(a), [_smiles(b)]).results[0]
        if hit.tanimoto > SIMILARITY_TOL:  # zero overlap would match any stray "0"
            break
    return {
        "question": rng.choice(SIMILARITY_PHRASINGS).format(a=a, b=b),
        "expected_tools": {
            "complete": [["name_to_smiles", "tanimoto_similarity"]],
            "partial": [["name_to_smiles"]],
        },
        "checks": {"numbers": [{"value": round(hit.tanimoto, 3), "tol": SIMILARITY_TOL}]},
    }


def sa_entry(rng: random.Random) -> dict[str, Any]:
    a, b = rng.sample(NAMES, 2)
    scores = synthesis.sa_scores([_smiles(a), _smiles(b)]).results
    return {
        "question": rng.choice(SA_PHRASINGS).format(a=a, b=b),
        "expected_tools": {
            "complete": [["name_to_smiles", "sa_score"]],
            "partial": [["name_to_smiles"]],
        },
        "checks": {
            "numbers": [{"value": round(s.sa_score, 2), "tol": 0.05} for s in scores],
        },
    }


BUILDERS = [property_entry, similarity_entry, sa_entry]


def generate(seed: int, count: int) -> list[dict[str, Any]]:
    """`count` entries cycling through the templates; same seed, same file."""
    rng = random.Random(seed)
    entries = []
    for n in range(count):
        entry = BUILDERS[n % len(BUILDERS)](rng)
        entries.append({"id": f"g{n + 1:03d}", "category": "generated", **entry})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate templated evaluation questions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    header = (
        f"# Generated by eval/generate_questions.py (seed {args.seed}, {args.count} entries).\n"
        "# Do not edit by hand: change the generator and regenerate.\n"
    )
    body = yaml.safe_dump(
        generate(args.seed, args.count), sort_keys=False, allow_unicode=True, width=100
    )
    args.output.write_text(header + body, encoding="utf-8")
    print(f"Wrote {args.count} questions to {args.output}")


if __name__ == "__main__":
    main()
