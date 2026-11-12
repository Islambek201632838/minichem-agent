from eval.generate_questions import OUTPUT_PATH, generate
from eval.run_eval import load_golden
from minichem.tools.pubchem import OFFLINE_COMPOUNDS


def test_generation_is_deterministic_and_seed_dependent():
    assert generate(0, 12) == generate(0, 12)
    assert generate(0, 12) != generate(1, 12)


def test_committed_file_matches_the_generator():
    committed = load_golden(OUTPUT_PATH)
    expected = generate(0, len(committed))
    for entry in expected:
        entry["turns"] = [entry["question"]]
    assert committed == expected


def test_every_entity_is_known_offline_and_every_expected_value_is_nonzero():
    entries = generate(3, 60)
    assert len({e["id"] for e in entries}) == 60
    for entry in entries:
        assert any(name in entry["question"] for name in OFFLINE_COMPOUNDS), entry["question"]
        for check in entry["checks"]["numbers"]:
            assert abs(check["value"]) > check["tol"]
