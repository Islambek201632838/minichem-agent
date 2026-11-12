from minichem.tools.synthesis import sa_scores, structural_flags
from tests.conftest import ASPIRIN, IBUPROFEN


def test_sa_scores_are_in_range_and_labelled():
    results = sa_scores([ASPIRIN, IBUPROFEN]).results
    assert all(1.0 <= r.sa_score <= 10.0 for r in results)
    assert results[0].sa_score == 1.58
    assert results[0].label == "easy"


def test_sa_reports_invalid_input_per_item():
    result = sa_scores(["bad", ASPIRIN])
    assert len(result.results) == 1
    assert result.errors[0].input == "bad"


def test_flags_detect_macrocycle():
    flags = structural_flags(["C1CCCCCCCCCCC1"]).results[0]
    assert flags.largest_ring_size == 12
    assert any("macrocycle" in c for c in flags.concerns)


def test_flags_count_unspecified_stereocentres():
    flags = structural_flags([IBUPROFEN]).results[0]
    assert flags.n_stereocenters == 1
    assert flags.n_unspecified_stereocenters == 1


def test_flags_report_brenk_alert_for_aspirin_phenol_ester():
    assert "phenol_ester" in structural_flags([ASPIRIN]).results[0].alerts


def test_flags_invalid_input():
    assert structural_flags(["bad"]).errors
