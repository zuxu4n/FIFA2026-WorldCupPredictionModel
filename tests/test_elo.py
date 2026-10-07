import pytest
from conftest import make_results

from wcpred import config as C
from wcpred.features.elo import compute_elo, expected_score, goal_diff_multiplier, k_factor


@pytest.mark.parametrize(
    ("gd", "mult"), [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 1.5), (3, 1.75), (-4, 1.875)]
)
def test_goal_diff_multiplier(gd, mult):
    assert goal_diff_multiplier(gd) == pytest.approx(mult)


def test_k_factor_scales_with_importance():
    assert k_factor(1.0) == pytest.approx(20.0)  # friendly
    assert k_factor(4.0) == pytest.approx(80.0)  # World Cup


def test_expected_score_symmetry():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1700, 1500) + expected_score(1500, 1700) == pytest.approx(1.0)


def test_single_neutral_friendly_win():
    elo = compute_elo(
        make_results(
            [
                ("2020-01-01", "A", "B", 1, 0, "Friendly", "X", "Y", True),
                ("2020-02-01", "A", "B", None, None, "Friendly", "X", "Y", True),
            ]
        )
    )
    # K = 20, expected 0.5, actual 1 -> +10 / -10
    assert elo.pre_home[0] == elo.pre_away[0] == C.ELO_START  # snapshot before the match
    assert elo.pre_home[1] == pytest.approx(C.ELO_START + 10)
    assert elo.pre_away[1] == pytest.approx(C.ELO_START - 10)
    assert elo.final == {"A": pytest.approx(1510), "B": pytest.approx(1490)}


def test_home_advantage_means_a_home_win_earns_less():
    neutral = compute_elo(
        make_results([("2020-01-01", "A", "B", 1, 0, "Friendly", "X", "Y", True)])
    )
    at_home = compute_elo(
        make_results([("2020-01-01", "A", "B", 1, 0, "Friendly", "X", "A", False)])
    )
    assert C.ELO_START < at_home.final["A"] < neutral.final["A"]


def test_updates_are_zero_sum_and_skip_unplayed():
    elo = compute_elo(
        make_results(
            [
                ("2020-01-01", "A", "B", 3, 0, "FIFA World Cup", "X", "Y", True),
                ("2020-01-05", "B", "C", 1, 1, "Friendly", "X", "B", False),
                ("2020-01-09", "C", "A", None, None, "Friendly", "X", "Y", True),
            ]
        )
    )
    assert sum(elo.final.values()) == pytest.approx(3 * C.ELO_START)
    assert elo.pre_home[2] == pytest.approx(elo.final["C"])  # unplayed row sees latest rating
