import numpy as np
import pandas as pd
import pytest
from conftest import StubModel, make_results, team_strengths
from scipy.stats import spearmanr

from wcpred import config as C
from wcpred.data import DataError
from wcpred.simulation import simulate_world_cup

STAGES = ["P_R32", "P_R16", "P_QF", "P_SF", "P_Final", "P_champion"]


@pytest.fixture(scope="module")
def model(wc_format):
    return StubModel(team_strengths(wc_format.teams))


@pytest.fixture(scope="module")
def sim(results, wc_format, model):
    return simulate_world_cup(results, wc_format, model, n_runs=4000, seed=1).table


def test_stage_probabilities_sum_to_bracket_sizes(sim):
    totals = sim[["P_win_group", *STAGES]].sum()
    expected = [12, 32, 16, 8, 4, 2, 1]
    np.testing.assert_allclose(totals.to_numpy(), expected)


def test_probabilities_valid_and_monotone(sim):
    p = sim[STAGES].to_numpy()
    assert ((p >= 0) & (p <= 1)).all()
    assert (np.diff(p, axis=1) <= 1e-12).all()  # can't reach a later stage more often
    assert (sim["P_win_group"] <= sim["P_R32"]).all()


def test_stronger_teams_do_better(sim, model, wc_format):
    strength = sim["team"].map(model.strengths)
    assert sim.iloc[0]["team"] == max(model.strengths, key=model.strengths.get)
    assert spearmanr(strength, sim["P_champion"]).statistic > 0.9
    by_team = sim.set_index("team")
    for members in wc_format.groups.values():  # listed strongest to weakest
        assert by_team.loc[members, "P_win_group"].is_monotonic_decreasing


def test_seeded_runs_are_reproducible(results, wc_format, model):
    a = simulate_world_cup(results, wc_format, model, n_runs=300, seed=5).table
    b = simulate_world_cup(results, wc_format, model, n_runs=300, seed=5).table
    c = simulate_world_cup(results, wc_format, model, n_runs=300, seed=6).table
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_model_is_called_once_per_round_not_per_match(results, wc_format):
    model = StubModel(team_strengths(wc_format.teams))
    simulate_world_cup(results, wc_format, model, n_runs=500, seed=0)
    assert model.calls <= 1 + 6  # group stage + one batch per knockout round


def _completed_groups(wc_format, upsets: dict[str, tuple[int, int]] | None = None):
    """Every group played: the i-th listed team beats every later one (no ties)."""
    rows = []
    for members in wc_format.groups.values():
        for i in range(4):
            for j in range(i + 1, 4):
                rows.append(
                    (
                        C.WC2026_START,
                        members[i],
                        members[j],
                        j - i + 1,
                        0,
                        C.WORLD_CUP,
                        "Houston",
                        "United States",
                        True,
                    )
                )
    return rows


def test_played_results_are_fixed(wc_format, model):
    rows = _completed_groups(wc_format)
    a2, b2 = wc_format.groups["A"][1], wc_format.groups["B"][1]
    # match 73 is 2A v 2B in Inglewood on 2026-06-28; the weaker side wins it
    winner = min((a2, b2), key=model.strengths.get)
    rows.append(
        (
            "2026-06-28",
            a2,
            b2,
            int(winner == a2),
            int(winner == b2),
            C.WORLD_CUP,
            "Inglewood",
            "United States",
            True,
        )
    )
    table = simulate_world_cup(make_results(rows), wc_format, model, n_runs=500, seed=0).table
    by_team = table.set_index("team")
    for members in wc_format.groups.values():
        assert by_team.loc[members[0], "P_win_group"] == 1.0
        assert by_team.loc[members[3], "P_R32"] == 0.0
    loser = b2 if winner == a2 else a2
    assert by_team.loc[winner, "P_R16"] == 1.0
    assert by_team.loc[loser, "P_R16"] == 0.0


def test_rejects_cross_group_fixture(wc_format, model):
    a, b = wc_format.groups["A"][0], wc_format.groups["B"][0]
    bad = make_results(
        [(C.WC2026_START, a, b, None, None, C.WORLD_CUP, "Houston", "United States", True)]
    )
    with pytest.raises(DataError, match="not a group-stage pairing"):
        simulate_world_cup(bad, wc_format, model, n_runs=10)
