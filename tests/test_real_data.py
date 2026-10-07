"""Checks against the real (pinned) dataset. Skipped unless `wcpred fetch-data` has run."""

import pytest
from conftest import StubModel, team_strengths

from wcpred import config as C
from wcpred.data import load_results, load_shootouts
from wcpred.data.download import sha256
from wcpred.simulation import simulate_world_cup

pytestmark = pytest.mark.skipif(
    not C.PATHS.results_csv.exists()
    or sha256(C.PATHS.results_csv.read_bytes()) != C.RESULTS_SHA256["results.csv"],
    reason="pinned dataset not downloaded (run `wcpred fetch-data`)",
)


@pytest.fixture(scope="module")
def real():
    return load_results()


def test_world_cup_2026_structure(real):
    wc = real[
        (real["tournament"] == C.WORLD_CUP)
        & (real["date"] >= C.WC2026_START)
        & (real["date"] <= C.WC2026_END)
    ]
    assert len(wc) == 104 and wc["played"].all()
    assert wc["is_knockout"].sum() == 32
    assert wc.loc[wc["is_knockout"], "date"].min() == C.WC2026_KNOCKOUT_START


def test_knockout_flag_only_in_final_tournaments(real):
    assert (
        not real.loc[real["is_knockout"], "tournament"]
        .str.contains("qualification|Nations League")
        .any()
    )


def test_simulation_reproduces_the_real_tournament(real, wc_format):
    """With every result known, group ranking + FIFA's third-place table must
    rebuild the real bracket, so all knockout results lock in."""
    model = StubModel(team_strengths(wc_format.teams))
    table = simulate_world_cup(
        real, wc_format, model, n_runs=200, seed=0, shootouts=load_shootouts()
    ).table.set_index("team")
    assert table.loc["Spain", "P_champion"] == 1.0
    assert table.loc["Argentina", "P_Final"] == 1.0
    r32 = real[
        (real["tournament"] == C.WORLD_CUP)
        & (real["date"] >= C.WC2026_KNOCKOUT_START)
        & (real["date"] < "2026-07-04")
    ]
    qualified = set(r32["home_team"]) | set(r32["away_team"])
    assert set(table.index[table["P_R32"] == 1.0]) == qualified
    assert model.calls == 0  # nothing left to predict
