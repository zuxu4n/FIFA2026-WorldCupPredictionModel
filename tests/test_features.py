import numpy as np
import pandas as pd
import pytest

from wcpred import config as C
from wcpred.data import as_of
from wcpred.features import UnknownTeamError, build_features
from wcpred.features.build import form_stats
from wcpred.match import MatchContext
from wcpred.models.goals import sample_weights


@pytest.fixture(scope="module")
def built(results, reference):
    return build_features(results, reference)


def _rows(long: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return long.loc[mask, C.ALL_FEATURES].reset_index(drop=True)


def test_feature_schema(built):
    long, _ = built
    assert len(C.ALL_FEATURES) == len(set(C.ALL_FEATURES))
    assert set(C.ALL_FEATURES) <= set(long.columns)
    assert len(long) == 2 * long["match_id"].nunique()


def test_feature_groups():
    assert "market_value" not in C.DEFAULT_FEATURE_GROUPS  # leaky single snapshot, opt-in only
    assert C.feature_columns(["elo"]) == ["elo_team", "elo_opp", "elo_diff"]
    with pytest.raises(ValueError, match="unknown feature group"):
        C.feature_columns(["nope"])


def test_changing_a_result_never_changes_earlier_features(results, reference, built):
    """Leakage test: features of matches up to day D must not depend on results from day D."""
    long, _ = built
    day = pd.Timestamp("2020-06-15")
    target = results.index[(results["date"] >= day) & results["played"]][0]
    day = results.loc[target, "date"]
    changed = results.copy()
    changed.loc[target, ["home_score", "away_score"]] = [9.0, 0.0]
    long2, _ = build_features(changed, reference)

    upto = long["date"] <= day
    pd.testing.assert_frame_equal(_rows(long, upto), _rows(long2, upto))
    later = long["date"] > day
    assert not _rows(long, later).equals(_rows(long2, later))  # the change does propagate


def test_prediction_rows_match_training_rows(results, reference, built):
    """Train/serve parity: featurising a fixture from a world truncated before its
    date gives exactly the row used for training."""
    long, _ = built
    for mid in results.index[results["played"]][[100, 1500, 3999]]:
        row = results.loc[mid]
        _, world = build_features(as_of(results, row["date"]), reference)
        ctx = MatchContext(
            row["date"],
            row["city"],
            row["country"],
            bool(row["neutral"]),
            row["tournament"],
            bool(row["is_knockout"]),
        )
        for is_home, team, opp in (
            (1, row["home_team"], row["away_team"]),
            (0, row["away_team"], row["home_team"]),
        ):
            got = world.matchup_row(team, opp, ctx, is_home=bool(is_home))
            expected = long[(long["match_id"] == mid) & (long["is_home_persp"] == is_home)]
            np.testing.assert_array_equal(  # exact, not approximate
                [got[c] for c in C.ALL_FEATURES],
                expected[C.ALL_FEATURES].to_numpy(dtype=float)[0],
            )


def test_unplayed_fixtures_have_features_but_no_target(built):
    long, _ = built
    upcoming = long[~long["played"]]
    assert len(upcoming) == 2 * 72  # the synthetic WC2026 group stage
    assert upcoming["goals"].isna().all()
    assert upcoming["elo_team"].notna().all()


def test_form_stats():
    stats = form_stats([(2, 0, 3), (0, 1, 0), (1, 1, 1)])
    assert stats["gf5"] == pytest.approx(1.0)
    assert stats["ga5"] == pytest.approx(2 / 3)
    assert stats["ppg5"] == pytest.approx(4 / 3)
    assert stats["nprior"] == 3
    assert np.isnan(form_stats([])["gf10"])


def test_unknown_team_suggests_close_matches(built):
    _, world = built
    assert world.resolve_team("spain") == "Spain"
    with pytest.raises(UnknownTeamError, match="Spain"):
        world.resolve_team("Spian")


def test_sample_weights_are_anchored_not_wall_clock():
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2014-01-01"]))
    importance = pd.Series([2.0, 2.0])
    anchor = pd.Timestamp("2020-01-01")
    w = sample_weights(dates, importance, anchor, halflife_years=6.0)
    assert w[0] == pytest.approx(2.0)
    assert w[1] == pytest.approx(1.0, rel=1e-3)  # six years older -> half weight
    np.testing.assert_array_equal(w, sample_weights(dates, importance, anchor, 6.0))
