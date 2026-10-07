"""End-to-end tests with the real XGBoost model on the synthetic dataset."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import team_strengths

from wcpred import config as C
from wcpred.features import UnknownTeamError, build_features
from wcpred.match import MatchContext
from wcpred.models import GoalsModel, ModelNotFoundError, TrainingConfig, fit_goals_model
from wcpred.predict import Predictor
from wcpred.simulation import simulate_world_cup

FAST = TrainingConfig(max_rounds=60, early_stopping_rounds=10)
CUTOFF = pd.Timestamp("2026-06-01")


@pytest.fixture(scope="module")
def trained(results, reference):
    long, world = build_features(results, reference)
    return long, Predictor(fit_goals_model(long, CUTOFF, FAST), world)


def test_training_never_sees_results_after_the_cutoff(results, reference):
    cutoff = pd.Timestamp("2024-01-01")
    changed = results.copy()
    later = changed["played"] & (changed["date"] >= cutoff)
    changed.loc[later, ["home_score", "away_score"]] = changed.loc[
        later, ["away_score", "home_score"]
    ].to_numpy()
    long_a, _ = build_features(results, reference)
    long_b, _ = build_features(changed, reference)
    a, b = fit_goals_model(long_a, cutoff, FAST), fit_goals_model(long_b, cutoff, FAST)
    rows = long_a[long_a["played"]].head(500)
    np.testing.assert_array_equal(a.predict_raw(rows), b.predict_raw(rows))
    assert a.calibration == b.calibration


def test_model_learns_team_strength(trained, wc_format):
    _, predictor = trained
    strength = team_strengths(wc_format.teams)
    strongest, weakest = max(strength, key=strength.get), min(strength, key=strength.get)
    ctx = MatchContext(CUTOFF, "Houston", "United States")
    p = predictor.predict(strongest, weakest, ctx)
    assert p.xg_home > p.xg_away
    assert p.p_home_win > 0.5 > p.p_away_win
    assert p.p_home_win + p.p_draw + p.p_away_win == pytest.approx(1.0)
    with pytest.raises(UnknownTeamError):
        predictor.predict("Atlantis", weakest, ctx)


def test_save_load_roundtrip(trained, tmp_path):
    long, predictor = trained
    predictor.model.save(tmp_path)
    loaded = GoalsModel.load(tmp_path)
    rows = long[long["played"]].tail(300)
    np.testing.assert_array_equal(loaded.predict_raw(rows), predictor.model.predict_raw(rows))
    assert loaded.calibration == predictor.model.calibration
    assert loaded.feature_cols == FAST.feature_cols

    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["feature_cols"].append("not_a_feature")
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="retrain"):
        GoalsModel.load(tmp_path)


def test_missing_model_is_reported(tmp_path):
    with pytest.raises(ModelNotFoundError):
        GoalsModel.load(tmp_path / "nothing-here")


def test_simulation_with_trained_model(trained, results, wc_format):
    _, predictor = trained
    table = simulate_world_cup(results, wc_format, predictor, n_runs=300, seed=0).table
    assert table["P_champion"].sum() == pytest.approx(1.0)
    assert table["P_R32"].sum() == pytest.approx(32)
    strength = team_strengths(wc_format.teams)
    assert strength[table.iloc[0]["team"]] > np.median(list(strength.values()))


def test_default_feature_set_excludes_market_value():
    assert not set(C.FEATURE_GROUPS["market_value"]) & set(TrainingConfig().feature_cols)
