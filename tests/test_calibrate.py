"""Tests for the calibration layers and incl-ET score distribution."""
import numpy as np
import pandas as pd
import pytest
from wcpred.calibrate import (segment_of, scale_lambdas, load_totals,
                              _cal_features, apply_probs)
from wcpred.predict import score_matrix, score_matrix_incl_et, wdl_from_matrix


def test_segment_classification():
    assert segment_of(4.0, 1.0) == "knockout"
    assert segment_of(4.0, 0.0) == "tournament_group"
    assert segment_of(2.5, 0.0) == "qualifier"
    assert segment_of(1.0, 0.0) == "other"


def test_scale_lambdas_uses_fitted_factors():
    tot = load_totals()
    if tot is None:
        pytest.skip("totals calibrator not fitted")
    lh, la = scale_lambdas(2.0, 1.0, 4.0, 0.0)   # tournament_group segment
    f = tot["factors"]["tournament_group"]
    assert abs(lh - 2.0 * f) < 1e-12 and abs(la - 1.0 * f) < 1e-12


def test_cal_features_shape():
    P = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    X = _cal_features(P)
    assert X.shape == (2, 4)
    assert np.allclose(X[:, 3], np.abs(X[:, 0] - X[:, 2]))


def test_apply_probs_normalised_both_versions():
    P = np.array([0.6, 0.25, 0.15])
    for ncoef in (3, 4):
        cal = {"coef": np.eye(3, ncoef), "intercept": np.zeros(3)}
        out = apply_probs(cal, P)
        assert abs(out.sum() - 1.0) < 1e-9


def test_incl_et_distribution():
    F = score_matrix_incl_et(1.6, 1.1)
    assert abs(F.sum() - 1.0) < 1e-6
    # ET adds goals only from 90' draws: P(total >= k) never decreases
    M = score_matrix(1.6, 1.1)
    G = M.shape[0]
    p90_over25 = sum(M[i, j] for i in range(G) for j in range(G) if i + j > 2.5)
    Ge = F.shape[0]
    pet_over25 = sum(F[i, j] for i in range(Ge) for j in range(Ge) if i + j > 2.5)
    assert pet_over25 >= p90_over25 - 1e-9


def test_bracket_played_knockouts_shape():
    from wcpred.bracket import played_knockouts
    df = pd.DataFrame({
        "tournament": ["FIFA World Cup"] * 2,
        "date": pd.to_datetime(["2026-06-29", "2026-06-30"]),
        "home_team": ["A", "C"], "away_team": ["B", "D"],
        "home_score": [2.0, 1.0], "away_score": [0.0, 1.0],
        "played": [True, True],
    })
    lock = played_knockouts(df)
    assert lock[frozenset(("A", "B"))][0] == "A"
    # drawn KO game without shootout record is left to the simulator
    assert frozenset(("C", "D")) not in lock
