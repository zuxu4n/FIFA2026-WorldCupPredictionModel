import numpy as np
import pytest

from wcpred.evaluation.metrics import (
    accuracy,
    brier_score,
    log_loss,
    outcome_labels,
)
from wcpred.models.calibration import Calibration, fit_rho, fit_totals
from wcpred.models.scoreline import score_matrices


def _sample_scores(lh, la, rho, rng):
    m = score_matrices(lh, la, rho).reshape(len(lh), -1)
    cdf = np.cumsum(m, axis=1)
    k = (cdf < rng.random(len(lh))[:, None]).sum(axis=1)
    return k // 13, k % 13


def test_fit_rho_recovers_the_generating_value():
    rng = np.random.default_rng(0)
    n = 20_000
    lh, la = rng.uniform(0.6, 2.2, n), rng.uniform(0.5, 1.8, n)
    gh, ga = _sample_scores(lh, la, -0.12, rng)
    assert fit_rho(lh, la, gh, ga) == pytest.approx(-0.12, abs=0.03)


def test_fit_totals_ratio_per_segment():
    lh = np.full(100, 1.5)
    la = np.full(100, 1.0)
    segments = np.array(["knockout"] * 50 + ["friendly"] * 40 + ["tiny"] * 10)
    gh = np.where(segments == "knockout", 1.2, 1.5)  # knockout: 80% of predicted goals
    ga = np.where(segments == "knockout", 0.8, 1.0)
    factors = fit_totals(lh, la, gh, ga, segments)
    assert factors["knockout"] == pytest.approx(0.8)
    assert factors["friendly"] == pytest.approx(1.0)
    assert "tiny" not in factors  # too few matches to fit


def test_calibration_scale_and_roundtrip():
    cal = Calibration(rho=-0.08, totals={"knockout": 0.9})
    lh, la = cal.scale([2.0, 2.0], [1.0, 1.0], ["knockout", "friendly"])
    np.testing.assert_allclose(lh, [1.8, 2.0])
    np.testing.assert_allclose(la, [0.9, 1.0])
    assert Calibration.from_dict(cal.to_dict()) == cal


def test_metrics_known_values():
    labels = outcome_labels(np.array([2, 1, 0]), np.array([0, 1, 3]))
    assert labels.tolist() == [0, 1, 2]
    uniform = np.full((3, 3), 1 / 3)
    assert log_loss(uniform, labels) == pytest.approx(np.log(3))
    perfect = np.eye(3)
    assert log_loss(perfect, labels) == pytest.approx(0.0, abs=1e-9)
    assert brier_score(perfect, labels) == pytest.approx(0.0)
    assert brier_score(uniform, labels) == pytest.approx(2 / 3)
    assert accuracy(perfect, labels) == 1.0
