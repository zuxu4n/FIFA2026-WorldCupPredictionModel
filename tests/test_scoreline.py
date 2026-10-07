import numpy as np
import pytest
from scipy.stats import poisson

from wcpred.models.scoreline import (
    advance_probability,
    expected_goals,
    outcome_probs,
    score_matrices,
    score_matrix,
    top_scorelines,
)

LAMBDAS = [(1.5, 1.2), (0.4, 3.3), (2.5, 2.5), (0.05, 0.05), (4.0, 0.3)]


@pytest.mark.parametrize(("lh", "la"), LAMBDAS)
def test_score_matrix_is_a_distribution(lh, la):
    m = score_matrix(lh, la, rho=-0.1)
    assert m.shape == (13, 13)
    assert (m >= 0).all()
    assert m.sum() == pytest.approx(1.0)
    assert outcome_probs(m).sum() == pytest.approx(1.0)


def test_rho_zero_is_independent_poisson():
    m = score_matrix(1.7, 1.1, rho=0.0)
    g = np.arange(13)
    indep = np.outer(poisson.pmf(g, 1.7), poisson.pmf(g, 1.1))
    np.testing.assert_allclose(m, indep / indep.sum())


def test_dixon_coles_only_changes_the_four_low_score_cells():
    lh, la, rho = 1.4, 1.1, -0.12
    ratio = score_matrix(lh, la, rho) / score_matrix(lh, la, 0.0)
    expected_tau = {
        (0, 0): 1 - lh * la * rho,
        (0, 1): 1 + lh * rho,
        (1, 0): 1 + la * rho,
        (1, 1): 1 - rho,
    }
    scale = ratio[5, 5]  # untouched cell: only the renormalisation constant
    for (i, j), tau in expected_tau.items():
        assert ratio[i, j] / scale == pytest.approx(tau)
    untouched = np.ones_like(ratio, dtype=bool)
    for i, j in expected_tau:
        untouched[i, j] = False
    np.testing.assert_allclose(ratio[untouched], scale)


def test_negative_rho_adds_draws():
    assert (
        outcome_probs(score_matrix(1.3, 1.3, -0.1))[1]
        > outcome_probs(score_matrix(1.3, 1.3, 0.0))[1]
    )


def test_vectorised_equals_scalar():
    lh, la = np.array([x for x, _ in LAMBDAS]), np.array([y for _, y in LAMBDAS])
    stacked = score_matrices(lh, la, rho=-0.07)
    for k, (x, y) in enumerate(LAMBDAS):
        np.testing.assert_allclose(stacked[k], score_matrix(x, y, rho=-0.07))
    np.testing.assert_allclose(outcome_probs(stacked)[2], outcome_probs(stacked[2]))


def test_outcomes_symmetric_and_monotone():
    h, d, a = outcome_probs(score_matrix(1.4, 1.4))
    assert h == pytest.approx(a)
    assert d > 0.25
    assert outcome_probs(score_matrix(2.2, 1.4))[0] > outcome_probs(score_matrix(1.0, 1.4))[0]


def test_expected_goals_recovered():
    assert expected_goals(score_matrix(1.9, 0.8, rho=0.0)) == pytest.approx((1.9, 0.8), abs=1e-3)


def test_advance_probability():
    assert advance_probability(1.3, 1.3)[0] == pytest.approx(0.5)
    strong = advance_probability(2.4, 0.7)[0]
    assert strong > 0.8
    assert strong + advance_probability(0.7, 2.4)[0] == pytest.approx(1.0)
    win90 = outcome_probs(score_matrix(2.4, 0.7))[0]
    assert strong > win90  # extra time and penalties can only add to a 90-minute win
    assert advance_probability(np.array([1.0, 2.0]), np.array([1.0, 1.0])).shape == (2,)


def test_top_scorelines_sorted_and_correct():
    m = score_matrix(1.6, 1.1)
    top = top_scorelines(m, 5)
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    assert top[0][1] == pytest.approx(m.max())
