"""Unit tests for the lambda -> W/D/L probability math (model-independent)."""
import numpy as np
from wcpred.predict import (
    score_matrix, wdl_from_matrix, knockout_advance, top_scorelines, summarize,
)


def test_score_matrix_normalised():
    M = score_matrix(1.7, 1.1)
    assert abs(M.sum() - 1.0) < 1e-9
    assert (M >= 0).all()


def test_wdl_sums_to_one():
    for lh, la in [(1.5, 1.2), (0.4, 3.3), (2.5, 2.5)]:
        assert abs(sum(wdl_from_matrix(score_matrix(lh, la))) - 1.0) < 1e-9


def test_symmetry_equal_lambdas():
    wh, dr, wa = wdl_from_matrix(score_matrix(1.4, 1.4))
    assert abs(wh - wa) < 1e-9
    assert dr > 0.2  # plenty of draws when evenly matched


def test_monotonic_in_home_strength():
    base = wdl_from_matrix(score_matrix(1.0, 1.4))[0]
    more = wdl_from_matrix(score_matrix(2.2, 1.4))[0]
    assert more > base


def test_expected_goals_match_lambda():
    M = score_matrix(1.9, 0.8)
    i = np.arange(M.shape[0])
    exp_home = (M.sum(axis=1) * i).sum()
    assert abs(exp_home - 1.9) < 0.05  # truncation-limited


def test_knockout_advance():
    a, b = knockout_advance(1.3, 1.3)
    assert abs(a + b - 1.0) < 1e-9
    assert abs(a - 0.5) < 1e-6
    strong, weak = knockout_advance(2.4, 0.7)
    assert strong > 0.8


def test_top_scorelines_sorted():
    sl = top_scorelines(score_matrix(1.6, 1.1), 5)
    probs = [p for _, p in sl]
    assert probs == sorted(probs, reverse=True)


def test_summarize_shape():
    p = summarize("A", "B", 1.8, 0.9)
    assert abs(p.p_home_win + p.p_draw + p.p_away_win - 1.0) < 1e-9
    assert p.exp_home_goals == 1.8


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
