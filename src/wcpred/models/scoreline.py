"""From expected goals to match outcome probabilities.

Given each side's expected goals (lambda_home, lambda_away), the score matrix is

    P[i, j] = Poisson(i; lambda_home) * Poisson(j; lambda_away) * tau(i, j)

where tau is the Dixon-Coles (1997) correction. Independent Poisson draws
under-predict 0-0 and 1-1 and over-predict 1-0 and 0-1 in real football; tau
adjusts exactly those four cells with a single parameter rho (rho < 0 adds
draws) and leaves the totals of each margin otherwise intact.

All functions accept scalars or 1-D arrays of lambdas; array inputs return a
stack of matrices so whole tournaments are evaluated in one call.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import poisson

from wcpred import config as C

_MIN_LAMBDA = 1e-6


def dixon_coles_tau(
    lam_home: ArrayLike, lam_away: ArrayLike, rho: float, max_goals: int = C.MAX_GOALS
) -> np.ndarray:
    """Correction factors, shape (n, G, G): 1 everywhere except the four low-score cells."""
    lh = np.atleast_1d(np.asarray(lam_home, dtype=float))
    la = np.atleast_1d(np.asarray(lam_away, dtype=float))
    tau = np.ones((lh.size, max_goals + 1, max_goals + 1))
    tau[:, 0, 0] = 1.0 - lh * la * rho
    tau[:, 0, 1] = 1.0 + lh * rho
    tau[:, 1, 0] = 1.0 + la * rho
    tau[:, 1, 1] = 1.0 - rho
    # extreme rho/lambda combinations could make a cell negative
    return np.clip(tau, 1e-6, None)


def score_matrices(
    lam_home: ArrayLike,
    lam_away: ArrayLike,
    rho: float = C.DIXON_COLES_RHO,
    max_goals: int = C.MAX_GOALS,
) -> np.ndarray:
    """Normalised score distributions, shape (n, G, G); P[k, i, j] = P(home i, away j)."""
    lh = np.maximum(np.atleast_1d(np.asarray(lam_home, dtype=float)), _MIN_LAMBDA)
    la = np.maximum(np.atleast_1d(np.asarray(lam_away, dtype=float)), _MIN_LAMBDA)
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals[None, :], lh[:, None])
    pa = poisson.pmf(goals[None, :], la[:, None])
    m = ph[:, :, None] * pa[:, None, :] * dixon_coles_tau(lh, la, rho, max_goals)
    # renormalise: truncation at max_goals drops a tiny amount of mass
    return m / m.sum(axis=(1, 2), keepdims=True)


def score_matrix(
    lam_home: float, lam_away: float, rho: float = C.DIXON_COLES_RHO, max_goals: int = C.MAX_GOALS
) -> np.ndarray:
    """Single (G, G) score distribution."""
    return score_matrices(lam_home, lam_away, rho, max_goals)[0]


def outcome_probs(m: np.ndarray) -> np.ndarray:
    """[P(home win), P(draw), P(away win)] for a (G, G) or (n, G, G) score matrix."""
    stack = m[None] if m.ndim == 2 else m
    home = np.tril(np.ones(stack.shape[1:]), -1)  # i > j
    away = np.triu(np.ones(stack.shape[1:]), 1)
    probs = np.stack(
        [
            (stack * home).sum(axis=(1, 2)),
            np.trace(stack, axis1=1, axis2=2),
            (stack * away).sum(axis=(1, 2)),
        ],
        axis=-1,
    )
    return probs[0] if m.ndim == 2 else probs


def advance_probability(
    lam_home: ArrayLike,
    lam_away: ArrayLike,
    rho: float = C.DIXON_COLES_RHO,
    max_goals: int = C.MAX_GOALS,
) -> np.ndarray:
    """P(home side advances) in a knockout tie: 90 minutes, extra time, penalties.

    Extra time is modelled as a third of a match at the same scoring rates;
    penalties are a coin flip (PENALTY_HOME_WIN_PROB).
    """
    lh = np.atleast_1d(np.asarray(lam_home, dtype=float))
    la = np.atleast_1d(np.asarray(lam_away, dtype=float))
    regular = outcome_probs(score_matrices(lh, la, rho, max_goals))
    f = C.EXTRA_TIME_FRACTION
    extra = outcome_probs(score_matrices(lh * f, la * f, rho, max_goals))
    after_draw = extra[:, 0] + extra[:, 1] * C.PENALTY_HOME_WIN_PROB
    return regular[:, 0] + regular[:, 1] * after_draw


def top_scorelines(m: np.ndarray, k: int = 5) -> list[tuple[tuple[int, int], float]]:
    """The k most likely (home goals, away goals) scorelines of a (G, G) matrix."""
    flat = np.argsort(m.ravel(), kind="stable")[::-1][:k]
    return [
        ((int(i), int(j)), float(m[i, j]))
        for i, j in zip(*np.unravel_index(flat, m.shape), strict=True)
    ]


def expected_goals(m: np.ndarray) -> tuple[float, float]:
    """Mean home and away goals of a (G, G) score matrix."""
    g = np.arange(m.shape[0])
    return float((m.sum(axis=1) * g).sum()), float((m.sum(axis=0) * g).sum())
