"""Post-hoc calibration fitted on held-out (out-of-sample) predictions.

1. Totals (optional, off by default): one multiplicative factor per match
   segment, actual goals divided by predicted goals, meant to correct goal-level
   drift in e.g. tight knockout matches. The backtest found no benefit and
   unstable factors between folds, so `TrainingConfig.calibrate_totals` is False.
2. Dixon-Coles rho: the value that maximises the likelihood of the observed
   scorelines given the (scaled) expected goals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar

from wcpred import config as C
from wcpred.models.scoreline import score_matrices

MIN_SEGMENT_MATCHES = 30


@dataclass
class Calibration:
    rho: float = C.DIXON_COLES_RHO
    totals: dict[str, float] = field(default_factory=dict)  # segment -> factor

    def scale(
        self, lam_home: ArrayLike, lam_away: ArrayLike, segments: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the per-segment totals factor (1.0 for unfitted segments)."""
        factor = np.array([self.totals.get(str(s), 1.0) for s in np.atleast_1d(segments)])
        lh = np.atleast_1d(np.asarray(lam_home, dtype=float))
        la = np.atleast_1d(np.asarray(lam_away, dtype=float))
        return lh * factor, la * factor

    def to_dict(self) -> dict[str, Any]:
        return {"rho": self.rho, "totals": dict(self.totals)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Calibration:
        totals = d.get("totals", {})
        return cls(
            rho=float(d.get("rho", C.DIXON_COLES_RHO)),
            totals={str(k): float(v) for k, v in totals.items()},
        )


def fit_totals(
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    segments: np.ndarray,
) -> dict[str, float]:
    """Per-segment actual/predicted goal ratio (segments with too few matches are skipped)."""
    factors: dict[str, float] = {}
    for seg in np.unique(segments):
        mask = segments == seg
        if mask.sum() < MIN_SEGMENT_MATCHES:
            continue
        predicted = lam_home[mask].sum() + lam_away[mask].sum()
        actual = goals_home[mask].sum() + goals_away[mask].sum()
        factors[str(seg)] = round(float(actual / predicted), 4)
    return factors


def scoreline_log_likelihood(
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    rho: float,
) -> float:
    m = score_matrices(lam_home, lam_away, rho)
    gh = np.minimum(goals_home.astype(int), C.MAX_GOALS)
    ga = np.minimum(goals_away.astype(int), C.MAX_GOALS)
    p = m[np.arange(len(gh)), gh, ga]
    return float(np.log(np.maximum(p, 1e-12)).sum())


def fit_rho(
    lam_home: np.ndarray, lam_away: np.ndarray, goals_home: np.ndarray, goals_away: np.ndarray
) -> float:
    """Maximum-likelihood Dixon-Coles rho within RHO_BOUNDS."""
    res = minimize_scalar(
        lambda rho: -scoreline_log_likelihood(lam_home, lam_away, goals_home, goals_away, rho),
        bounds=C.RHO_BOUNDS,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return round(float(res.x), 4)
