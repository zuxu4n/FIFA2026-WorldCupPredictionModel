"""Turn a pair of expected-goals (lambda_home, lambda_away) into match outcome
probabilities via a Dixon-Coles-adjusted bivariate Poisson score matrix, and
wire the trained model + FeatureWorld together to predict real fixtures.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import config as C
from .features import FEATURE_COLS


# --------------------------------------------------------------------------
# Pure probability math
# --------------------------------------------------------------------------
def _dc_tau(max_goals: int, lam: float, mu: float, rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction matrix (1.0 everywhere except 4 cells)."""
    tau = np.ones((max_goals + 1, max_goals + 1))
    tau[0, 0] = 1.0 - lam * mu * rho
    tau[0, 1] = 1.0 + lam * rho
    tau[1, 0] = 1.0 + mu * rho
    tau[1, 1] = 1.0 - rho
    return np.clip(tau, 1e-6, None)


def _resolve_rho(rho):
    if rho is not None:
        return rho
    from . import calibrate as _cal
    return _cal.fitted_rho()


def score_matrix(lam_home: float, lam_away: float,
                 max_goals: int = C.MAX_GOALS,
                 rho: float | None = None) -> np.ndarray:
    """P[i, j] = Prob(home scores i, away scores j).

    rho=None uses the fitted Dixon-Coles rho (models/totals_calibrator.json)
    with the config value as fallback.
    """
    rho = _resolve_rho(rho)
    i = np.arange(max_goals + 1)
    ph = poisson.pmf(i, max(lam_home, 1e-6))
    pa = poisson.pmf(i, max(lam_away, 1e-6))
    M = np.outer(ph, pa)
    M *= _dc_tau(max_goals, lam_home, lam_away, rho)
    s = M.sum()
    return M / s if s > 0 else M


def score_matrix_incl_et(lam_home: float, lam_away: float,
                         max_goals: int = C.MAX_GOALS,
                         rho: float | None = None) -> np.ndarray:
    """Final-score distribution INCLUDING extra time: draws after 90' get a
    30-minute continuation at 1/3 match rate convolved on top (still-level
    scores then go to penalties, which don't add goals)."""
    M = score_matrix(lam_home, lam_away, max_goals, rho)
    E = score_matrix(lam_home / 3.0, lam_away / 3.0, max_goals, rho)
    G = 2 * max_goals + 1
    F = np.zeros((G, G))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i != j:
                F[i, j] += M[i, j]
            else:
                F[i:i + max_goals + 1, j:j + max_goals + 1] += M[i, j] * E
    return F


def wdl_from_matrix(M: np.ndarray) -> tuple[float, float, float]:
    """Return (home_win, draw, away_win) probabilities."""
    home = float(np.tril(M, -1).sum())   # i > j
    draw = float(np.trace(M))
    away = float(np.triu(M, 1).sum())    # i < j
    return home, draw, away


def top_scorelines(M: np.ndarray, k: int = 5):
    idx = np.dstack(np.unravel_index(np.argsort(M.ravel())[::-1], M.shape))[0]
    return [((int(i), int(j)), float(M[i, j])) for i, j in idx[:k]]


def knockout_advance(lam_home: float, lam_away: float,
                     max_goals: int = C.MAX_GOALS,
                     rho: float | None = None,
                     pen_home: float = 0.5) -> tuple[float, float]:
    """Probability each side advances in a knockout (90' -> 30' ET -> penalties).

    Returns (P(home advances), P(away advances)), which sum to 1.
    """
    M = score_matrix(lam_home, lam_away, max_goals, rho)
    w_h, draw, w_a = wdl_from_matrix(M)
    # extra time: ~1/3 of a match's goals
    Met = score_matrix(lam_home / 3.0, lam_away / 3.0, max_goals, rho)
    eh, ed, ea = wdl_from_matrix(Met)
    adv_home = w_h + draw * (eh + ed * pen_home)
    return adv_home, 1.0 - adv_home


@dataclass
class MatchPrediction:
    home: str
    away: str
    lam_home: float
    lam_away: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    exp_home_goals: float
    exp_away_goals: float
    top_scorelines: list
    p_home_advance: float
    p_away_advance: float
    context: dict

    def as_row(self) -> dict:
        ml = self.top_scorelines[0][0]
        return {
            "home": self.home, "away": self.away,
            "xg_home": round(self.lam_home, 2), "xg_away": round(self.lam_away, 2),
            "p_home_win": round(self.p_home_win, 3),
            "p_draw": round(self.p_draw, 3),
            "p_away_win": round(self.p_away_win, 3),
            "most_likely": f"{ml[0]}-{ml[1]}",
            "p_home_adv": round(self.p_home_advance, 3),
        }


def summarize(home: str, away: str, lam_home: float, lam_away: float,
              context: dict | None = None, calibrate: bool = True) -> MatchPrediction:
    M = score_matrix(lam_home, lam_away)
    wh, dr, wa = wdl_from_matrix(M)
    if calibrate:
        from . import calibrate as _cal
        cal = _cal.load()
        if cal is not None:
            wh, dr, wa = _cal.apply_one(cal, wh, dr, wa)
    adv_h, adv_a = knockout_advance(lam_home, lam_away)
    return MatchPrediction(
        home=home, away=away, lam_home=float(lam_home), lam_away=float(lam_away),
        p_home_win=wh, p_draw=dr, p_away_win=wa,
        exp_home_goals=float(lam_home), exp_away_goals=float(lam_away),
        top_scorelines=top_scorelines(M, 5),
        p_home_advance=adv_h, p_away_advance=adv_a,
        context=context or {},
    )


# --------------------------------------------------------------------------
# Model-driven expected goals
# --------------------------------------------------------------------------
def _feat_matrix(rows: list[dict]) -> "np.ndarray":
    df = pd.DataFrame(rows)
    return df[FEATURE_COLS].to_numpy(dtype=float)


def predict_lambdas(booster, world, home: str, away: str, *,
                    city: str, country: str, neutral: bool,
                    importance: float = C.TOURNAMENT_IMPORTANCE[1][1],
                    asof: pd.Timestamp | None = None) -> tuple[float, float]:
    """Predict (lambda_home, lambda_away) for a fixture using the model."""
    import xgboost as xgb
    asof = asof or pd.Timestamp("today").normalize()
    row_home = world.matchup_row(home, away, city=city, country=country,
                                 neutral=neutral, importance=importance,
                                 asof=asof, is_home=True)
    row_away = world.matchup_row(away, home, city=city, country=country,
                                 neutral=neutral, importance=importance,
                                 asof=asof, is_home=False)
    X = _feat_matrix([row_home, row_away])
    dm = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    pred = booster.predict(dm)
    # fitted per-segment lambda shrinkage (identity if not calibrated)
    from . import calibrate as _cal
    return _cal.scale_lambdas(float(pred[0]), float(pred[1]),
                              importance, row_home["is_knockout"])


def predict_fixture(booster, world, home: str, away: str, *,
                    city: str, country: str, neutral: bool,
                    importance: float = C.TOURNAMENT_IMPORTANCE[1][1],
                    asof: pd.Timestamp | None = None) -> MatchPrediction:
    lh, la = predict_lambdas(booster, world, home, away, city=city, country=country,
                             neutral=neutral, importance=importance, asof=asof)
    return summarize(home, away, lh, la,
                     context={"city": city, "country": country, "neutral": neutral})
